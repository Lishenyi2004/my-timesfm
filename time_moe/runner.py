import os
import math
import random
import inspect
import json
import re
from functools import reduce
from operator import mul

import torch
from torch.utils.data import random_split

from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.models.modeling_timesfm_2p5 import TimesFM2p5ForTraining
from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
from time_moe.utils.dist_util import get_world_size
from time_moe.utils.log_util import logger, log_in_local_rank_0


class TimeMoeRunner:
    def __init__(
            self,
            model_path: str = None,
            output_path: str = 'logs/time_moe2',
            seed: int = 9899,
            model_family: str = 'timesfm_2p5',
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.seed = seed
        self.model_family = model_family

    def load_model(self, model_path: str = None, from_scatch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path

        model_family = kwargs.pop('model_family', None) or self.model_family

        if model_family == 'timesfm_2p5':
            if from_scatch:
                model = TimesFM2p5ForTraining(
                    torch_dtype=kwargs.get('torch_dtype'),
                    use_quantile_loss=kwargs.get('use_quantile_loss', True),
                    quantile_loss_weight=kwargs.get('quantile_loss_weight', 1.0),
                    timesfm_num_layers=kwargs.get('timesfm_num_layers'),
                    timesfm_num_heads=kwargs.get('timesfm_num_heads'),
                    timesfm_model_dims=kwargs.get('timesfm_model_dims'),
                    use_revin_norm=kwargs.get('use_revin_norm', True),
                    use_revin_denorm=kwargs.get('use_revin_denorm', True),
                )
            else:
                model = TimesFM2p5ForTraining.from_pretrained(
                    model_path=model_path,
                    torch_dtype=kwargs.get('torch_dtype'),
                    use_quantile_loss=kwargs.get('use_quantile_loss', True),
                    quantile_loss_weight=kwargs.get('quantile_loss_weight', 1.0),
                    timesfm_num_layers=kwargs.get('timesfm_num_layers'),
                    timesfm_num_heads=kwargs.get('timesfm_num_heads'),
                    timesfm_model_dims=kwargs.get('timesfm_model_dims'),
                    use_revin_norm=kwargs.get('use_revin_norm', True),
                    use_revin_denorm=kwargs.get('use_revin_denorm', True),
                )
            return model

        if model_family != 'time_moe':
            raise ValueError(f'Unknown model_family: {model_family}')

        # attn = kwargs.pop('attn_implementation', None)
        # if attn is None:
        #     attn = 'eager'
        # elif attn == 'auto':
        #     # try to use flash-attention
        #     try:
        #         from flash_attn import flash_attn_func, flash_attn_varlen_func
        #         from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
        #         attn = 'flash_attention_2'
        #     except:
        #         log_in_local_rank_0('Flash attention import failed, switching to eager attention.', type='warn')
        #         attn = 'eager'

        # if attn == 'eager':
        #     log_in_local_rank_0('Use Eager Attention')
        # elif attn == 'flash_attention_2':
        #     log_in_local_rank_0('Use Flash Attention 2')
        # else:
        #     raise ValueError(f'Unknown attention method: {attn}')
        # kwargs['attn_implementation'] = attn

        # if from_scatch:
        #     config = TimeMoeConfig.from_pretrained(model_path, _attn_implementation=attn)
        #     model = TimeMoeForPrediction(config)
        # else:
        #     model = TimeMoeForPrediction.from_pretrained(model_path, **kwargs)
        # return model

    def train_model(self, from_scratch: bool = False, **kwargs):
        setup_seed(self.seed)

        train_config = kwargs

        num_devices = get_world_size()
        log_in_local_rank_0(
            'Distributed env:',
            f'RANK={os.getenv("RANK")}',
            f'LOCAL_RANK={os.getenv("LOCAL_RANK")}',
            f'WORLD_SIZE={os.getenv("WORLD_SIZE")}',
            f'LOCAL_WORLD_SIZE={os.getenv("LOCAL_WORLD_SIZE")}',
            f'Detected num_devices={num_devices}',
        )

        global_batch_size = train_config.get('global_batch_size', None)
        micro_batch_size = train_config.get('micro_batch_size', None)

        if global_batch_size is None and micro_batch_size is None:
            raise ValueError('Must set at lease one argument: "global_batch_size" or "micro_batch_size"')
        elif global_batch_size is None:
            gradient_accumulation_steps = 1
            global_batch_size = micro_batch_size * num_devices
        elif micro_batch_size is None:
            micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = 1
        else:
            if micro_batch_size * num_devices > global_batch_size:
                if num_devices > global_batch_size:
                    micro_batch_size = 1
                    global_batch_size = num_devices
                else:
                    micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = math.ceil(global_batch_size / num_devices / micro_batch_size)
            global_batch_size = int(gradient_accumulation_steps * num_devices * micro_batch_size)

        if ('train_steps' in train_config
                and train_config['train_steps'] is not None
                and train_config['train_steps'] > 0):
            train_steps = int(train_config["train_steps"])
            num_train_epochs = -1
        else:
            train_steps = -1
            num_train_epochs = _safe_float(train_config.get("num_train_epochs", 1))

        precision = train_config.get('precision', 'bf16')
        if precision not in ['bf16', 'fp16', 'fp32']:
            log_in_local_rank_0(f'Precision {precision} is not set, use fp32 default!', type='warn')
            precision = 'fp32'

        if precision == 'bf16':
            torch_dtype = torch.bfloat16
        elif precision == 'fp16':
            # use fp32 to load model but uses fp15 to train model
            torch_dtype = torch.float32
        elif precision == 'fp32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f'Unsupported precision {precision}')

        log_in_local_rank_0(f'Set global_batch_size to {global_batch_size}')
        log_in_local_rank_0(f'Set micro_batch_size to {micro_batch_size}')
        log_in_local_rank_0(f'Set gradient_accumulation_steps to {gradient_accumulation_steps}')
        log_in_local_rank_0(f'Set precision to {precision}')
        log_in_local_rank_0(f'Set normalization to {train_config["normalization_method"]}')

        eval_strategy_arg_name = 'evaluation_strategy'
        if 'eval_strategy' in inspect.signature(TimeMoETrainingArguments.__init__).parameters:
            eval_strategy_arg_name = 'eval_strategy'

        ddp_find_unused_parameters = train_config.get('ddp_find_unused_parameters')
        if ddp_find_unused_parameters is None:
            ddp_find_unused_parameters = (train_config.get('model_family') or self.model_family) == 'timesfm_2p5'
        log_in_local_rank_0(f'Set ddp_find_unused_parameters to {bool(ddp_find_unused_parameters)}')

        # Validation split and effective strategies.
        validation_split_ratio = float(train_config.get('validation_split_ratio', 0.01))
        enable_validation_split = bool(train_config.get('enable_validation_split', True))

        eval_strategy = train_config.get("evaluation_strategy", 'no')
        save_strategy = train_config.get("save_strategy", "no")
        eval_steps = _safe_float(train_config.get("eval_steps", None))
        save_steps = _safe_float(train_config.get("save_steps", None))
        load_best_model_at_end = bool(train_config.get('load_best_model_at_end', True))
        metric_for_best_model = train_config.get('metric_for_best_model', 'eval_loss')
        greater_is_better = train_config.get('greater_is_better')
        if greater_is_better is None and metric_for_best_model.endswith('loss'):
            greater_is_better = False

        if enable_validation_split and validation_split_ratio <= 0:
            enable_validation_split = False

        training_args_kwargs = dict(
            output_dir=self.output_path,
            num_train_epochs=num_train_epochs,
            # use_cpu=True,
            max_steps=train_steps,
            logging_strategy=train_config.get('logging_strategy', 'steps'),
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            learning_rate=float(train_config.get("learning_rate", 1e-5)),
            min_learning_rate=float(train_config.get("min_learning_rate", 0)),
            adam_beta1=float(train_config.get("adam_beta1", 0.9)),
            adam_beta2=float(train_config.get("adam_beta2", 0.95)),
            adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
            lr_scheduler_type=train_config.get("lr_scheduler_type", 'constant'),
            cosine_num_cycles=float(train_config.get("cosine_num_cycles", 0.5)),
            warmup_ratio=float(train_config.get("warmup_ratio") or 0.0),
            warmup_steps=int(train_config.get("warmup_steps", 0)),
            weight_decay=float(train_config.get("weight_decay", 0.1)),
            per_device_train_batch_size=int(micro_batch_size),
            per_device_eval_batch_size=int(micro_batch_size * 2),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            gradient_checkpointing=train_config.get("gradient_checkpointing", False),
            bf16=True if precision == 'bf16' else False,
            fp16=True if precision == 'fp16' else False,
            deepspeed=train_config.get("deepspeed"),
            push_to_hub=False,
            logging_first_step=True,
            log_on_each_node=False,
            logging_steps=int(train_config.get('logging_steps', 1)),
            report_to=train_config.get('report_to', ['tensorboard']),
            seed=self.seed,
            data_seed=self.seed,
            max_grad_norm=train_config.get('max_grad_norm', 1.0),
            optim=train_config.get('optim', 'adamw_torch'),
            torch_compile=train_config.get('torch_compile', False),
            dataloader_num_workers=train_config.get('dataloader_num_workers') or 2,
            ddp_timeout=int(train_config.get('ddp_timeout', 1800)),
            ddp_find_unused_parameters=bool(ddp_find_unused_parameters),

            logging_dir=os.path.join(self.output_path, 'tb_logs'),
            save_only_model=train_config.get('save_only_model', True),
            save_total_limit=train_config.get('save_total_limit'),
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
        )
        training_args_kwargs[eval_strategy_arg_name] = eval_strategy

        model_path = train_config.pop('model_path', None) or self.model_path
        model_family = train_config.get('model_family') or self.model_family
        if model_path is not None:
            model = self.load_model(
                model_path=model_path,
                from_scatch=from_scratch,
                torch_dtype=torch_dtype,
                model_family="timesfm_2p5",
                use_quantile_loss=bool(train_config.get('use_quantile_loss', True)),
                quantile_loss_weight=float(train_config.get('quantile_loss_weight', 1.0)),
                timesfm_num_layers=train_config.get('timesfm_num_layers'),
                timesfm_num_heads=train_config.get('timesfm_num_heads'),
                timesfm_model_dims=train_config.get('timesfm_model_dims'),
                use_revin_norm=bool(train_config.get('use_revin_norm', True)),
                use_revin_denorm=bool(train_config.get('use_revin_denorm', True)),
                attn_implementation=train_config.get('attn_implementation', 'eager'),
            )
            log_in_local_rank_0(f'Load model parameters from: {model_path}')
        else:
            raise ValueError('Model path is None')

        num_total_params = 0
        for p in model.parameters():
            num_total_params += reduce(mul, p.shape)

        # print statistics info
        log_in_local_rank_0(train_config)
        log_in_local_rank_0(model.config)
        log_in_local_rank_0(f'Number of the model parameters: {length_to_str(num_total_params)}')

        if train_steps > 0:
            total_train_tokens = train_steps * global_batch_size * train_config['max_length']
            log_in_local_rank_0(f'Tokens will consume: {length_to_str(total_train_tokens)}')

        # Training
        train_ds, eval_ds = self.get_train_val_datasets(
            train_config["data_path"],
            max_length=train_config["max_length"],
            stride=train_config["stride"],
            normalization_method=train_config["normalization_method"],
            validation_split_ratio=validation_split_ratio,
            enable_validation_split=enable_validation_split,
        )

        if eval_ds is not None:
            if eval_strategy == 'no':
                eval_strategy = 'steps'
            if save_strategy == 'no':
                save_strategy = eval_strategy
            if eval_strategy == 'steps' and eval_steps is None:
                eval_steps = max(1, int(train_config.get('logging_steps', 100)))
            if save_strategy == 'steps' and save_steps is None:
                save_steps = eval_steps
            if load_best_model_at_end and eval_strategy == 'steps' and save_strategy == 'steps':
                # Keep compatibility with Transformers constraints.
                save_steps = eval_steps
            training_args_kwargs[eval_strategy_arg_name] = eval_strategy
            training_args_kwargs['save_strategy'] = save_strategy
            training_args_kwargs['eval_steps'] = eval_steps
            training_args_kwargs['save_steps'] = save_steps
            log_in_local_rank_0(
                f'Validation enabled: eval_strategy={eval_strategy}, save_strategy={save_strategy}, '
                f'eval_steps={eval_steps}, save_steps={save_steps}, '
                f'load_best_model_at_end={bool(load_best_model_at_end)}'
            )
        else:
            training_args_kwargs[eval_strategy_arg_name] = 'no'
            training_args_kwargs['save_strategy'] = save_strategy
            training_args_kwargs['load_best_model_at_end'] = False
            log_in_local_rank_0('Validation split disabled or too small; skip eval and best-model selection.', type='warn')

        training_args = TimeMoETrainingArguments(**training_args_kwargs)
        log_in_local_rank_0(training_args)

        resume_from_checkpoint = train_config.get('resume_from_checkpoint')
        if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == 'auto':
            from transformers.trainer_utils import get_last_checkpoint
            resume_from_checkpoint = get_last_checkpoint(self.output_path)
            if resume_from_checkpoint is None:
                log_in_local_rank_0('No checkpoint found for auto resume, start from scratch/current model.', type='warn')
            else:
                log_in_local_rank_0(f'Auto resume from checkpoint: {resume_from_checkpoint}')

        if isinstance(resume_from_checkpoint, str) and len(resume_from_checkpoint) > 0 and not os.path.exists(resume_from_checkpoint):
            log_in_local_rank_0(
                f'Resume checkpoint path does not exist: {resume_from_checkpoint}. Start without resume.',
                type='warn',
            )
            resume_from_checkpoint = None

        trainer = TimeMoeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        best_model_checkpoint = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric
        best_step = None
        if isinstance(best_model_checkpoint, str):
            matched = re.search(r'checkpoint-(\d+)$', best_model_checkpoint)
            if matched:
                best_step = int(matched.group(1))

        explicit_best_dir = None
        if best_model_checkpoint:
            explicit_best_dir = os.path.join(self.output_path, 'best_model')
            trainer.save_model(explicit_best_dir)
            log_in_local_rank_0(
                f'Best model saved to {explicit_best_dir}, '
                f'best_checkpoint={best_model_checkpoint}, best_metric={best_metric}, best_step={best_step}'
            )

        best_model_info = {
            'best_model_checkpoint': best_model_checkpoint,
            'best_metric': best_metric,
            'best_step': best_step,
            'metric_for_best_model': training_args.metric_for_best_model,
            'greater_is_better': training_args.greater_is_better,
            'explicit_best_model_dir': explicit_best_dir,
        }
        with open(os.path.join(self.output_path, 'best_model_info.json'), 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, ensure_ascii=False, indent=2)

        trainer.save_model(self.output_path)
        log_in_local_rank_0(f'Saving model to {self.output_path}')

        return trainer.model

    def get_train_dataset(self, data_path, max_length, stride, normalization_method):
        log_in_local_rank_0('Loading dataset...')
        dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
        log_in_local_rank_0('Processing dataset to fixed-size sub-sequences...')
        window_dataset = TimeMoEWindowDataset(dataset, context_length=max_length, prediction_length=0, stride=stride, shuffle=False)
        return window_dataset

    def get_train_val_datasets(
            self,
            data_path,
            max_length,
            stride,
            normalization_method,
            validation_split_ratio: float = 0.01,
            enable_validation_split: bool = True,
    ):
        log_in_local_rank_0('Loading dataset...')
        dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
        log_in_local_rank_0('Processing dataset to fixed-size sub-sequences...')
        window_dataset = TimeMoEWindowDataset(
            dataset,
            context_length=max_length,
            prediction_length=0,
            stride=stride,
            shuffle=False,
        )

        if not enable_validation_split:
            return window_dataset, None

        total_size = len(window_dataset)
        if total_size < 2:
            log_in_local_rank_0('Dataset too small for validation split (<2 windows).', type='warn')
            return window_dataset, None

        val_size = max(1, int(total_size * validation_split_ratio))
        val_size = min(val_size, total_size - 1)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, eval_dataset = random_split(window_dataset, [train_size, val_size], generator=generator)
        log_in_local_rank_0(
            f'Split dataset: train={train_size}, val={val_size}, val_ratio={val_size / total_size:.4f}'
        )
        return train_dataset, eval_dataset


def setup_seed(seed: int = 9899):
    """
    Setup seed for all known operations.

    Args:
        seed (int): seed number.

    Returns:

    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def length_to_str(length):
    if length >= 1e12:
        return f'{length / 1e12:.3f}T'
    if length >= 1e9:
        return f'{length / 1e9:.3f}B'
    elif length >= 1e6:
        return f'{length / 1e6:.3f}M'
    else:
        return f'{length / 1e3:.3f}K'


def _safe_float(number):
    if number is None:
        return None
    else:
        return float(number)