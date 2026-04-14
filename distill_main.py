import argparse
import math
import os
import re
from functools import reduce
from operator import mul

import torch

from time_moe.models.modeling_timesfm_2p5 import TimesFM2p5ForTraining
from time_moe.runner import TimeMoeRunner, setup_seed
from time_moe.trainer.distill_trainer import TimesFMDistillTrainer
from time_moe.trainer.hf_trainer import TimeMoETrainingArguments
from time_moe.utils.dist_util import get_world_size
from time_moe.utils.log_util import log_in_local_rank_0


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _compute_batch_settings(global_batch_size, micro_batch_size, num_devices):
    if global_batch_size is None and micro_batch_size is None:
        raise ValueError('Must set at least one of global_batch_size or micro_batch_size.')
    if global_batch_size is None:
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

    return int(global_batch_size), int(micro_batch_size), int(gradient_accumulation_steps)


def _copy_matching_weights(source_model: torch.nn.Module, target_model: torch.nn.Module):
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    matched = {}

    for key, value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            continue
        if source_value.shape == value.shape:
            matched[key] = source_value

    target_state.update(matched)
    missing, unexpected = target_model.load_state_dict(target_state, strict=False)
    return len(matched), missing, unexpected


def _truncate_student_layers(student: TimesFM2p5ForTraining, student_num_layers: int):
    full_layers = len(student.backbone.stacked_xf)
    if student_num_layers <= 0:
        raise ValueError(f'student_num_layers must be > 0, got {student_num_layers}')
    if student_num_layers > full_layers:
        raise ValueError(
            f'student_num_layers={student_num_layers} cannot exceed base layers={full_layers}'
        )
    if student_num_layers < full_layers:
        student.backbone.stacked_xf = torch.nn.ModuleList(
            list(student.backbone.stacked_xf)[:student_num_layers]
        )
        student.backbone.x = student_num_layers


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, required=True)
    parser.add_argument(
        '--teacher_model_path',
        type=str,
        default='/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm-2.5-200m-pytorch',
        help='Path to teacher checkpoint (directory with model.safetensors).',
    )
    parser.add_argument('--output_path', '-o', type=str, default='logs/distill_timesfm2p5_student')
    parser.add_argument('--seed', type=int, default=9899)

    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--normalization_method', choices=['none', 'zero', 'max'], default='none')
    parser.add_argument('--max_train_sequences', type=int, default=None)

    parser.add_argument('--student_num_layers', type=int, default=8)
    parser.add_argument(
        '--init_student_from_teacher',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Copy matching parameters from teacher to student before distillation.',
    )

    parser.add_argument('--distill_supervised_weight', type=float, default=1.0)
    parser.add_argument('--distill_point_weight', type=float, default=1.0)
    parser.add_argument('--distill_quantile_weight', type=float, default=0.5)

    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--micro_batch_size', type=int, default=64)
    parser.add_argument('--train_steps', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.95)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--lr_scheduler_type', choices=['constant', 'linear', 'cosine', 'constant_with_warmup'], default='cosine')
    parser.add_argument('--cosine_num_cycles', type=float, default=0.5)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--warmup_steps', type=int, default=0)

    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--ddp_timeout', type=int, default=7200)
    parser.add_argument(
        '--ddp_find_unused_parameters',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--evaluation_strategy', choices=['steps', 'epoch', 'no'], default='steps')
    parser.add_argument('--eval_steps', type=int, default=4000)
    parser.add_argument('--save_strategy', choices=['steps', 'epoch', 'no'], default='steps')
    parser.add_argument('--save_steps', type=int, default=4000)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--save_only_model', action='store_true')

    parser.add_argument('--validation_split_ratio', type=float, default=0.1)
    parser.add_argument(
        '--enable_validation_split',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--load_best_model_at_end',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss')
    parser.add_argument(
        '--greater_is_better',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    parser.add_argument('--dataloader_num_workers', type=int, default=1)

    parser.add_argument('--use_quantile_loss', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--quantile_loss_weight', type=float, default=1.0)
    parser.add_argument('--use_revin_norm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_revin_denorm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_gt', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        '--enable_overfit_fixed_window',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Recommended for stable distillation alignment.',
    )
    parser.add_argument('--overfit_hist_length', type=int, default=384)
    parser.add_argument('--overfit_gt_length', type=int, default=128)

    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.normalization_method == 'none':
        args.normalization_method = None

    rank_for_seed = int(os.getenv('RANK', '0'))
    setup_seed(int(args.seed) + rank_for_seed)

    num_devices = get_world_size()
    global_batch_size, micro_batch_size, gradient_accumulation_steps = _compute_batch_settings(
        args.global_batch_size,
        args.micro_batch_size,
        num_devices,
    )

    if args.train_steps is not None and args.train_steps > 0:
        max_steps = int(args.train_steps)
        num_train_epochs = -1
    else:
        max_steps = -1
        num_train_epochs = float(args.num_train_epochs)

    if args.precision == 'bf16':
        torch_dtype = torch.bfloat16
    elif args.precision == 'fp16':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32

    os.makedirs(args.output_path, exist_ok=True)

    log_in_local_rank_0(
        'Distillation env:',
        f'RANK={os.getenv("RANK")}',
        f'LOCAL_RANK={os.getenv("LOCAL_RANK")}',
        f'WORLD_SIZE={os.getenv("WORLD_SIZE")}',
        f'num_devices={num_devices}',
        f'global_batch_size={global_batch_size}',
        f'micro_batch_size={micro_batch_size}',
        f'grad_acc_steps={gradient_accumulation_steps}',
    )

    teacher_model = TimesFM2p5ForTraining.from_pretrained(
        model_path=args.teacher_model_path,
        torch_dtype=torch_dtype,
        use_quantile_loss=bool(args.use_quantile_loss),
        quantile_loss_weight=float(args.quantile_loss_weight),
        use_revin_norm=bool(args.use_revin_norm),
        use_gt=bool(args.use_gt),
        use_revin_denorm=bool(args.use_revin_denorm),
        enable_overfit_fixed_window=bool(args.enable_overfit_fixed_window),
        overfit_hist_length=int(args.overfit_hist_length),
        overfit_gt_length=int(args.overfit_gt_length),
    )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False
    log_in_local_rank_0(f'Loaded teacher from {args.teacher_model_path}')

    student_model = TimesFM2p5ForTraining(
        torch_dtype=torch_dtype,
        use_quantile_loss=bool(args.use_quantile_loss),
        quantile_loss_weight=float(args.quantile_loss_weight),
        use_revin_norm=bool(args.use_revin_norm),
        use_gt=bool(args.use_gt),
        use_revin_denorm=bool(args.use_revin_denorm),
        enable_overfit_fixed_window=bool(args.enable_overfit_fixed_window),
        overfit_hist_length=int(args.overfit_hist_length),
        overfit_gt_length=int(args.overfit_gt_length),
    )
    _truncate_student_layers(student_model, int(args.student_num_layers))
    student_model.config.student_num_layers = int(args.student_num_layers)
    log_in_local_rank_0(
        f'Built student with {args.student_num_layers} transformer layers '
        f'(full={len(teacher_model.backbone.stacked_xf)})'
    )

    if args.init_student_from_teacher:
        matched, missing, unexpected = _copy_matching_weights(
            teacher_model,
            student_model,
        )
        log_in_local_rank_0(
            f'Warm-start student from teacher: matched={matched}, '
            f'missing={len(missing)}, unexpected={len(unexpected)}'
        )

    num_total_params = 0
    for parameter in student_model.parameters():
        num_total_params += reduce(mul, parameter.shape)
    log_in_local_rank_0(f'Student parameters: {num_total_params}')

    runner = TimeMoeRunner(
        model_path=args.teacher_model_path,
        output_path=args.output_path,
        seed=args.seed,
        model_family='timesfm_2p5',
    )
    train_dataset, eval_dataset = runner.get_train_val_datasets(
        data_path=args.data_path,
        max_length=args.max_length,
        stride=args.stride,
        normalization_method=args.normalization_method,
        validation_split_ratio=float(args.validation_split_ratio),
        enable_validation_split=bool(args.enable_validation_split),
        max_train_sequences=args.max_train_sequences,
    )

    metric_for_best_model = args.metric_for_best_model
    greater_is_better = args.greater_is_better
    if greater_is_better is None and metric_for_best_model.endswith('loss'):
        greater_is_better = False

    eval_strategy_arg_name = 'evaluation_strategy'
    if 'eval_strategy' in TimeMoETrainingArguments.__dataclass_fields__:
        eval_strategy_arg_name = 'eval_strategy'

    eval_strategy = args.evaluation_strategy
    save_strategy = args.save_strategy
    eval_steps = _safe_float(args.eval_steps)
    save_steps = _safe_float(args.save_steps)
    load_best_model_at_end = bool(args.load_best_model_at_end)
    if eval_dataset is None:
        eval_strategy = 'no'
        load_best_model_at_end = False

    training_args_kwargs = dict(
        output_dir=args.output_path,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        logging_strategy='steps',
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        learning_rate=float(args.learning_rate),
        min_learning_rate=float(args.min_learning_rate),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_epsilon=float(args.adam_epsilon),
        lr_scheduler_type=args.lr_scheduler_type,
        cosine_num_cycles=float(args.cosine_num_cycles),
        warmup_ratio=float(args.warmup_ratio or 0.0),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        per_device_train_batch_size=int(micro_batch_size),
        per_device_eval_batch_size=int(micro_batch_size * 2),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        bf16=True if args.precision == 'bf16' else False,
        fp16=True if args.precision == 'fp16' else False,
        deepspeed=args.deepspeed,
        push_to_hub=False,
        logging_first_step=True,
        log_on_each_node=False,
        logging_steps=int(args.logging_steps),
        report_to=['tensorboard'],
        seed=int(args.seed) + rank_for_seed,
        data_seed=int(args.seed),
        max_grad_norm=float(args.max_grad_norm),
        optim='adamw_torch',
        torch_compile=False,
        dataloader_num_workers=int(args.dataloader_num_workers),
        ddp_timeout=int(args.ddp_timeout),
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
        logging_dir=os.path.join(args.output_path, 'tb_logs'),
        save_only_model=bool(args.save_only_model),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
    )
    training_args_kwargs[eval_strategy_arg_name] = eval_strategy

    training_args = TimeMoETrainingArguments(**training_args_kwargs)

    resume_from_checkpoint = args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == 'auto':
        from transformers.trainer_utils import get_last_checkpoint
        resume_from_checkpoint = get_last_checkpoint(args.output_path)
        if resume_from_checkpoint is None:
            log_in_local_rank_0('No checkpoint found for auto resume; start fresh.', type='warn')
        else:
            log_in_local_rank_0(f'Auto resume from checkpoint: {resume_from_checkpoint}')
    if isinstance(resume_from_checkpoint, str) and len(resume_from_checkpoint) > 0 and not os.path.exists(resume_from_checkpoint):
        log_in_local_rank_0(
            f'Resume checkpoint path does not exist: {resume_from_checkpoint}. Start without resume.',
            type='warn',
        )
        resume_from_checkpoint = None

    trainer = TimesFMDistillTrainer(
        model=student_model,
        teacher_model=teacher_model,
        supervised_weight=float(args.distill_supervised_weight),
        point_distill_weight=float(args.distill_point_weight),
        quantile_distill_weight=float(args.distill_quantile_weight),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_model_checkpoint = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    best_step = None
    if isinstance(best_model_checkpoint, str):
        matched = re.search(r'checkpoint-(\d+)$', best_model_checkpoint)
        if matched:
            best_step = int(matched.group(1))

    best_model_info = {
        'best_model_checkpoint': best_model_checkpoint,
        'best_metric': best_metric,
        'best_step': best_step,
        'metric_for_best_model': training_args.metric_for_best_model,
        'greater_is_better': training_args.greater_is_better,
        'student_num_layers': int(args.student_num_layers),
        'teacher_model_path': args.teacher_model_path,
        'distill_supervised_weight': float(args.distill_supervised_weight),
        'distill_point_weight': float(args.distill_point_weight),
        'distill_quantile_weight': float(args.distill_quantile_weight),
    }
    with open(os.path.join(args.output_path, 'best_model_info.json'), 'w', encoding='utf-8') as f:
        import json

        json.dump(best_model_info, f, ensure_ascii=False, indent=2)

    final_save_dir = os.path.join(args.output_path, 'student_final')
    trainer.save_model(final_save_dir)
    log_in_local_rank_0(f'Saved distilled student model to: {final_save_dir}')


if __name__ == '__main__':
    main()
