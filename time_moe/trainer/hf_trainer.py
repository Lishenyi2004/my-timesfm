#!/usr/bin/env python
# -*- coding:utf-8 _*-
import math
from dataclasses import field, dataclass
from functools import partial

import inspect

import transformers
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler


class TimeMoeTrainer(transformers.Trainer):
    epsilon = 1e-8

    def __init__(self, label_column: str = 'labels', loss_mask_column: str = 'loss_mask', *positional_args, **kwargs):
        super().__init__(*positional_args, **kwargs)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.label_column = label_column
        self.loss_mask_column = loss_mask_column
        self._extra_loss_sums = {}
        self._extra_loss_counts = {}

    @staticmethod
    def _extract_metric_value(outputs, key: str):
        value = None
        if isinstance(outputs, dict):
            value = outputs.get(key)
        else:
            value = getattr(outputs, key, None)

        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.detach().float().mean().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _accumulate_extra_metric(self, name: str, value):
        if value is None:
            return
        self._extra_loss_sums[name] = self._extra_loss_sums.get(name, 0.0) + float(value)
        self._extra_loss_counts[name] = self._extra_loss_counts.get(name, 0) + 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is not None:
            outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
        else:
            outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        if model.training:
            self._accumulate_extra_metric('train_loss', self._extract_metric_value(outputs, 'train_loss'))
            self._accumulate_extra_metric('quantile_loss_sum', self._extract_metric_value(outputs, 'quantile_loss_sum'))

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, start_time=None):
        if self._extra_loss_sums:
            logs = dict(logs)
            for metric_name, metric_sum in self._extra_loss_sums.items():
                metric_count = self._extra_loss_counts.get(metric_name, 0)
                if metric_count > 0:
                    logs[metric_name] = metric_sum / metric_count
            self._extra_loss_sums.clear()
            self._extra_loss_counts.clear()

        if start_time is None:
            return super().log(logs)
        return super().log(logs, start_time=start_time)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        optimizer = self.optimizer if optimizer is None else optimizer
        min_lr_ratio = self.args.min_learning_rate / self.args.learning_rate
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == 'cosine':
                self.lr_scheduler = get_cosine_schedule_with_warmup_min_lr(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    num_cycles=self.args.cosine_num_cycles,
                    min_lr_ratio=min_lr_ratio,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            params = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns = list(set(
                params + self.label_names + [
                    "label",
                    "label_ids",
                    self.label_column,
                    self.loss_mask_column
                ]
            ))


@dataclass
class TimeMoETrainingArguments(transformers.TrainingArguments):
    min_learning_rate: float = field(
        default=0, metadata={"help": "Minimum learning rate for cosine_schedule"}
    )
    cosine_num_cycles: float = field(
        default=0.5, metadata={"help": "Number of cosine cycles when using cosine scheduler"}
    )


def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_ratio)


def get_cosine_schedule_with_warmup_min_lr(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0,
        last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_and_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)