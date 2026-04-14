import torch
import torch.nn.functional as F

from time_moe.trainer.hf_trainer import TimeMoeTrainer


class TimesFMDistillTrainer(TimeMoeTrainer):
    def __init__(
        self,
        teacher_model,
        supervised_weight: float = 1.0,
        point_distill_weight: float = 1.0,
        quantile_distill_weight: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.supervised_weight = float(supervised_weight)
        self.point_distill_weight = float(point_distill_weight)
        self.quantile_distill_weight = float(quantile_distill_weight)
        self._teacher_device_initialized = False

    def _ensure_teacher_device(self, model):
        if self._teacher_device_initialized:
            return
        student_device = next(model.parameters()).device
        self.teacher_model.to(student_device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False
        self._teacher_device_initialized = True

    @staticmethod
    def _restore_rng_state(cpu_state, cuda_states):
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._ensure_teacher_device(model)

        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        if num_items_in_batch is not None:
            student_outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
        else:
            student_outputs = model(**inputs)

        with torch.no_grad():
            self._restore_rng_state(cpu_rng_state, cuda_rng_states)
            teacher_outputs = self.teacher_model(**inputs)

        supervised_loss = student_outputs.loss
        point_distill_loss = F.mse_loss(
            student_outputs.logits.float(),
            teacher_outputs.logits.float(),
        )

        quantile_distill_loss = supervised_loss.new_zeros(())
        student_q = getattr(student_outputs, 'quantile_logits', None)
        teacher_q = getattr(teacher_outputs, 'quantile_logits', None)
        if student_q is not None and teacher_q is not None:
            quantile_distill_loss = F.mse_loss(student_q.float(), teacher_q.float())

        total_loss = (
            self.supervised_weight * supervised_loss
            + self.point_distill_weight * point_distill_loss
            + self.quantile_distill_weight * quantile_distill_loss
        )

        if model.training:
            self._accumulate_extra_metric('train_loss', self._extract_metric_value(student_outputs, 'train_loss'))
            self._accumulate_extra_metric('quantile_loss_sum', self._extract_metric_value(student_outputs, 'quantile_loss_sum'))
            self._accumulate_extra_metric('distill_point_loss', point_distill_loss.detach().float().mean().item())
            self._accumulate_extra_metric('distill_quantile_loss', quantile_distill_loss.detach().float().mean().item())
            self._accumulate_extra_metric('distill_total_loss', total_loss.detach().float().mean().item())

        if return_outputs:
            if isinstance(student_outputs, dict):
                student_outputs['loss'] = total_loss
            else:
                student_outputs.loss = total_loss
            return total_loss, student_outputs

        return total_loss
