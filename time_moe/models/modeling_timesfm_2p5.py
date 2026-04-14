import os
import sys
import json
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from dataclasses import dataclass


@dataclass
class TimesFM2p5CausalLMOutput(CausalLMOutput):
    train_loss: Optional[torch.FloatTensor] = None
    quantile_loss_sum: Optional[torch.FloatTensor] = None
    quantile_logits: Optional[torch.FloatTensor] = None


class TimesFM2p5TrainingConfig(PretrainedConfig):
    model_type = 'timesfm_2p5'
    use_return_dict = True

    def __init__(
        self,
        patch_len: int = 32,
        output_patch_len: int = 128,
        decode_index: int = 5,
        num_quantiles: int = 10,
        use_quantile_loss: bool = True,
        quantile_loss_weight: float = 1.0,
        use_revin_norm: bool = True,
        use_gt: bool = True,
        use_revin_denorm: bool = True,
        fix_quantile_crossing: bool = True,
        infer_is_positive: bool = True,
        enable_overfit_fixed_window: bool = False,
        overfit_hist_length: int = 384,
        overfit_gt_length: int = 128,
        debug_input_dump_path: Optional[str] = None,
        debug_input_dump_every_n_steps: int = 0,
        debug_input_dump_max_steps: int = 20,
        debug_input_dump_include_values: bool = False,
        debug_input_dump_max_values: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.output_patch_len = output_patch_len
        self.decode_index = decode_index
        self.num_quantiles = num_quantiles
        self.use_quantile_loss = use_quantile_loss
        self.quantile_loss_weight = quantile_loss_weight
        self.use_revin_norm = use_revin_norm
        self.use_revin_denorm = use_revin_denorm
        self.use_gt = use_gt
        self.fix_quantile_crossing = bool(fix_quantile_crossing)
        self.infer_is_positive = bool(infer_is_positive)
        self.enable_overfit_fixed_window = bool(enable_overfit_fixed_window)
        self.overfit_hist_length = int(overfit_hist_length)
        self.overfit_gt_length = int(overfit_gt_length)
        self.debug_input_dump_path = debug_input_dump_path
        self.debug_input_dump_every_n_steps = int(debug_input_dump_every_n_steps)
        self.debug_input_dump_max_steps = int(debug_input_dump_max_steps)
        self.debug_input_dump_include_values = bool(debug_input_dump_include_values)
        self.debug_input_dump_max_values = int(debug_input_dump_max_values)


class TimesFM2p5ForTraining(nn.Module):
    def __init__(
        self,
        torch_dtype: Optional[torch.dtype] = None,
        use_quantile_loss: bool = True,
        quantile_loss_weight: float = 1.0,
        use_revin_norm: bool = True,
        use_gt: bool = True,
        use_revin_denorm: bool = True,
        fix_quantile_crossing: bool = True,
        infer_is_positive: bool = True,
        enable_overfit_fixed_window: bool = False,
        overfit_hist_length: int = 384,
        overfit_gt_length: int = 128,
        debug_input_dump_path: Optional[str] = None,
        debug_input_dump_every_n_steps: int = 0,
        debug_input_dump_max_steps: int = 20,
        debug_input_dump_include_values: bool = False,
        debug_input_dump_max_values: int = 32,
    ):
        super().__init__()
        self._ensure_timesfm_importable()

        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        from timesfm.torch import util as timesfm_util

        self.timesfm_util = timesfm_util
        self.backbone = TimesFM_2p5_200M_torch_module(
        )

        self.use_revin_norm = bool(use_revin_norm)
        self.use_revin_denorm = bool(use_revin_denorm)
        self.use_gt = bool(use_gt)
        self.fix_quantile_crossing = bool(fix_quantile_crossing)
        self.infer_is_positive = bool(infer_is_positive)
        self.enable_overfit_fixed_window = bool(enable_overfit_fixed_window)
        self.overfit_hist_length = int(overfit_hist_length)
        self.overfit_gt_length = int(overfit_gt_length)
        self.debug_input_dump_path = debug_input_dump_path
        self.debug_input_dump_every_n_steps = int(debug_input_dump_every_n_steps)
        self.debug_input_dump_max_steps = int(debug_input_dump_max_steps)
        self.debug_input_dump_include_values = bool(debug_input_dump_include_values)
        self.debug_input_dump_max_values = int(debug_input_dump_max_values)
        self._debug_input_forward_step = 0
        self._debug_input_dumped_steps = 0

        self.patch_len = self.backbone.p
        self.output_patch_len = self.backbone.o
        self.decode_index = self.backbone.aridx
        self.num_quantiles = self.backbone.q
        self.quantiles = list(self.backbone.config.quantiles)
        self.use_quantile_loss = use_quantile_loss
        self.quantile_loss_weight = float(quantile_loss_weight)
        self.config = TimesFM2p5TrainingConfig(
            patch_len=self.patch_len,
            output_patch_len=self.output_patch_len,
            decode_index=self.decode_index,
            num_quantiles=self.num_quantiles,
            use_quantile_loss=self.use_quantile_loss,
            quantile_loss_weight=self.quantile_loss_weight,
            use_revin_norm=self.use_revin_norm,
            use_revin_denorm=self.use_revin_denorm,
            use_gt=self.use_gt,
            fix_quantile_crossing=self.fix_quantile_crossing,
            infer_is_positive=self.infer_is_positive,
            enable_overfit_fixed_window=self.enable_overfit_fixed_window,
            overfit_hist_length=self.overfit_hist_length,
            overfit_gt_length=self.overfit_gt_length,
            debug_input_dump_path=self.debug_input_dump_path,
            debug_input_dump_every_n_steps=self.debug_input_dump_every_n_steps,
            debug_input_dump_max_steps=self.debug_input_dump_max_steps,
            debug_input_dump_include_values=self.debug_input_dump_include_values,
            debug_input_dump_max_values=self.debug_input_dump_max_values,
        )

        if torch_dtype is not None:
            self.backbone = self.backbone.to(dtype=torch_dtype)

    @staticmethod
    def _ensure_timesfm_importable():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        timesfm_src = os.path.join(repo_root, 'timesfm', 'src')
        if timesfm_src not in sys.path:
            sys.path.insert(0, timesfm_src)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        use_quantile_loss: bool = True,
        quantile_loss_weight: float = 1.0,
        use_revin_norm: bool = True,
        use_revin_denorm: bool = True,
        use_gt: bool = True,
        fix_quantile_crossing: bool = True,
        infer_is_positive: bool = True,
        enable_overfit_fixed_window: bool = False,
        overfit_hist_length: int = 384,
        overfit_gt_length: int = 128,
        debug_input_dump_path: Optional[str] = None,
        debug_input_dump_every_n_steps: int = 0,
        debug_input_dump_max_steps: int = 20,
        debug_input_dump_include_values: bool = False,
        debug_input_dump_max_values: int = 32,
    ):
        model = cls(
            torch_dtype=torch_dtype,
            use_quantile_loss=use_quantile_loss,
            quantile_loss_weight=quantile_loss_weight,
            use_revin_norm=use_revin_norm,
            use_revin_denorm=use_revin_denorm,
            use_gt=use_gt,
            fix_quantile_crossing=fix_quantile_crossing,
            infer_is_positive=infer_is_positive,
            enable_overfit_fixed_window=enable_overfit_fixed_window,
            overfit_hist_length=overfit_hist_length,
            overfit_gt_length=overfit_gt_length,
            debug_input_dump_path=debug_input_dump_path,
            debug_input_dump_every_n_steps=debug_input_dump_every_n_steps,
            debug_input_dump_max_steps=debug_input_dump_max_steps,
            debug_input_dump_include_values=debug_input_dump_include_values,
            debug_input_dump_max_values=debug_input_dump_max_values,
        )
        weight_path = model._resolve_weight_path(model_path)

        from safetensors.torch import load_file

        state_dict = load_file(weight_path)
        model.backbone.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def _resolve_weight_path(model_path: str) -> str:
        if os.path.isdir(model_path):
            weight_path = os.path.join(model_path, 'model.safetensors')
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f'Cannot find model.safetensors in {model_path}')
            return weight_path

        if os.path.isfile(model_path):
            return model_path

        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=model_path, filename='model.safetensors')

    def _get_patch_stats(self, patched_inputs: torch.Tensor, patched_masks: torch.Tensor):
        batch_size, num_patches, _ = patched_inputs.shape
        n = torch.zeros(batch_size, device=patched_inputs.device)
        mu = torch.zeros(batch_size, device=patched_inputs.device)
        sigma = torch.zeros(batch_size, device=patched_inputs.device)

        patch_mu = []
        patch_sigma = []

        for patch_idx in range(num_patches):
            (n, mu, sigma), _ = self.timesfm_util.update_running_stats(
                n,
                mu,
                sigma,
                patched_inputs[:, patch_idx],
                patched_masks[:, patch_idx],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        return torch.stack(patch_mu, dim=1), torch.stack(patch_sigma, dim=1)

    @staticmethod
    def _quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
        dev = target - pred
        loss_first = dev * quantile
        loss_second = -dev * (1.0 - quantile)
        return 2.0 * torch.where(loss_first >= 0, loss_first, loss_second)

    @staticmethod
    def _flip_quantile_dim(x: torch.Tensor) -> torch.Tensor:
        # Keep index 0 (mean) unchanged and reverse quantile dimensions.
        return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)

    @staticmethod
    def _fix_quantile_crossing(pred_quantile_patches: torch.Tensor) -> torch.Tensor:
        for quantile_index in [4, 3, 2, 1]:
            pred_quantile_patches[..., quantile_index] = torch.where(
                pred_quantile_patches[..., quantile_index]
                < pred_quantile_patches[..., quantile_index + 1],
                pred_quantile_patches[..., quantile_index],
                pred_quantile_patches[..., quantile_index + 1],
            )
        for quantile_index in [6, 7, 8, 9]:
            pred_quantile_patches[..., quantile_index] = torch.where(
                pred_quantile_patches[..., quantile_index]
                > pred_quantile_patches[..., quantile_index - 1],
                pred_quantile_patches[..., quantile_index],
                pred_quantile_patches[..., quantile_index - 1],
            )
        return pred_quantile_patches

    @staticmethod
    def _infer_is_positive(full_series: torch.Tensor, full_valid: torch.Tensor) -> torch.Tensor:
        inf_tensor = torch.full_like(full_series, float('inf'))
        valid_values = torch.where(full_valid > 0, full_series, inf_tensor)
        return valid_values.min(dim=1).values >= 0

    def _debug_summarize_tensor(self, value: torch.Tensor):
        detached = value.detach()
        summary = {
            'shape': list(detached.shape),
            'dtype': str(detached.dtype),
            'device': str(detached.device),
        }

        if detached.numel() > 0:
            if detached.dtype == torch.bool:
                summary['true_ratio'] = float(detached.to(torch.float32).mean().item())
            else:
                stats_tensor = detached.to(torch.float32)
                summary['min'] = float(stats_tensor.min().item())
                summary['max'] = float(stats_tensor.max().item())
                summary['mean'] = float(stats_tensor.mean().item())
                summary['std'] = float(stats_tensor.std(unbiased=False).item())

            if self.debug_input_dump_include_values:
                max_values = max(1, self.debug_input_dump_max_values)
                sample = detached.reshape(-1)[:max_values].to('cpu')
                summary['sample_values'] = sample.tolist()

        return summary

    def _debug_serialize_value(self, value):
        if isinstance(value, torch.Tensor):
            return self._debug_summarize_tensor(value)
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._debug_serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self._debug_serialize_value(v) for k, v in value.items()}
        return str(value)

    def _maybe_dump_forward_inputs(self, branch: str, payload: dict):
        self._debug_input_forward_step += 1
        if not self.debug_input_dump_path:
            return
        if self.debug_input_dump_every_n_steps <= 0:
            return
        if self._debug_input_forward_step % self.debug_input_dump_every_n_steps != 0:
            return
        if self.debug_input_dump_max_steps > 0 and self._debug_input_dumped_steps >= self.debug_input_dump_max_steps:
            return

        record = {
            'forward_step': int(self._debug_input_forward_step),
            'branch': branch,
            'rank': os.getenv('RANK', '0'),
            'local_rank': os.getenv('LOCAL_RANK', '0'),
            'pid': os.getpid(),
            'payload': self._debug_serialize_value(payload),
        }

        try:
            os.makedirs(os.path.dirname(self.debug_input_dump_path) or '.', exist_ok=True)
            with open(self.debug_input_dump_path, 'a', encoding='utf-8') as fw:
                fw.write(json.dumps(record, ensure_ascii=False) + '\n')
            self._debug_input_dumped_steps += 1
        except Exception as exc:
            print(f'[TimesFM2p5ForTraining] failed to dump debug input json: {exc}')

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        labels: Optional[torch.FloatTensor] = None,
        loss_masks: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if self.use_gt :    
            if input_ids is None:
                raise ValueError('input_ids cannot be None')

            if return_dict is None:
                return_dict = self.config.use_return_dict

            device = input_ids.device
            input_ids = input_ids.to(dtype=torch.float32)
            # 1. 准备完整的序列数据 (Full Series)
            full_series = input_ids
            
            if loss_masks is None:
                full_valid = torch.ones_like(full_series, dtype=torch.float32, device=device)
            else:
                full_valid = loss_masks.to(device=device, dtype=torch.float32)
            is_positive = self._infer_is_positive(full_series, full_valid) if self.infer_is_positive else None

            batch_size, seq_len = full_series.shape
            
            # 2. 计算可以切分多少个 Input Patch
            num_patches = seq_len // self.patch_len
            
            # 截断多余的部分，保证是 patch_len 的倍数
            valid_len = num_patches * self.patch_len
            input_series = full_series[:, :valid_len]

            input_mask_series = full_valid[:, :valid_len].clone()
            if num_patches < 2:
                raise ValueError(f"Sequence too short: need at least 2 patches, got {num_patches}")

            if self.enable_overfit_fixed_window:
                if self.overfit_hist_length <= 0 or self.overfit_gt_length <= 0:
                    raise ValueError(
                        'overfit_hist_length and overfit_gt_length must be positive when enable_overfit_fixed_window=True'
                    )
                if self.overfit_hist_length % self.patch_len != 0:
                    raise ValueError(
                        f'overfit_hist_length must be multiple of patch_len={self.patch_len}, got {self.overfit_hist_length}'
                    )
                fixed_total_len = self.overfit_hist_length + self.overfit_gt_length
                if fixed_total_len > valid_len:
                    raise ValueError(
                        f'Sequence too short for overfit fixed window: valid_len={valid_len}, required={fixed_total_len}'
                    )
                hist_len = self.overfit_hist_length
                hist_last_patch_idx = (hist_len // self.patch_len) - 1
            else:
                max_hist_last_patch_idx = num_patches - 4
                if max_hist_last_patch_idx <= 0:
                    raise ValueError(
                        f"Sequence too short for random hist/gt split: num_patches={num_patches}, require >=5"
                    )
                hist_last_patch_idx = torch.randint(
                    low=3,
                    high=max_hist_last_patch_idx,
                    size=(1,),
                    device=device,
                ).item()
                hist_len = (hist_last_patch_idx + 1) * self.patch_len

            # 使用整条序列统计量做统一归一化（基于有效 mask）
            full_valid_count = torch.clamp(input_mask_series.sum(dim=1, keepdim=True), min=1.0)
            full_mu = (input_series * input_mask_series).sum(dim=1, keepdim=True) / full_valid_count
            full_var = ((input_series - full_mu) ** 2 * input_mask_series).sum(dim=1, keepdim=True) / full_valid_count
            full_sigma = torch.sqrt(torch.clamp(full_var, min=1e-8))

            input_series = (input_series - full_mu) / full_sigma

            # 3. 构建 Input Patches [B, N, 32]
            patched_inputs = input_series.reshape(batch_size, num_patches, self.patch_len)
            patched_masks = torch.logical_not(
                input_mask_series.reshape(batch_size, num_patches, self.patch_len).bool()
            )

            # 4. 正向分支：统计量与归一化
            context_mu, context_sigma = self._get_patch_stats(patched_inputs, patched_masks)
            if self.use_revin_norm:
                normed_inputs = self.timesfm_util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
            else:
                normed_inputs = patched_inputs
            normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
            backbone_first_param = next(self.backbone.parameters(), None)
            model_dtype = next(self.backbone.parameters()).dtype
            normed_inputs = normed_inputs.to(dtype=model_dtype)

            # 4.1 负向分支：独立统计量与归一化（对 -x 单独计算）
            neg_patched_inputs = -patched_inputs
            neg_context_mu, neg_context_sigma = self._get_patch_stats(neg_patched_inputs, patched_masks)
            if self.use_revin_norm:
                neg_normed_inputs = self.timesfm_util.revin(
                    neg_patched_inputs,
                    neg_context_mu,
                    neg_context_sigma,
                    reverse=False,
                )
            else:
                neg_normed_inputs = neg_patched_inputs
            neg_normed_inputs = torch.where(patched_masks, 0.0, neg_normed_inputs)
            neg_normed_inputs = neg_normed_inputs.to(dtype=model_dtype)

            # 5. 模型前向传播
            (_, _, output_ts, output_quantile_spread), _ = self.backbone(normed_inputs, patched_masks)
            (_, _, flipped_output_ts, flipped_output_quantile_spread), _ = self.backbone(
                neg_normed_inputs, patched_masks
            )

            # 5.1 在 reshape 前先反归一化（与 debug 流程一致）
            if self.use_revin_denorm:
                output_ts = self.timesfm_util.revin(output_ts, context_mu, context_sigma, reverse=True)
                output_quantile_spread = self.timesfm_util.revin(
                    output_quantile_spread,
                    context_mu,
                    context_sigma,
                    reverse=True,
                )
                flipped_output_ts = self.timesfm_util.revin(
                    flipped_output_ts,
                    neg_context_mu,
                    neg_context_sigma,
                    reverse=True,
                )
                flipped_output_quantile_spread = self.timesfm_util.revin(
                    flipped_output_quantile_spread,
                    neg_context_mu,
                    neg_context_sigma,
                    reverse=True,
                )

            # Reshape 输出为 [B, N, 128, num_quantiles]
            output_ts = output_ts.reshape(
                batch_size,
                num_patches,
                self.output_patch_len,
                self.num_quantiles,
            )
            flipped_output_ts = flipped_output_ts.reshape(
                batch_size,
                num_patches,
                self.output_patch_len,
                self.num_quantiles,
            )

            # Apply force-flip-invariance in normalized space:
            # f(x) <- 0.5 * (f(x) - flip_quantile(f(-x)))
            output_quantile_spread = output_quantile_spread.reshape(
                batch_size,
                num_patches,
                -1,
                self.num_quantiles,
            )
            flipped_output_quantile_spread = flipped_output_quantile_spread.reshape(
                batch_size,
                num_patches,
                -1,
                self.num_quantiles,
            )

            # 6. 翻转分位数后做相减取平均
            flipped_output_ts = self._flip_quantile_dim(flipped_output_ts)
            output_ts = 0.5 * (output_ts - flipped_output_ts)
            
            flipped_output_quantile_spread = self._flip_quantile_dim(flipped_output_quantile_spread)
            output_quantile_spread = 0.5 * (
                output_quantile_spread - flipped_output_quantile_spread
            )
            # Keep downstream logic unchanged (expects [B, N, 1024 * num_quantiles]).
            output_quantile_spread = output_quantile_spread.reshape(
                batch_size,
                num_patches,
                -1,
            )
            
            full_quantile_spread = output_quantile_spread[:, hist_last_patch_idx, :]
            total_horizon = full_quantile_spread.shape[-1]
            num_future_patches = total_horizon//self.num_quantiles // self.output_patch_len
            quantile_spread_flat = full_quantile_spread.reshape(
                batch_size,
                1024,
                self.num_quantiles,
            )  # [B, 1024, 10]

            quantile_spread_unfolded = quantile_spread_flat.unfold(1, self.output_patch_len, self.patch_len)

            quantile_spread_patches = quantile_spread_unfolded.permute(0, 1, 3, 2).contiguous()
            
            # 提取点预测 (Point Forecast) -> [B, N, 128]
            target_start = hist_len
            pred_start_idx = hist_last_patch_idx
            if self.enable_overfit_fixed_window:
                target_end = min(valid_len, hist_len + self.overfit_gt_length)
            else:
                target_end = valid_len
            source_for_targets = input_series[:, target_start:target_end]
            mask_for_targets = full_valid[:, target_start:target_end]
            source_for_MSE = input_series[:, self.patch_len:]
            mask_for_MSE = full_valid[:, self.patch_len:]
            if source_for_targets.shape[1] < self.output_patch_len:
                raise ValueError(f"Sequence too short for output length {self.output_patch_len}")
            targets_unfolded = source_for_targets.unfold(1, self.output_patch_len, self.patch_len)
            masks_unfolded = mask_for_targets.unfold(1, self.output_patch_len, self.patch_len)
            mse_source_unfolded = source_for_MSE.unfold(1, self.output_patch_len, self.patch_len)
            mse_masks_unfolded = mask_for_MSE.unfold(1, self.output_patch_len, self.patch_len)
            pred_available = output_ts.shape[1] - pred_start_idx
            target_available = targets_unfolded.shape[1]
            min_patches = min(pred_available, target_available)
            min_patch_mse = min(output_ts.shape[1], mse_source_unfolded.shape[1])
            pred_aligned = output_ts[:, pred_start_idx:pred_start_idx + min_patches, :, :].clone()
            continuous_quantile_patches = min(min_patches, quantile_spread_patches.shape[1])

            # for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
            #     pred_aligned[:, :continuous_quantile_patches, :, quantile_index] = (
            #         quantile_spread_patches[:, :continuous_quantile_patches, :self.output_patch_len, quantile_index]
            #         - quantile_spread_patches[:, :continuous_quantile_patches, :self.output_patch_len, self.decode_index]
            #         + pred_aligned[:, :continuous_quantile_patches, :self.output_patch_len, self.decode_index]
            #     )
            targets_aligned = targets_unfolded[:, :min_patches, :]
            masks_aligned = masks_unfolded[:, :min_patches, :]
            targets_aligned_mse = mse_source_unfolded[:, :min_patch_mse, :]
            masks_aligned_mse = mse_masks_unfolded[:, :min_patch_mse, :]
            pred = output_ts[:, :min_patch_mse , :, self.decode_index].clone()
            target = targets_unfolded[:, :continuous_quantile_patches, :]
            mask = masks_unfolded[:, :continuous_quantile_patches, :]

            pred_quantile_patches = output_ts[:, :min_patch_mse, :, :].clone()
            if self.fix_quantile_crossing:
                pred_quantile_patches = self._fix_quantile_crossing(pred_quantile_patches)

            if is_positive is not None:
                zero_in_norm_3d = (-full_mu / full_sigma.clamp(min=1e-5)).unsqueeze(-1)
                zero_in_norm_4d = zero_in_norm_3d.unsqueeze(-1)
                pred = torch.where(
                    is_positive[:, None, None],
                    torch.maximum(pred, zero_in_norm_3d),
                    pred,
                )
                pred_quantile_patches = torch.where(
                    is_positive[:, None, None, None],
                    torch.maximum(pred_quantile_patches, zero_in_norm_4d),
                    pred_quantile_patches,
                )
            self._maybe_dump_forward_inputs(
                branch='use_gt',
                payload={
                    'input_ids': input_ids,
                    'labels': labels,
                    'loss_masks': loss_masks,
                    'attention_mask': attention_mask,
                    'full_valid': full_valid,
                    'input_series': input_series,
                    'input_mask_series': input_mask_series,
                    'patched_inputs': patched_inputs,
                    'patched_masks': patched_masks,
                    'targets_unfolded': targets_unfolded,
                    'masks_unfolded': masks_unfolded,
                    'mse_source_unfolded': mse_source_unfolded,
                    'mse_masks_unfolded': mse_masks_unfolded,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'num_patches': num_patches,
                    'valid_len': valid_len,
                    'hist_len': hist_len,
                    'hist_last_patch_idx': hist_last_patch_idx,
                    'target_start': target_start,
                    'target_end': target_end,
                    'pred_start_idx': pred_start_idx,
                    'min_patches': min_patches,
                    'min_patch_mse': min_patch_mse,
                    'continuous_quantile_patches': continuous_quantile_patches,
                    'num_future_patches': num_future_patches,
                },
            )


            
            # 7. 计算 Loss
            point_loss = (pred - targets_aligned_mse) ** 2
            
            # 应用 Mask (包括原始数据的 mask 和 边界不足导致的截断)
            weighted_loss = point_loss * masks_aligned_mse
            valid_count = torch.clamp(masks_aligned_mse.sum(), min=1.0)
            train_loss = weighted_loss.sum() / valid_count
            quantile_loss_sum = train_loss.new_zeros(())
            loss = train_loss

            # 8. 分位数 Loss (Optional)
            if self.use_quantile_loss:
                quantile_preds = pred_quantile_patches[:, :, :, 1:] # [B, N, 128, Q]
                quantile_targets = targets_aligned_mse[:, :, :].unsqueeze(-1)   # [B, N, 128, 1]
                quantile_masks = masks_aligned_mse[:, :, :].unsqueeze(-1)       # [B, N, 128, 1]
                quantile_valid_count = torch.clamp(quantile_masks.sum(), min=1.0)

                for q_idx, q_val in enumerate(self.quantiles):
                    q_pred = quantile_preds[:, :, :, q_idx]
                    q_loss = self._quantile_loss(q_pred, quantile_targets[..., 0], q_val)
                    quantile_loss_sum = quantile_loss_sum + (q_loss * quantile_masks[..., 0]).sum() / quantile_valid_count

                loss = train_loss + self.quantile_loss_weight * quantile_loss_sum
            if not return_dict:
                return (loss, pred, train_loss, quantile_loss_sum)

            return TimesFM2p5CausalLMOutput(
                loss=loss,
                logits=pred, # 返回与 train_loss 对齐后的预测值
                train_loss=train_loss,
                quantile_loss_sum=quantile_loss_sum,
                quantile_logits=pred_quantile_patches,
            )
        
        else :
            if input_ids is None:
                raise ValueError('input_ids cannot be None')

            if return_dict is None:
                return_dict = self.config.use_return_dict

            device = input_ids.device
            input_ids = input_ids.to(dtype=torch.float32)
            
            # 1. 准备完整的序列数据 (Full Series)
            full_series = input_ids
            
            if loss_masks is None:
                full_valid = torch.ones_like(full_series, dtype=torch.float32, device=device)
            else:
                full_valid = loss_masks.to(device=device, dtype=torch.float32)
            is_positive = self._infer_is_positive(full_series, full_valid) if self.infer_is_positive else None

            batch_size, seq_len = full_series.shape
            
            # 2. 计算可以切分多少个 Input Patch
            num_patches = seq_len // self.patch_len
            
            # 截断多余的部分，保证是 patch_len 的倍数
            valid_len = num_patches * self.patch_len
            input_series = full_series[:, :valid_len]
            input_mask_series = full_valid[:, :valid_len].clone()

            if not self.enable_overfit_fixed_window:
                first_patch_mask_end = torch.randint(
                    low=0,
                    high=self.patch_len,
                    size=(batch_size,),
                    device=device,
                )
                first_patch_positions = torch.arange(self.patch_len, device=device).unsqueeze(0)
                first_patch_prefix_mask = first_patch_positions <= first_patch_mask_end.unsqueeze(1)
                input_mask_series[:, :self.patch_len] = input_mask_series[:, :self.patch_len] * torch.logical_not(
                    first_patch_prefix_mask
                ).to(dtype=input_mask_series.dtype)
            
            # 3. 构建 Input Patches [B, N, 32]
            patched_inputs = input_series.reshape(batch_size, num_patches, self.patch_len)
            patched_masks = torch.logical_not(
                input_mask_series.reshape(batch_size, num_patches, self.patch_len).bool()
            )
            # 4. 获取统计量并归一化输入
            context_mu, context_sigma = self._get_patch_stats(patched_inputs, patched_masks)
            if self.use_revin_norm:
                normed_inputs = self.timesfm_util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
            else:
                normed_inputs = patched_inputs
            normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
            backbone_first_param = next(self.backbone.parameters(), None)
            model_dtype = backbone_first_param.dtype if backbone_first_param is not None else normed_inputs.dtype
            normed_inputs = normed_inputs.to(dtype=model_dtype)

            # 5. 模型前向传播
            (_, _, output_ts, output_quantile_spread), _ = self.backbone(normed_inputs, patched_masks)

            # Reshape 输出为 [B, N, 128, num_quantiles]
            output_ts = output_ts.reshape(
                batch_size,
                num_patches,
                self.output_patch_len,
                self.num_quantiles,
            )

            output_quantile_spread = output_quantile_spread.reshape(
                batch_size,
                num_patches,
                1024,
                self.num_quantiles,
            )

            # 反归一化输出
            if self.use_revin_denorm:
                output_ts = self.timesfm_util.revin(output_ts, context_mu, context_sigma, reverse=True)
                output_quantile_spread = self.timesfm_util.revin(output_quantile_spread, context_mu, context_sigma, reverse=True)
            
            # 提取点预测 (Point Forecast) -> [B, N, 128]
            # 震荡的原因？
            pred_full_horizon = output_ts[:, :, :, self.decode_index]
            target_start_idx = self.patch_len
            source_for_targets = full_series[:, target_start_idx:]
            mask_for_targets = full_valid[:, target_start_idx:]
            
            if source_for_targets.shape[1] < self.output_patch_len:
                 raise ValueError(f"Sequence too short for output length {self.output_patch_len}")
            targets_unfolded = source_for_targets.unfold(1, self.output_patch_len, self.patch_len)
            masks_unfolded = mask_for_targets.unfold(1, self.output_patch_len, self.patch_len)
            
            min_patches = min(pred_full_horizon.shape[1], targets_unfolded.shape[1])
            
            pred_aligned = pred_full_horizon[:, :min_patches, :]
            targets_aligned = targets_unfolded[:, :min_patches, :]
            masks_aligned = masks_unfolded[:, :min_patches, :]

            self._maybe_dump_forward_inputs(
                branch='no_gt',
                payload={
                    'input_ids': input_ids,
                    'labels': labels,
                    'loss_masks': loss_masks,
                    'attention_mask': attention_mask,
                    'full_valid': full_valid,
                    'input_series': input_series,
                    'input_mask_series': input_mask_series,
                    'patched_inputs': patched_inputs,
                    'patched_masks': patched_masks,
                    'source_for_targets': source_for_targets,
                    'mask_for_targets': mask_for_targets,
                    'targets_unfolded': targets_unfolded,
                    'masks_unfolded': masks_unfolded,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'num_patches': num_patches,
                    'valid_len': valid_len,
                    'target_start_idx': target_start_idx,
                    'min_patches': min_patches,
                },
            )

            # 7. 计算 Loss
            point_loss = (pred_aligned - targets_aligned) ** 2
            
            # 应用 Mask (包括原始数据的 mask 和 边界不足导致的截断)
            weighted_loss = point_loss * masks_aligned
            valid_count = torch.clamp(masks_aligned.sum(), min=1.0)
            train_loss = weighted_loss.sum() / valid_count
            quantile_loss_sum = train_loss.new_zeros(())
            loss = train_loss

            quantile_preds_full = output_ts[:, :min_patches, :, :].clone()  # [B, N, 128, Q]
            for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
                quantile_preds_full[:, :, :, quantile_index] = (
                        output_quantile_spread[:, :min_patches, :self.output_patch_len, quantile_index]
                        - output_quantile_spread[:, :min_patches, :self.output_patch_len, self.decode_index]
                        + quantile_preds_full[:, :, :self.output_patch_len, self.decode_index]
                    )

            if self.fix_quantile_crossing:
                quantile_preds_full = self._fix_quantile_crossing(quantile_preds_full)

            if is_positive is not None:
                zero_in_norm_3d = torch.zeros(1, device=device)
                zero_in_norm_4d = torch.zeros(1, device=device)
                pred_aligned = torch.where(
                    is_positive[:, None, None],
                    torch.maximum(pred_aligned, zero_in_norm_3d),
                    pred_aligned,
                )
                quantile_preds_full = torch.where(
                    is_positive[:, None, None, None],
                    torch.maximum(quantile_preds_full, zero_in_norm_4d),
                    quantile_preds_full,
                )

            # 8. 分位数 Loss (Optional)
            if self.use_quantile_loss:
                quantile_preds = quantile_preds_full[:, :, :, 1:] # [B, N, 128, Q]
                quantile_targets = targets_aligned.unsqueeze(-1)   # [B, N, 128, 1]
                quantile_masks = masks_aligned.unsqueeze(-1)       # [B, N, 128, 1]

                for q_idx, q_val in enumerate(self.quantiles):
                    q_pred = quantile_preds[:, :, :, q_idx]
                    q_loss = self._quantile_loss(q_pred, quantile_targets[..., 0], q_val)
                    quantile_loss_sum = quantile_loss_sum + (q_loss * quantile_masks[..., 0]).sum() / valid_count

                loss = train_loss + self.quantile_loss_weight * quantile_loss_sum
            if not return_dict:
                return (loss, pred_aligned, train_loss, quantile_loss_sum)

            return TimesFM2p5CausalLMOutput(
                loss=loss,
                logits=pred_aligned, # 返回对齐后的预测值
                train_loss=train_loss,
                quantile_loss_sum=quantile_loss_sum,
                quantile_logits=quantile_preds_full,
            )


