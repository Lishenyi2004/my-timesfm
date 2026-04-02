import os
import sys
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
        timesfm_num_layers: Optional[int] = None,
        timesfm_num_heads: Optional[int] = None,
        timesfm_model_dims: Optional[int] = None,
        use_revin_norm: bool = True,
        use_gt: bool = True,
        use_revin_denorm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.output_patch_len = output_patch_len
        self.decode_index = decode_index
        self.num_quantiles = num_quantiles
        self.use_quantile_loss = use_quantile_loss
        self.quantile_loss_weight = quantile_loss_weight
        self.timesfm_num_layers = timesfm_num_layers
        self.timesfm_num_heads = timesfm_num_heads
        self.timesfm_model_dims = timesfm_model_dims
        self.use_revin_norm = use_revin_norm
        self.use_revin_denorm = use_revin_denorm
        self.use_gt = use_gt


class TimesFM2p5ForTraining(nn.Module):
    def __init__(
        self,
        torch_dtype: Optional[torch.dtype] = None,
        use_quantile_loss: bool = True,
        quantile_loss_weight: float = 1.0,
        timesfm_num_layers: Optional[int] = None,
        timesfm_num_heads: Optional[int] = None,
        timesfm_model_dims: Optional[int] = None,
        use_revin_norm: bool = True,
        use_gt: bool = True,
        use_revin_denorm: bool = True,
    ):
        super().__init__()
        self._ensure_timesfm_importable()

        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        from timesfm.torch import util as timesfm_util

        self.timesfm_util = timesfm_util
        self.backbone = TimesFM_2p5_200M_torch_module(
            num_layers=timesfm_num_layers,
            num_heads=timesfm_num_heads,
            model_dims=timesfm_model_dims,
        )

        self.use_revin_norm = bool(use_revin_norm)
        self.use_revin_denorm = bool(use_revin_denorm)
        self.use_gt = bool(use_gt)

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
            timesfm_num_layers=self.backbone.x,
            timesfm_num_heads=self.backbone.h,
            timesfm_model_dims=self.backbone.md,
            use_revin_norm=self.use_revin_norm,
            use_revin_denorm=self.use_revin_denorm,
            use_gt=self.use_gt,
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
        timesfm_num_layers: Optional[int] = None,
        timesfm_num_heads: Optional[int] = None,
        timesfm_model_dims: Optional[int] = None,
        use_revin_norm: bool = True,
        use_revin_denorm: bool = True,
        use_gt: bool = True,
    ):
        model = cls(
            torch_dtype=torch_dtype,
            use_quantile_loss=use_quantile_loss,
            quantile_loss_weight=quantile_loss_weight,
            timesfm_num_layers=timesfm_num_layers,
            timesfm_num_heads=timesfm_num_heads,
            timesfm_model_dims=timesfm_model_dims,
            use_revin_norm=use_revin_norm,
            use_revin_denorm=use_revin_denorm,
            use_gt=use_gt,
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

            batch_size, seq_len = full_series.shape
            
            # 2. 计算可以切分多少个 Input Patch
            num_patches = seq_len // self.patch_len
            
            # 截断多余的部分，保证是 patch_len 的倍数
            valid_len = num_patches * self.patch_len
            input_series = full_series[:, :valid_len]
            input_mask_series = full_valid[:, :valid_len].clone()
            if num_patches < 2:
                raise ValueError(f"Sequence too short: need at least 2 patches, got {num_patches}")

    
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


            max_hist_last_patch_idx = num_patches - 4
            if max_hist_last_patch_idx <= 0:
                raise ValueError(
                    f"Sequence too short for random hist/gt split: num_patches={num_patches}, require >=5"
                )
            hist_last_patch_idx = torch.randint(
                low=0,
                high=max_hist_last_patch_idx,
                size=(1,),
                device=device,
            ).item()
            hist_len = (hist_last_patch_idx + 1) * self.patch_len

            hist = full_series[:, :hist_len]
            gt = full_series[:, hist_len:valid_len]

            hist_valid = full_valid[:, :hist_len]
            gt_valid = full_valid[:, hist_len:valid_len]

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
            model_dtype = next(self.backbone.parameters()).dtype
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
            
            # 反归一化输出
            if self.use_revin_denorm:
                output_ts = self.timesfm_util.revin(output_ts, context_mu, context_sigma, reverse=True)
                output_quantile_spread = self.timesfm_util.revin(output_quantile_spread, context_mu, context_sigma, reverse=True)
            
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
            source_for_targets = full_series[:, target_start:valid_len]
            mask_for_targets = full_valid[:, target_start:valid_len]
            source_for_MSE = full_series[:, self.patch_len:]
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

            for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
                pred_aligned[:, :continuous_quantile_patches, :, quantile_index] = (
                    quantile_spread_patches[:, :continuous_quantile_patches, :self.output_patch_len, quantile_index]
                    - quantile_spread_patches[:, :continuous_quantile_patches, :self.output_patch_len, self.decode_index]
                    + pred_aligned[:, :continuous_quantile_patches, :self.output_patch_len, self.decode_index]
                )
            pred_aligned_mean = output_ts[:, :min_patch_mse, :, 0]
            targets_aligned = targets_unfolded[:, :min_patches, :]
            masks_aligned = masks_unfolded[:, :min_patches, :]
            targets_aligned_mse = mse_source_unfolded[:, :min_patch_mse, :]
            masks_aligned_mse = mse_masks_unfolded[:, :min_patch_mse, :]
            
            # 7. 计算 Loss
            point_loss = (pred_aligned_mean - targets_aligned_mse) ** 2
            
            # 应用 Mask (包括原始数据的 mask 和 边界不足导致的截断)
            weighted_loss = point_loss * masks_aligned_mse
            valid_count = torch.clamp(masks_aligned_mse.sum(), min=1.0)
            train_loss = weighted_loss.sum() / valid_count
            quantile_loss_sum = train_loss.new_zeros(())
            loss = train_loss

            # 8. 分位数 Loss (Optional)
            if self.use_quantile_loss:
                quantile_preds = pred_aligned[:, :continuous_quantile_patches, :, 1:] # [B, N, 128, Q]
                quantile_targets = targets_aligned[:, :continuous_quantile_patches, :].unsqueeze(-1)   # [B, N, 128, 1]
                quantile_masks = masks_aligned[:, :continuous_quantile_patches, :].unsqueeze(-1)       # [B, N, 128, 1]
                quantile_valid_count = torch.clamp(quantile_masks.sum(), min=1.0)

                for q_idx, q_val in enumerate(self.quantiles):
                    q_pred = quantile_preds[:, :, :, q_idx]
                    q_loss = self._quantile_loss(q_pred, quantile_targets[..., 0], q_val)
                    quantile_loss_sum += (q_loss * quantile_masks[..., 0]).sum() / quantile_valid_count

                loss = train_loss + self.quantile_loss_weight * quantile_loss_sum
            if not return_dict:
                return (loss, pred_aligned_mean, train_loss, quantile_loss_sum)

            return TimesFM2p5CausalLMOutput(
                loss=loss,
                logits=pred_aligned_mean, # 返回对齐后的预测值
                train_loss=train_loss,
                quantile_loss_sum=quantile_loss_sum,
                quantile_logits=pred_aligned,
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

            batch_size, seq_len = full_series.shape
            
            # 2. 计算可以切分多少个 Input Patch
            num_patches = seq_len // self.patch_len
            
            # 截断多余的部分，保证是 patch_len 的倍数
            valid_len = num_patches * self.patch_len
            input_series = full_series[:, :valid_len]
            input_mask_series = full_valid[:, :valid_len]
            
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
            model_dtype = next(self.backbone.parameters()).dtype
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

            # 8. 分位数 Loss (Optional)
            if self.use_quantile_loss:
                quantile_preds = quantile_preds_full[:, :, :, 1:] # [B, N, 128, Q]
                quantile_targets = targets_aligned.unsqueeze(-1)   # [B, N, 128, 1]
                quantile_masks = masks_aligned.unsqueeze(-1)       # [B, N, 128, 1]

                for q_idx, q_val in enumerate(self.quantiles):
                    q_pred = quantile_preds[:, :, :, q_idx]
                    q_loss = self._quantile_loss(q_pred, quantile_targets[..., 0], q_val)
                    quantile_loss_sum += (q_loss * quantile_masks[..., 0]).sum() / valid_count

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


