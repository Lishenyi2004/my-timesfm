import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESFM_SRC = PROJECT_ROOT / "timesfm" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if TIMESFM_SRC.exists() and str(TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(TIMESFM_SRC))

from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.models.modeling_timesfm_2p5 import TimesFM2p5ForTraining
from timesfm import configs
from timesfm.timesfm_2p5 import timesfm_2p5_base, timesfm_2p5_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug one-step train/infer IO for TimesFM2.5 overfit setup. "
            "Builds both paths with random frozen weights and saves detailed tensors/statistics to JSON."
        )
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to Time-300B data")
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/time300b_train_infer_io_debug.json",
        help="Where to save debug json",
    )
    parser.add_argument(
        "--normalization-method",
        type=str,
        choices=["none", "zero", "max"],
        default="zero",
        help="Must match training/eval normalization",
    )
    parser.add_argument(
        "--target-seq-idx",
        type=int,
        default=0,
        help="Target source sequence index",
    )
    parser.add_argument(
        "--window-offset",
        type=int,
        default=0,
        help="Window start offset in source sequence",
    )
    parser.add_argument(
        "--train-max-length",
        type=int,
        default=512,
        help="Training max_length (window input length)",
    )
    parser.add_argument(
        "--overfit-hist-length",
        type=int,
        default=384,
        help="Fixed history length used in training forward",
    )
    parser.add_argument(
        "--overfit-gt-length",
        type=int,
        default=128,
        help="Fixed gt length used in training forward",
    )
    parser.add_argument(
        "--per-core-batch-size",
        type=int,
        default=1,
        help="Forecast per_core_batch_size",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=4096,
        help="Max context length for compile context resolution",
    )
    parser.add_argument(
        "--forecast-normalize-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match eval script compile(normalize_inputs=True/False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260410,
        help="Random seed for Python/NumPy/PyTorch",
    )
    parser.add_argument(
        "--max-dump-values",
        type=int,
        default=512,
        help="Max number of flattened values stored per tensor field in JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for one-step run",
    )
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def summarize_tensor(x: Any, max_dump_values: int) -> Dict[str, Any]:
    arr = to_numpy(x)
    flat = arr.reshape(-1)
    out: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(flat.size),
    }
    if flat.size > 0 and np.issubdtype(arr.dtype, np.number):
        out.update(
            {
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
            }
        )

    if flat.size <= max_dump_values:
        out["values"] = flat.astype(np.float64).tolist() if np.issubdtype(arr.dtype, np.number) else flat.tolist()
    else:
        head_len = max_dump_values // 2
        tail_len = max_dump_values - head_len
        out["head"] = flat[:head_len].astype(np.float64).tolist() if np.issubdtype(arr.dtype, np.number) else flat[:head_len].tolist()
        out["tail"] = flat[-tail_len:].astype(np.float64).tolist() if np.issubdtype(arr.dtype, np.number) else flat[-tail_len:].tolist()
        out["truncated"] = True
    return out


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def freeze_model_params(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def build_train_eval_aligned_window(
    dataset: TimeMoEDataset,
    target_seq_idx: int,
    window_offset: int,
    train_max_length: int,
    overfit_hist_length: int,
    overfit_gt_length: int,
) -> Dict[str, np.ndarray]:
    if train_max_length <= 0:
        raise ValueError(f"train_max_length must be positive, got {train_max_length}")
    if overfit_hist_length <= 0 or overfit_gt_length <= 0:
        raise ValueError("overfit_hist_length and overfit_gt_length must be positive")
    if overfit_hist_length + overfit_gt_length != train_max_length:
        raise ValueError(
            "For strict one-window alignment, require overfit_hist_length + overfit_gt_length == train_max_length, "
            f"got {overfit_hist_length} + {overfit_gt_length} != {train_max_length}"
        )
    if target_seq_idx < 0 or target_seq_idx >= len(dataset):
        raise ValueError(f"target_seq_idx out of range: {target_seq_idx}")
    if window_offset < 0:
        raise ValueError(f"window_offset must be >= 0, got {window_offset}")

    seq = np.asarray(dataset[target_seq_idx], dtype=np.float32)
    raw_len = int(seq.shape[0])
    window_size_plus_one = train_max_length + 1
    end_idx = window_offset + window_size_plus_one
    if end_idx > raw_len:
        raise ValueError(
            f"Sequence too short for selected window: len={raw_len}, need >= {end_idx}"
        )

    seq_slice = seq[window_offset:end_idx]  # len = train_max_length + 1
    train_input_ids = seq_slice[:-1]  # len = train_max_length
    train_labels = seq_slice[1:]      # shifted labels from dataset window (TimeMoEWindowDataset behavior)
    train_loss_masks = np.ones_like(train_input_ids, dtype=np.float32)

    context = train_input_ids[:overfit_hist_length]
    horizon_label = train_input_ids[overfit_hist_length: overfit_hist_length + overfit_gt_length]

    return {
        "full_sequence": seq,
        "window_seq_plus_one": seq_slice,
        "train_input_ids": train_input_ids,
        "train_labels_shifted": train_labels,
        "train_loss_masks": train_loss_masks,
        "eval_context": context,
        "eval_label": horizon_label,
    }


@torch.no_grad()
def trace_train_forward(
    model: TimesFM2p5ForTraining,
    input_ids_np: np.ndarray,
    loss_masks_np: np.ndarray,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    input_ids = torch.from_numpy(input_ids_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    loss_masks = torch.from_numpy(loss_masks_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    full_series = input_ids
    full_valid = loss_masks
    batch_size, seq_len = full_series.shape

    num_patches = seq_len // model.patch_len
    valid_len = num_patches * model.patch_len
    input_series = full_series[:, :valid_len]
    input_mask_series = full_valid[:, :valid_len].clone()

    if not model.enable_overfit_fixed_window:
        raise RuntimeError("trace_train_forward expects enable_overfit_fixed_window=True")

    hist_len = model.overfit_hist_length
    hist_last_patch_idx = (hist_len // model.patch_len) - 1

    patched_inputs = input_series.reshape(batch_size, num_patches, model.patch_len)
    patched_masks = torch.logical_not(
        input_mask_series.reshape(batch_size, num_patches, model.patch_len).bool()
    )

    context_mu, context_sigma = model._get_patch_stats(patched_inputs, patched_masks)
    if model.use_revin_norm:
        normed_inputs = model.timesfm_util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
    else:
        normed_inputs = patched_inputs
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
    model_dtype = next(model.backbone.parameters()).dtype
    normed_inputs = normed_inputs.to(dtype=model_dtype)

    (_, _, output_ts, output_quantile_spread), _ = model.backbone(normed_inputs, patched_masks)
    (_, _, flipped_output_ts, flipped_output_quantile_spread), _ = model.backbone(-normed_inputs, patched_masks)

    output_ts = output_ts.reshape(batch_size, num_patches, model.output_patch_len, model.num_quantiles)
    flipped_output_ts = flipped_output_ts.reshape(batch_size, num_patches, model.output_patch_len, model.num_quantiles)
    flipped_output_ts = model._flip_quantile_dim(flipped_output_ts)
    output_ts = 0.5 * (output_ts - flipped_output_ts)

    output_quantile_spread = output_quantile_spread.reshape(batch_size, num_patches, -1, model.num_quantiles)
    flipped_output_quantile_spread = flipped_output_quantile_spread.reshape(
        batch_size, num_patches, -1, model.num_quantiles
    )
    flipped_output_quantile_spread = model._flip_quantile_dim(flipped_output_quantile_spread)
    output_quantile_spread = 0.5 * (output_quantile_spread - flipped_output_quantile_spread)
    output_quantile_spread = output_quantile_spread.reshape(batch_size, num_patches, -1)

    if model.use_revin_denorm:
        output_ts = model.timesfm_util.revin(output_ts, context_mu, context_sigma, reverse=True)
        output_quantile_spread = model.timesfm_util.revin(
            output_quantile_spread, context_mu, context_sigma, reverse=True
        )

    full_quantile_spread = output_quantile_spread[:, hist_last_patch_idx, :]
    quantile_spread_flat = full_quantile_spread.reshape(batch_size, 1024, model.num_quantiles)
    quantile_spread_unfolded = quantile_spread_flat.unfold(1, model.output_patch_len, model.patch_len)
    quantile_spread_patches = quantile_spread_unfolded.permute(0, 1, 3, 2).contiguous()

    target_start = hist_len
    pred_start_idx = hist_last_patch_idx
    target_end = min(valid_len, hist_len + model.overfit_gt_length)

    source_for_targets = full_series[:, target_start:target_end]
    mask_for_targets = full_valid[:, target_start:target_end]
    source_for_mse = full_series[:, model.patch_len:]
    mask_for_mse = full_valid[:, model.patch_len:]

    targets_unfolded = source_for_targets.unfold(1, model.output_patch_len, model.patch_len)
    masks_unfolded = mask_for_targets.unfold(1, model.output_patch_len, model.patch_len)
    mse_source_unfolded = source_for_mse.unfold(1, model.output_patch_len, model.patch_len)
    mse_masks_unfolded = mask_for_mse.unfold(1, model.output_patch_len, model.patch_len)

    pred_available = output_ts.shape[1] - pred_start_idx
    target_available = targets_unfolded.shape[1]
    min_patches = min(pred_available, target_available)
    min_patch_mse = min(output_ts.shape[1], mse_source_unfolded.shape[1])

    pred_aligned = output_ts[:, pred_start_idx:pred_start_idx + min_patches, :, :].clone()
    continuous_quantile_patches = min(min_patches, quantile_spread_patches.shape[1])

    for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
        pred_aligned[:, :continuous_quantile_patches, :, quantile_index] = (
            quantile_spread_patches[:, :continuous_quantile_patches, :model.output_patch_len, quantile_index]
            - quantile_spread_patches[:, :continuous_quantile_patches, :model.output_patch_len, model.decode_index]
            + pred_aligned[:, :continuous_quantile_patches, :model.output_patch_len, model.decode_index]
        )

    pred_aligned_mean = output_ts[:, :min_patch_mse, :, model.decode_index]
    targets_aligned_mse = mse_source_unfolded[:, :min_patch_mse, :]
    masks_aligned_mse = mse_masks_unfolded[:, :min_patch_mse, :]

    point_loss = (pred_aligned_mean - targets_aligned_mse) ** 2
    weighted_loss = point_loss * masks_aligned_mse
    valid_count = torch.clamp(masks_aligned_mse.sum(), min=1.0)
    train_loss = weighted_loss.sum() / valid_count

    model_out = model(input_ids=input_ids, loss_masks=loss_masks, return_dict=True)

    return {
        "train_forward": {
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "num_patches": int(num_patches),
            "valid_len": int(valid_len),
            "hist_len": int(hist_len),
            "hist_last_patch_idx": int(hist_last_patch_idx),
            "target_start": int(target_start),
            "target_end": int(target_end),
            "pred_start_idx": int(pred_start_idx),
            "min_patches": int(min_patches),
            "min_patch_mse": int(min_patch_mse),
        },
        "train_tensors": {
            "input_ids": input_ids,
            "loss_masks": loss_masks,
            "patched_inputs": patched_inputs,
            "patched_masks": patched_masks,
            "context_mu": context_mu,
            "context_sigma": context_sigma,
            "normed_inputs": normed_inputs,
            "output_ts_after_flip_and_denorm": output_ts,
            "output_quantile_spread_after_flip_and_denorm": output_quantile_spread,
            "pred_aligned_eval_window_full_quantiles": pred_aligned,
            "pred_aligned_mean_for_mse": pred_aligned_mean,
            "targets_aligned_for_mse": targets_aligned_mse,
            "masks_aligned_for_mse": masks_aligned_mse,
            "point_loss": point_loss,
            "train_loss_recomputed": train_loss,
            "train_loss_model_forward": model_out.train_loss,
            "loss_model_forward": model_out.loss,
            "logits_model_forward": model_out.logits,
            "quantile_logits_model_forward": model_out.quantile_logits,
        },
    }


@torch.no_grad()
def trace_decode_internals(
    tfm_model: torch.nn.Module,
    horizon: int,
    inputs: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    batch_size, context = inputs.shape[0], inputs.shape[1]
    num_decode_steps = (horizon - 1) // tfm_model.o
    num_input_patches = context // tfm_model.p
    decode_cache_size = num_input_patches + num_decode_steps * tfm_model.m

    patched_inputs = torch.reshape(inputs, (batch_size, -1, tfm_model.p))
    patched_masks = torch.reshape(masks, (batch_size, -1, tfm_model.p))

    n = torch.zeros(batch_size, device=inputs.device)
    mu = torch.zeros(batch_size, device=inputs.device)
    sigma = torch.zeros(batch_size, device=inputs.device)
    patch_mu = []
    patch_sigma = []
    for i in range(num_input_patches):
        (n, mu, sigma), _ = timesfm_2p5_torch.util.update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    last_n, last_mu, last_sigma = n, mu, sigma
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    decode_caches = [
        timesfm_2p5_torch.util.DecodeCache(
            next_index=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
            num_masked=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
            key=torch.zeros(
                batch_size,
                decode_cache_size,
                tfm_model.h,
                tfm_model.hd,
                device=inputs.device,
            ),
            value=torch.zeros(
                batch_size,
                decode_cache_size,
                tfm_model.h,
                tfm_model.hd,
                device=inputs.device,
            ),
        )
        for _ in range(tfm_model.x)
    ]

    normed_inputs = timesfm_2p5_torch.revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
    (_, _, normed_outputs, normed_quantile_spread), decode_caches = tfm_model(
        normed_inputs, patched_masks, decode_caches
    )

    renormed_outputs = torch.reshape(
        timesfm_2p5_torch.revin(normed_outputs, context_mu, context_sigma, reverse=True),
        (batch_size, -1, tfm_model.o, tfm_model.q),
    )
    renormed_quantile_spread = torch.reshape(
        timesfm_2p5_torch.revin(normed_quantile_spread, context_mu, context_sigma, reverse=True),
        (batch_size, -1, tfm_model.os, tfm_model.q),
    )[:, -1, ...]

    ar_outputs = []
    last_renormed_output = renormed_outputs[:, -1, :, tfm_model.aridx]

    for _ in range(num_decode_steps):
        new_patched_input = torch.reshape(last_renormed_output, (batch_size, tfm_model.m, tfm_model.p))
        new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)

        n, mu, sigma = last_n, last_mu, last_sigma
        new_mus, new_sigmas = [], []
        for i in range(tfm_model.m):
            (n, mu, sigma), _ = timesfm_2p5_torch.util.update_running_stats(
                n, mu, sigma, new_patched_input[:, i], new_mask[:, i]
            )
            new_mus.append(mu)
            new_sigmas.append(sigma)
        last_n, last_mu, last_sigma = n, mu, sigma
        new_mu = torch.stack(new_mus, dim=1)
        new_sigma = torch.stack(new_sigmas, dim=1)

        new_normed_input = timesfm_2p5_torch.revin(new_patched_input, new_mu, new_sigma, reverse=False)
        (_, _, new_normed_output, _), decode_caches = tfm_model(new_normed_input, new_mask, decode_caches)

        new_renormed_output = torch.reshape(
            timesfm_2p5_torch.revin(new_normed_output, new_mu, new_sigma, reverse=True),
            (batch_size, tfm_model.m, tfm_model.o, tfm_model.q),
        )
        ar_outputs.append(new_renormed_output[:, -1, ...])
        last_renormed_output = new_renormed_output[:, -1, :, tfm_model.aridx]

    if num_decode_steps > 0:
        ar_renormed_outputs = torch.stack(ar_outputs, dim=1)
    else:
        ar_renormed_outputs = None

    trace = {
        "decode_meta": {
            "batch_size": int(batch_size),
            "context": int(context),
            "horizon": int(horizon),
            "num_decode_steps": int(num_decode_steps),
            "num_input_patches": int(num_input_patches),
            "decode_cache_size": int(decode_cache_size),
        },
        "decode_tensors": {
            "inputs": inputs,
            "masks": masks,
            "patched_inputs": patched_inputs,
            "patched_masks": patched_masks,
            "context_mu": context_mu,
            "context_sigma": context_sigma,
            "normed_inputs": normed_inputs,
            "normed_outputs": normed_outputs,
            "normed_quantile_spread": normed_quantile_spread,
            "renormed_outputs": renormed_outputs,
            "renormed_quantile_spread": renormed_quantile_spread,
            "ar_renormed_outputs": ar_renormed_outputs,
        },
    }
    return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs, trace


@torch.no_grad()
def traced_compiled_decode(
    tfm: timesfm_2p5_torch.TimesFM_2p5_200M_torch,
    forecast_config: configs.ForecastConfig,
    horizon: int,
    values: List[np.ndarray],
    masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    fc = forecast_config
    model = tfm.model

    if horizon > fc.max_horizon:
        raise ValueError(f"horizon {horizon} > max_horizon {fc.max_horizon}")

    inputs = torch.from_numpy(np.array(values)).to(model.device).to(torch.float32)
    masks_t = torch.from_numpy(np.array(masks)).to(model.device).to(torch.bool)

    if fc.infer_is_positive:
        is_positive = torch.all(inputs >= 0, dim=-1, keepdim=True)
    else:
        is_positive = None

    if fc.normalize_inputs:
        mu = torch.mean(inputs, dim=-1, keepdim=True)
        sigma = torch.std(inputs, dim=-1, keepdim=True)
        decode_inputs = timesfm_2p5_torch.revin(inputs, mu, sigma, reverse=False)
    else:
        mu, sigma = None, None
        decode_inputs = inputs

    pf_outputs, quantile_spreads, ar_outputs, decode_trace = trace_decode_internals(
        tfm_model=model,
        horizon=fc.max_horizon,
        inputs=decode_inputs,
        masks=masks_t,
    )

    batch_size = inputs.shape[0]
    to_cat = [pf_outputs[:, -1, ...]]
    if ar_outputs is not None:
        to_cat.append(ar_outputs.reshape(batch_size, -1, model.q))
    full_forecast = torch.cat(to_cat, dim=1)

    def flip_quantile_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)

    flipped_trace = None
    if fc.force_flip_invariance:
        flipped_pf_outputs, flipped_quantile_spreads, flipped_ar_outputs, flipped_decode_trace = trace_decode_internals(
            tfm_model=model,
            horizon=fc.max_horizon,
            inputs=-decode_inputs,
            masks=masks_t,
        )
        flipped_quantile_spreads = flip_quantile_fn(flipped_quantile_spreads)
        flipped_pf_outputs = flip_quantile_fn(flipped_pf_outputs)

        flipped_to_cat = [flipped_pf_outputs[:, -1, ...]]
        if flipped_ar_outputs is not None:
            flipped_to_cat.append(flipped_ar_outputs.reshape(batch_size, -1, model.q))
        flipped_full_forecast = torch.cat(flipped_to_cat, dim=1)

        quantile_spreads = (quantile_spreads - flipped_quantile_spreads) / 2
        pf_outputs = (pf_outputs - flipped_pf_outputs) / 2
        full_forecast = (full_forecast - flipped_full_forecast) / 2
        flipped_trace = flipped_decode_trace

    if fc.use_continuous_quantile_head:
        for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
            full_forecast[:, :, quantile_index] = (
                quantile_spreads[:, : fc.max_horizon, quantile_index]
                - quantile_spreads[:, : fc.max_horizon, 5]
                + full_forecast[:, : fc.max_horizon, 5]
            )

    full_forecast = full_forecast[:, :horizon, :]

    if fc.return_backcast:
        full_backcast = pf_outputs[:, :-1, : model.p, :].reshape(batch_size, -1, model.q)
        full_forecast = torch.cat([full_backcast, full_forecast], dim=1)

    if fc.fix_quantile_crossing:
        for i in [4, 3, 2, 1]:
            full_forecast[:, :, i] = torch.where(
                full_forecast[:, :, i] < full_forecast[:, :, i + 1],
                full_forecast[:, :, i],
                full_forecast[:, :, i + 1],
            )
        for i in [6, 7, 8, 9]:
            full_forecast[:, :, i] = torch.where(
                full_forecast[:, :, i] > full_forecast[:, :, i - 1],
                full_forecast[:, :, i],
                full_forecast[:, :, i - 1],
            )

    if fc.normalize_inputs:
        full_forecast = timesfm_2p5_torch.revin(full_forecast, mu, sigma, reverse=True)

    if is_positive is not None:
        full_forecast = torch.where(
            is_positive[..., None],
            torch.maximum(full_forecast, torch.zeros_like(full_forecast)),
            full_forecast,
        )

    full_forecast_np = full_forecast.detach().cpu().numpy()
    trace = {
        "compiled_decode_tensors": {
            "inputs_before_optional_norm": inputs,
            "masks": masks_t,
            "decode_inputs_after_optional_norm": decode_inputs,
            "full_forecast_after_all_postprocess": full_forecast,
        },
        "decode_positive": decode_trace,
        "decode_negative": flipped_trace,
    }
    return full_forecast_np[..., 5], full_forecast_np, trace


def traced_forecast(
    tfm: timesfm_2p5_torch.TimesFM_2p5_200M_torch,
    forecast_config: configs.ForecastConfig,
    horizon: int,
    inputs: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    context = forecast_config.max_context
    num_inputs = len(inputs)
    assert tfm.global_batch_size > 0

    input_traces = []
    values, masks = [], []
    output_points, output_quantiles = [], []

    padded_inputs = list(inputs)
    if (w := num_inputs % tfm.global_batch_size) != 0:
        padded_inputs = padded_inputs + [np.array([0.0] * 3)] * (tfm.global_batch_size - w)

    idx = 0
    for each_input in padded_inputs:
        raw_arr = np.array(each_input)
        value = timesfm_2p5_base.linear_interpolation(timesfm_2p5_base.strip_leading_nans(raw_arr.copy()))
        if (wlen := len(value)) >= context:
            value_ctx = value[-context:]
            mask = np.zeros_like(value_ctx, dtype=bool)
            pad_mode = "truncate_or_exact"
        else:
            mask = np.array([True] * (context - wlen) + [False] * wlen)
            value_ctx = np.pad(value, (context - wlen, 0), "constant", constant_values=0.0)
            pad_mode = "left_pad_zero"

        input_traces.append(
            {
                "raw_length": int(len(raw_arr)),
                "post_nan_clean_length": int(len(value)),
                "context_length": int(len(value_ctx)),
                "pad_mode": pad_mode,
                "pad_count": int(np.sum(mask)),
            }
        )

        values.append(value_ctx)
        masks.append(mask)
        idx += 1
        if idx == tfm.global_batch_size:
            idx = 0
            point_forecast, quantile_forecast, decode_trace = traced_compiled_decode(
                tfm=tfm,
                forecast_config=forecast_config,
                horizon=horizon,
                values=values,
                masks=masks,
            )
            output_points.append(point_forecast)
            output_quantiles.append(quantile_forecast)
            values, masks = [], []

    output_points = np.concatenate(output_points, axis=0)
    output_quantiles = np.concatenate(output_quantiles, axis=0)
    final_trace = {
        "forecast_meta": {
            "num_inputs": int(num_inputs),
            "global_batch_size": int(tfm.global_batch_size),
            "max_context": int(forecast_config.max_context),
            "max_horizon": int(forecast_config.max_horizon),
        },
        "per_input_preprocess": input_traces,
        "compiled_decode_trace_first_batch": decode_trace,
    }
    return output_points[:num_inputs], output_quantiles[:num_inputs], final_trace


def build_forecast_config(
    tfm: timesfm_2p5_torch.TimesFM_2p5_200M_torch,
    context_len: int,
    horizon: int,
    max_context_length: int,
    per_core_batch_size: int,
    normalize_inputs: bool,
) -> Tuple[configs.ForecastConfig, int]:
    max_context = ((context_len + tfm.model.p - 1) // tfm.model.p) * tfm.model.p
    compile_max_context = min(max_context, max_context_length)
    compile_max_context = max(tfm.model.p, compile_max_context)
    compile_max_context = (compile_max_context // tfm.model.p) * tfm.model.p
    if compile_max_context <= 0:
        compile_max_context = tfm.model.p

    fc = configs.ForecastConfig(
        max_context=compile_max_context,
        max_horizon=horizon,
        infer_is_positive=True,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
        force_flip_invariance=True,
        return_backcast=False,
        normalize_inputs=normalize_inputs,
        per_core_batch_size=per_core_batch_size,
    )
    return fc, compile_max_context


def summarize_block(block: Dict[str, Any], max_dump_values: int) -> Dict[str, Any]:
    summarized: Dict[str, Any] = {}
    for k, v in block.items():
        if isinstance(v, dict):
            summarized[k] = summarize_block(v, max_dump_values)
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            summarized[k] = summarize_tensor(v, max_dump_values)
        else:
            summarized[k] = v
    return summarized


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    normalization_method = None if args.normalization_method == "none" else args.normalization_method
    device = resolve_device(args.device)

    dataset = TimeMoEDataset(
        args.data_path,
        normalization_method=normalization_method,
        max_sequences=max(args.target_seq_idx + 1, 1),
    )

    aligned = build_train_eval_aligned_window(
        dataset=dataset,
        target_seq_idx=args.target_seq_idx,
        window_offset=args.window_offset,
        train_max_length=args.train_max_length,
        overfit_hist_length=args.overfit_hist_length,
        overfit_gt_length=args.overfit_gt_length,
    )

    train_model = TimesFM2p5ForTraining(
        torch_dtype=torch.float32,
        use_quantile_loss=True,
        quantile_loss_weight=1.0,
        use_revin_norm=True,
        use_revin_denorm=True,
        use_gt=True,
        enable_overfit_fixed_window=True,
        overfit_hist_length=args.overfit_hist_length,
        overfit_gt_length=args.overfit_gt_length,
    ).to(device)
    freeze_model_params(train_model)

    infer_wrapper = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
    infer_wrapper.model = timesfm_2p5_torch.TimesFM_2p5_200M_torch_module().to(device)
    infer_wrapper.model.load_state_dict(train_model.backbone.state_dict(), strict=True)
    freeze_model_params(infer_wrapper.model)

    # Ensure both paths use identical random initialized weights.
    weights_l2 = 0.0
    for p_train, p_infer in zip(train_model.backbone.parameters(), infer_wrapper.model.parameters()):
        weights_l2 += float(torch.sum((p_train.detach() - p_infer.detach()) ** 2).item())

    train_trace_raw = trace_train_forward(
        model=train_model,
        input_ids_np=aligned["train_input_ids"],
        loss_masks_np=aligned["train_loss_masks"],
    )

    context = aligned["eval_context"].astype(np.float32)
    label = aligned["eval_label"].astype(np.float32)

    forecast_config, compile_max_context = build_forecast_config(
        tfm=infer_wrapper,
        context_len=context.shape[0],
        horizon=label.shape[0],
        max_context_length=args.max_context_length,
        per_core_batch_size=args.per_core_batch_size,
        normalize_inputs=args.forecast_normalize_inputs,
    )
    infer_wrapper.forecast_config = forecast_config
    infer_wrapper.global_batch_size = forecast_config.per_core_batch_size * infer_wrapper.model.device_count

    infer_point, infer_full_quantiles, infer_trace_raw = traced_forecast(
        tfm=infer_wrapper,
        forecast_config=forecast_config,
        horizon=label.shape[0],
        inputs=[context],
    )

    train_normed_inputs_full = train_trace_raw["train_tensors"]["normed_inputs"].detach().cpu().numpy()
    # Compare only the history/context-aligned prefix patches used by inference decode.
    hist_patch_count = args.overfit_hist_length // train_model.patch_len
    train_normed_inputs_hist = train_normed_inputs_full[:, :hist_patch_count, :]

    infer_decode_positive = infer_trace_raw["compiled_decode_trace_first_batch"]["decode_positive"]
    infer_normed_inputs = infer_decode_positive["decode_tensors"]["normed_inputs"].detach().cpu().numpy()

    if train_normed_inputs_hist.shape != infer_normed_inputs.shape:
        raise RuntimeError(
            "train/infer normed_inputs shape mismatch: "
            f"train_hist={train_normed_inputs_hist.shape}, infer={infer_normed_inputs.shape}"
        )

    normed_inputs_diff = np.abs(train_normed_inputs_hist - infer_normed_inputs)

    infer_q50 = infer_full_quantiles[0, :, 5]
    infer_q50_index4_from_slice = infer_full_quantiles[0, :, 1:10][:, 4]

    train_eval_q50 = (
        train_trace_raw["train_tensors"]["pred_aligned_eval_window_full_quantiles"]
        .detach()
        .cpu()
        .numpy()[0, 0, :, 5]
    )

    train_context_from_input = aligned["train_input_ids"][: args.overfit_hist_length]

    checks = {
        "weights_l2_diff_train_backbone_vs_infer_model": weights_l2,
        "context_max_abs_diff_train_vs_eval": float(np.max(np.abs(train_context_from_input - context))),
        "label_max_abs_diff_train_eval_label_vs_train_input_tail": float(
            np.max(np.abs(label - aligned["train_input_ids"][args.overfit_hist_length:]))
        ),
        "infer_q50_vs_label_mse": float(np.mean((infer_q50 - label) ** 2)),
        "train_eval_q50_vs_label_mse": float(np.mean((train_eval_q50 - label) ** 2)),
        "train_eval_q50_vs_infer_q50_max_abs_diff": float(np.max(np.abs(train_eval_q50 - infer_q50))),
        "infer_q50_index5_vs_index4_from_1_10_slice_max_abs_diff": float(
            np.max(np.abs(infer_q50 - infer_q50_index4_from_slice))
        ),
        "train_hist_normed_inputs_vs_infer_normed_inputs_max_abs_diff": float(np.max(normed_inputs_diff)),
        "train_hist_normed_inputs_vs_infer_normed_inputs_mean_abs_diff": float(np.mean(normed_inputs_diff)),
        "note_normed_inputs_compare": (
            "If forecast_normalize_inputs=True, inference path adds an extra per-series normalize before decode; "
            "intermediate normed_inputs may differ from training path even when final outputs are aligned."
        ),
    }

    report_raw: Dict[str, Any] = {
        "meta": {
            "seed": args.seed,
            "device": str(device),
            "data_path": str(Path(args.data_path).expanduser().resolve()),
            "normalization_method": args.normalization_method,
            "target_seq_idx": args.target_seq_idx,
            "window_offset": args.window_offset,
            "train_max_length": args.train_max_length,
            "overfit_hist_length": args.overfit_hist_length,
            "overfit_gt_length": args.overfit_gt_length,
            "compile_max_context": int(compile_max_context),
            "forecast_normalize_inputs": bool(args.forecast_normalize_inputs),
            "model_frozen": True,
            "model_init": "random_init_no_checkpoint",
        },
        "aligned_data": {
            "full_sequence": aligned["full_sequence"],
            "window_seq_plus_one": aligned["window_seq_plus_one"],
            "train_input_ids": aligned["train_input_ids"],
            "train_labels_shifted": aligned["train_labels_shifted"],
            "train_loss_masks": aligned["train_loss_masks"],
            "eval_context": aligned["eval_context"],
            "eval_label": aligned["eval_label"],
        },
        "train_path": train_trace_raw,
        "infer_path": {
            "forecast_config": {
                "max_context": forecast_config.max_context,
                "max_horizon": forecast_config.max_horizon,
                "normalize_inputs": forecast_config.normalize_inputs,
                "per_core_batch_size": forecast_config.per_core_batch_size,
                "use_continuous_quantile_head": forecast_config.use_continuous_quantile_head,
                "force_flip_invariance": forecast_config.force_flip_invariance,
                "fix_quantile_crossing": forecast_config.fix_quantile_crossing,
                "infer_is_positive": forecast_config.infer_is_positive,
            },
            "infer_point_output": infer_point,
            "infer_full_quantiles": infer_full_quantiles,
            "infer_q50_index5": infer_q50,
            "infer_q50_index4_from_1_10_slice": infer_q50_index4_from_slice,
            "trace": infer_trace_raw,
        },
        "normed_inputs_compare": {
            "train_normed_inputs_hist": train_normed_inputs_hist,
            "infer_normed_inputs": infer_normed_inputs,
            "abs_diff": normed_inputs_diff,
        },
        "consistency_checks": checks,
    }

    report = summarize_block(report_raw, args.max_dump_values)

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_json": str(output_path), "consistency_checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
