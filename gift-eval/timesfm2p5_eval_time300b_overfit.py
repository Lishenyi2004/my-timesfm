import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timesfm.src.timesfm import configs
from timesfm.src.timesfm.timesfm_2p5 import timesfm_2p5_torch


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm


from time_moe.datasets.time_moe_dataset import TimeMoEDataset



QUANTILE_LEVELS = list(np.arange(1, 10) / 10.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overfit evaluation on a small Time-300B subset using fixed tail GT split."
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to Time-300B data")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str("/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm-2.5-200m-pytorch"),
        help="Path to checkpoint dir or model.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("results") / "time300b_overfit_original"),
        help="Output directory for metrics and plots",
    )
    parser.add_argument("--model-name", type=str, default="TimesFM-2.5-original")
    parser.add_argument(
        "--target-seq-idx",
        type=int,
        default=0,
        help="Target sequence index to evaluate (default: 0, i.e., first sequence)",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        choices=["train_window", "series_tail"],
        default="train_window",
        help="How to build eval sample from target sequence: train_window aligns with training window logic",
    )
    parser.add_argument(
        "--train-max-length",
        type=int,
        default=1024,
        help="Training max_length used to construct one training window when sample-mode=train_window",
    )
    parser.add_argument(
        "--train-fixed-gt-length",
        type=int,
        default=256,
        help="Training fixed_gt_length used when sample-mode=train_window",
    )
    parser.add_argument(
        "--window-offset",
        type=int,
        default=0,
        help="Start offset in target sequence for selecting the training window",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=128,
        help="Fixed GT tail length; prediction horizon equals this value",
    )
    parser.add_argument(
        "--min-context-length",
        type=int,
        default=128,
        help="Minimum history length required before the GT tail",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=4096,
        help="Maximum context length sent to model (use most recent points)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--per-core-batch-size",
        type=int,
        default=1,
        help="per_core_batch_size for TimesFM compile/forecast config",
    )
    parser.add_argument(
        "--mase-seasonality",
        type=int,
        default=1,
        help="Seasonality used for MASE scaling denominator (seasonal naive error)",
    )
    parser.add_argument(
        "--normalization-method",
        type=str,
        choices=["none", "zero", "max"],
        default="zero",
        help="Normalization applied by TimeMoEDataset (match training config)",
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--plot-context-len", type=int, default=200)
    parser.add_argument("--max-plot-samples", type=int, default=16)
    return parser.parse_args()


def resolve_checkpoint_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        ckpt = path / "model.safetensors"
    else:
        ckpt = path
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return ckpt


def load_tfm_checkpoint_compatible(tfm, checkpoint_path: Path, torch_compile: bool = False) -> None:
    try:
        tfm.model.load_checkpoint(str(checkpoint_path), torch_compile=torch_compile)
        return
    except RuntimeError as first_error:
        tensors = load_file(str(checkpoint_path))
        if not tensors:
            raise first_error

        if all(k.startswith("backbone.") for k in tensors.keys()):
            remapped = {
                k[len("backbone."):] if k.startswith("backbone.") else k: v
                for k, v in tensors.items()
            }
            tfm.model.load_state_dict(remapped, strict=True)
            tfm.model.to(tfm.model.device)
            if torch_compile:
                print("Compiling model...")
                tfm.model = torch.compile(tfm.model)
            tfm.model.eval()
            print("Loaded checkpoint with auto-remap: stripped 'backbone.' prefix.")
            return

        raise first_error


def _to_1d_float_array(seq) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 1:
        return None
    return arr


def build_overfit_samples(
    dataset: TimeMoEDataset,
    prediction_length: int,
    min_context_length: int,
    target_seq_idx: int,
    sample_mode: str,
    train_max_length: int,
    train_fixed_gt_length: int,
    window_offset: int,
) -> List[Dict[str, np.ndarray]]:
    dataset_size = len(dataset)
    if target_seq_idx < 0 or target_seq_idx >= dataset_size:
        raise ValueError(f"target_seq_idx={target_seq_idx} out of range [0, {dataset_size - 1}]")

    seq = _to_1d_float_array(dataset[target_seq_idx])
    if seq is None:
        raise ValueError(f"Sequence {target_seq_idx} is not a 1D numeric array")

    if sample_mode == "train_window":
        if window_offset < 0:
            raise ValueError(f"window_offset must be non-negative, got {window_offset}")
        if train_max_length <= 0:
            raise ValueError(f"train_max_length must be positive, got {train_max_length}")
        if train_fixed_gt_length <= 0 or train_fixed_gt_length >= train_max_length:
            raise ValueError(
                f"train_fixed_gt_length must be in (0, train_max_length), got "
                f"{train_fixed_gt_length} with train_max_length={train_max_length}"
            )
        if prediction_length > train_fixed_gt_length:
            raise ValueError(
                f"prediction_length={prediction_length} cannot exceed "
                f"train_fixed_gt_length={train_fixed_gt_length} in train_window mode"
            )

        window_size_plus_one = train_max_length + 1
        end_idx = window_offset + window_size_plus_one
        if end_idx > seq.shape[0]:
            raise ValueError(
                f"Sequence {target_seq_idx} too short for train window: len={seq.shape[0]}, "
                f"required >= {end_idx} (window_offset + train_max_length + 1)"
            )

        raw_window = seq[window_offset:end_idx]
        input_window = raw_window[:-1]

        hist_len = train_max_length - train_fixed_gt_length
        context = input_window[:hist_len]
        label = input_window[hist_len:hist_len + prediction_length]
    elif sample_mode == "series_tail":
        if seq.shape[0] < prediction_length + min_context_length:
            raise ValueError(
                f"Sequence {target_seq_idx} too short: len={seq.shape[0]}, "
                f"required >= prediction_length + min_context_length = "
                f"{prediction_length + min_context_length}"
            )
        context = seq[:-prediction_length]
        label = seq[-prediction_length:]
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    samples = [
        {
            "seq_idx": target_seq_idx,
            "context": context,
            "label": label,
        }
    ]

    print(
        f"Collected {len(samples)} series for overfit eval; "
        f"target_seq_idx={target_seq_idx}, sample_mode={sample_mode}, "
        f"context_len={len(context)}, label_len={len(label)}"
    )

    return samples


def predict_quantiles(
    tfm,
    samples: List[Dict[str, np.ndarray]],
    prediction_length: int,
    batch_size: int,
    max_context_length: int,
    per_core_batch_size: int,
) -> np.ndarray:
    forecast_outputs = []
    total_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_id in tqdm(range(total_batches), desc="Forecast"):
        batch = samples[batch_id * batch_size: (batch_id + 1) * batch_size]

        context_batch = []
        max_context = 0
        for sample in batch:
            context = sample["context"]
            if max_context_length is not None and context.shape[0] > max_context_length:
                context = context[-max_context_length:]
            context_batch.append(context)
            max_context = max(max_context, context.shape[0])

        max_context = ((max_context + tfm.model.p - 1) // tfm.model.p) * tfm.model.p
        compile_max_context = min(max_context, max_context_length)
        compile_max_context = max(tfm.model.p, compile_max_context)
        compile_max_context = (compile_max_context // tfm.model.p) * tfm.model.p
        if compile_max_context <= 0:
            compile_max_context = tfm.model.p

        tfm.compile(
            forecast_config=configs.ForecastConfig(
                max_context=compile_max_context,
                max_horizon=1024,
                infer_is_positive=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                return_backcast=False,
                normalize_inputs=True,
                per_core_batch_size=per_core_batch_size,
            ),
        )

        _, full_preds = tfm.forecast(
            horizon=prediction_length,
            inputs=context_batch,
        )

        if full_preds.shape[-1] < 10:
            raise RuntimeError(
                f"Unexpected forecast output shape {full_preds.shape}, expected last dim >= 10"
            )

        quantile_preds = full_preds[:, 0:prediction_length, 1:10].astype(np.float32)
        forecast_outputs.append(quantile_preds)

    return np.concatenate(forecast_outputs, axis=0)


def compute_series_mase(
    contexts: List[np.ndarray],
    labels: np.ndarray,
    median_preds: np.ndarray,
    seasonality: int,
) -> np.ndarray:
    eps = 1e-8
    n_series = labels.shape[0]
    series_mase = np.full(n_series, np.nan, dtype=np.float32)

    for idx in range(n_series):
        context = np.asarray(contexts[idx], dtype=np.float32)
        if context.ndim != 1 or context.shape[0] <= seasonality:
            continue

        scale = float(np.mean(np.abs(context[seasonality:] - context[:-seasonality])))
        if not np.isfinite(scale) or scale <= eps:
            continue

        mae = float(np.mean(np.abs(median_preds[idx] - labels[idx])))
        series_mase[idx] = mae / scale

    return series_mase


def geometric_mean_nonnegative(values: np.ndarray) -> float:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    valid = valid[valid >= 0.0]

    if valid.size == 0:
        return float("nan")

    if np.any(valid == 0.0):
        return 0.0

    return float(np.exp(np.mean(np.log(valid))))


def compute_metrics(
    contexts: List[np.ndarray],
    labels: np.ndarray,
    quantile_preds: np.ndarray,
    mase_seasonality: int,
) -> Dict[str, float]:
    eps = 1e-8
    median_preds = quantile_preds[:, :, 4]

    diff = median_preds - labels
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    nd = float(np.sum(np.abs(diff)) / (np.sum(np.abs(labels)) + eps))
    smape = float(np.mean(2.0 * np.abs(diff) / (np.abs(median_preds) + np.abs(labels) + eps)))

    series_mase = compute_series_mase(
        contexts=contexts,
        labels=labels,
        median_preds=median_preds,
        seasonality=mase_seasonality,
    )
    mase = geometric_mean_nonnegative(series_mase)

    quantile_losses = []
    metric_dict = {
        "mse": mse,
        "mae": mae,
        "mase": mase,
        "rmse": rmse,
        "nd": nd,
        "smape": smape,
    }
    for q_idx, q in enumerate(QUANTILE_LEVELS):
        q_pred = quantile_preds[:, :, q_idx]
        dev = labels - q_pred
        q_loss = 2.0 * np.where(dev >= 0, dev * q, -dev * (1.0 - q))
        q_loss_mean = float(np.mean(q_loss))
        quantile_losses.append(q_loss_mean)
        metric_dict[f"quantile_loss_q{int(q * 100)}"] = q_loss_mean

    metric_dict["mean_quantile_loss"] = float(np.mean(quantile_losses))
    return metric_dict


def save_forecast_plot(
    sample: Dict[str, np.ndarray],
    quantile_pred: np.ndarray,
    save_path: str,
    context_len_to_show: int,
):
    context = sample["context"]
    label = sample["label"]

    ctx_show = context[-context_len_to_show:]
    ctx_x = np.arange(-len(ctx_show), 0)
    fut_x = np.arange(0, label.shape[0])

    pred_q10 = quantile_pred[:, 0]
    pred_q20 = quantile_pred[:, 1]
    pred_q50 = quantile_pred[:, 4]
    pred_q80 = quantile_pred[:, 7]
    pred_q90 = quantile_pred[:, 8]

    plt.figure(figsize=(14, 4))
    plt.plot(ctx_x, ctx_show, color="#4C72B0", linewidth=1.2, label="History")
    plt.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.fill_between(fut_x, pred_q10, pred_q90, alpha=0.15, color="#DD8452", label="80% CI")
    plt.fill_between(fut_x, pred_q20, pred_q80, alpha=0.25, color="#DD8452", label="60% CI")
    plt.plot(fut_x, pred_q50, color="#DD8452", linewidth=1.8, label="Forecast (Q50)")
    plt.plot(fut_x, label, color="#55A868", linewidth=1.8, label="Ground Truth")

    plt.plot([ctx_x[-1], fut_x[0]], [ctx_show[-1], pred_q50[0]], color="#DD8452", linewidth=1.8)
    plt.plot([ctx_x[-1], fut_x[0]], [ctx_show[-1], label[0]], color="#55A868", linewidth=1.8)

    plt.title(f"Seq #{sample['seq_idx']} | pred_len={label.shape[0]}")
    plt.xlabel("Time Steps (0 = forecast start)")
    plt.ylabel("Value")
    plt.legend(loc="upper left", fontsize=8, framealpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    if args.prediction_length <= 0:
        raise ValueError("--prediction-length must be positive")
    if args.prediction_length > 1024:
        raise ValueError("--prediction-length > 1024 is not supported by this evaluation script")
    if args.target_seq_idx < 0:
        raise ValueError("--target-seq-idx must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.per_core_batch_size <= 0:
        raise ValueError("--per-core-batch-size must be positive")
    if args.max_context_length <= 0:
        raise ValueError("--max-context-length must be positive")
    if args.mase_seasonality <= 0:
        raise ValueError("--mase-seasonality must be positive")
    if args.sample_mode == "train_window":
        if args.train_max_length <= 0:
            raise ValueError("--train-max-length must be positive")
        if args.train_fixed_gt_length <= 0 or args.train_fixed_gt_length >= args.train_max_length:
            raise ValueError("--train-fixed-gt-length must be in (0, --train-max-length)")
        if args.window_offset < 0:
            raise ValueError("--window-offset must be non-negative")
        if args.prediction_length > args.train_fixed_gt_length:
            raise ValueError("--prediction-length cannot exceed --train-fixed-gt-length in train_window mode")

    normalization_method = None if args.normalization_method == "none" else args.normalization_method

    dataset = TimeMoEDataset(
        args.data_path,
        normalization_method=normalization_method,
    )
    samples = build_overfit_samples(
        dataset=dataset,
        prediction_length=args.prediction_length,
        min_context_length=args.min_context_length,
        target_seq_idx=args.target_seq_idx,
        sample_mode=args.sample_mode,
        train_max_length=args.train_max_length,
        train_fixed_gt_length=args.train_fixed_gt_length,
        window_offset=args.window_offset,
    )

    labels = np.stack([item["label"] for item in samples], axis=0)

    tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    load_tfm_checkpoint_compatible(
        tfm=tfm,
        checkpoint_path=checkpoint_path,
        torch_compile=args.torch_compile,
    )

    quantile_preds = predict_quantiles(
        tfm=tfm,
        samples=samples,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        per_core_batch_size=args.per_core_batch_size,
    )
    contexts = [item["context"] for item in samples]
    metrics = compute_metrics(
        contexts=contexts,
        labels=labels,
        quantile_preds=quantile_preds,
        mase_seasonality=args.mase_seasonality,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model_name,
        "checkpoint": str(checkpoint_path),
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "num_series": len(samples),
        "target_seq_idx": args.target_seq_idx,
        "sample_mode": args.sample_mode,
        "train_max_length": args.train_max_length,
        "train_fixed_gt_length": args.train_fixed_gt_length,
        "window_offset": args.window_offset,
        "prediction_length": args.prediction_length,
        "min_context_length": args.min_context_length,
        "max_context_length": args.max_context_length,
        "mase_seasonality": args.mase_seasonality,
        "batch_size": args.batch_size,
        "per_core_batch_size": args.per_core_batch_size,
        "normalization_method": args.normalization_method,
        "metrics": metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    median_preds = quantile_preds[:, :, 4]
    eps = 1e-8
    diff = median_preds - labels
    series_mse = np.mean(diff ** 2, axis=1)
    series_mae = np.mean(np.abs(diff), axis=1)
    series_rmse = np.sqrt(series_mse)
    series_nd = np.sum(np.abs(diff), axis=1) / (np.sum(np.abs(labels), axis=1) + eps)
    series_smape = np.mean(2.0 * np.abs(diff) / (np.abs(median_preds) + np.abs(labels) + eps), axis=1)
    series_mase = compute_series_mase(
        contexts=contexts,
        labels=labels,
        median_preds=median_preds,
        seasonality=args.mase_seasonality,
    )

    with open(output_dir / "per_series_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seq_idx",
            "context_len",
            "prediction_len",
            "mse",
            "mae",
            "mase",
            "rmse",
            "nd",
            "smape",
        ])
        for idx, sample in enumerate(samples):
            writer.writerow([
                sample["seq_idx"],
                len(sample["context"]),
                len(sample["label"]),
                float(series_mse[idx]),
                float(series_mae[idx]),
                float(series_mase[idx]),
                float(series_rmse[idx]),
                float(series_nd[idx]),
                float(series_smape[idx]),
            ])

    if args.save_plots:
        plot_dir = output_dir / "plots"
        n_plots = min(args.max_plot_samples, len(samples))
        for idx in range(n_plots):
            sample = samples[idx]
            save_path = plot_dir / f"seq_{sample['seq_idx']}.png"
            save_forecast_plot(
                sample=sample,
                quantile_pred=quantile_preds[idx],
                save_path=str(save_path),
                context_len_to_show=args.plot_context_len,
            )

    print("Overfit evaluation completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
