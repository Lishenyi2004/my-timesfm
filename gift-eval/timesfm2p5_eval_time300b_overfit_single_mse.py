import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESFM_SRC = PROJECT_ROOT / "timesfm" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if TIMESFM_SRC.exists() and str(TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(TIMESFM_SRC))

from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from timesfm import configs
from timesfm.timesfm_2p5 import timesfm_2p5_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-sample overfit evaluation on Time-300B with MSE only."
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to Time-300B data")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint directory or model.safetensors",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/time300b_overfit_single_mse.json",
        help="Where to save evaluation json summary",
    )
    parser.add_argument(
        "--normalization-method",
        type=str,
        choices=["none", "zero", "max"],
        default="zero",
        help="Must match training normalization",
    )
    parser.add_argument(
        "--target-seq-idx",
        type=int,
        default=0,
        help="Target source sequence index (default: first sequence)",
    )
    parser.add_argument(
        "--window-offset",
        type=int,
        default=0,
        help="Window start offset in the source sequence (default: first window)",
    )
    parser.add_argument(
        "--train-max-length",
        type=int,
        default=512,
        help="Training max_length used in TimeMoEWindowDataset",
    )
    parser.add_argument(
        "--train-fixed-gt-length",
        type=int,
        default=128,
        help="GT tail length for overfit fixed window",
    )
    parser.add_argument(
        "--per-core-batch-size",
        type=int,
        default=1,
        help="per_core_batch_size for TimesFM forecast compile",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=4096,
        help="Max context length used in compile config",
    )
    parser.add_argument("--torch-compile", action="store_true")
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
                import torch

                tfm.model = torch.compile(tfm.model)
            tfm.model.eval()
            print("Loaded checkpoint with auto-remap: stripped 'backbone.' prefix.")
            return

        raise first_error


def build_train_aligned_sample(
    dataset: TimeMoEDataset,
    target_seq_idx: int,
    window_offset: int,
    train_max_length: int,
    train_fixed_gt_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if train_max_length <= 0:
        raise ValueError(f"train_max_length must be positive, got {train_max_length}")
    if train_fixed_gt_length <= 0 or train_fixed_gt_length >= train_max_length:
        raise ValueError(
            "train_fixed_gt_length must be in (0, train_max_length), "
            f"got {train_fixed_gt_length} and {train_max_length}"
        )
    if target_seq_idx < 0 or target_seq_idx >= len(dataset):
        raise ValueError(f"target_seq_idx out of range: {target_seq_idx}")
    if window_offset < 0:
        raise ValueError(f"window_offset must be >= 0, got {window_offset}")

    seq = np.asarray(dataset[target_seq_idx], dtype=np.float32)
    window_size_plus_one = train_max_length + 1
    end_idx = window_offset + window_size_plus_one
    if end_idx > seq.shape[0]:
        raise ValueError(
            f"Sequence too short for selected window: len={seq.shape[0]}, "
            f"need >= {end_idx} (window_offset + train_max_length + 1)"
        )

    # Match training window in TimeMoEWindowDataset:
    # seq_slice has length (max_length + 1), model input uses seq_slice[:-1].
    seq_slice = seq[window_offset:end_idx]
    input_window = seq_slice[:-1]  # len = train_max_length
    print(input_window)
    hist_len = train_max_length - train_fixed_gt_length
    context = input_window[:hist_len]
    label = input_window[hist_len:]
    print(f"context: {context.shape}, label: {label.shape}")
    return context, label


def forecast_q50(
    tfm,
    context: np.ndarray,
    horizon: int,
    max_context_length: int,
    per_core_batch_size: int,
) -> np.ndarray:
    if max_context_length is not None and context.shape[0] > max_context_length:
        context = context[-max_context_length:]

    max_context = ((context.shape[0] + tfm.model.p - 1) // tfm.model.p) * tfm.model.p
    compile_max_context = min(max_context, max_context_length)
    compile_max_context = max(tfm.model.p, compile_max_context)
    compile_max_context = (compile_max_context // tfm.model.p) * tfm.model.p
    if compile_max_context <= 0:
        compile_max_context = tfm.model.p

    # Keep this inference path close to existing eval scripts.
    tfm.compile(
        forecast_config=configs.ForecastConfig(
            max_context=compile_max_context,
            max_horizon=horizon,
            infer_is_positive=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            force_flip_invariance=True,
            return_backcast=False,
            normalize_inputs=True,
            per_core_batch_size=per_core_batch_size,
        ),
    )

    _, full_preds = tfm.forecast(horizon=horizon, inputs=[context])
    if full_preds.shape[-1] < 6:
        raise RuntimeError(f"Unexpected forecast output shape: {full_preds.shape}")

    # Existing scripts use index [1:10] for q10..q90, so q50 is index 4 in that slice.
    quantile_preds = full_preds[:, 0:horizon, 1:10].astype(np.float32)
    q50 = quantile_preds[0, :, 4]
    print("q50:", q50)
    return q50


def main():
    args = parse_args()
    normalization_method = None if args.normalization_method == "none" else args.normalization_method

    dataset = TimeMoEDataset(
        args.data_path,
        normalization_method=normalization_method,
        max_sequences=max(args.target_seq_idx + 1, 1),
    )

    context, label = build_train_aligned_sample(
        dataset=dataset,
        target_seq_idx=args.target_seq_idx,
        window_offset=args.window_offset,
        train_max_length=args.train_max_length,
        train_fixed_gt_length=args.train_fixed_gt_length,
    )

    tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    load_tfm_checkpoint_compatible(
        tfm=tfm,
        checkpoint_path=checkpoint_path,
        torch_compile=args.torch_compile,
    )

    q50_pred = forecast_q50(
        tfm=tfm,
        context=context,
        horizon=label.shape[0],
        max_context_length=args.max_context_length,
        per_core_batch_size=args.per_core_batch_size,
    )
    mse = float(np.mean((q50_pred - label) ** 2))

    output = {
        "metric": "mse",
        "mse": mse,
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "target_seq_idx": args.target_seq_idx,
        "window_offset": args.window_offset,
        "normalization_method": args.normalization_method,
        "train_max_length": args.train_max_length,
        "train_fixed_gt_length": args.train_fixed_gt_length,
        "context_length": int(context.shape[0]),
        "prediction_length": int(label.shape[0]),
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
