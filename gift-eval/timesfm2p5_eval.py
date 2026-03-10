import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import batcher
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMESFM_SRC = REPO_ROOT / "timesfm" / "src"
GIFT_EVAL_SRC = Path(__file__).resolve().parent / "src"
if TIMESFM_SRC.exists() and str(TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(TIMESFM_SRC))
if GIFT_EVAL_SRC.exists() and str(GIFT_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(GIFT_EVAL_SRC))

from gift_eval.data import Dataset
from timesfm import configs
from timesfm.timesfm_2p5 import timesfm_2p5_torch


DEFAULT_MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H bitbrains_fast_storage/5T "
    "bitbrains_rnd/5T bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record: logging.LogRecord) -> bool:
        return self.text_to_filter not in record.getMessage()


class TimesFmPredictor:
    def __init__(self, tfm, prediction_length: int):
        self.tfm = tfm
        self.prediction_length = prediction_length
        self.quantiles = list(np.arange(1, 10) / 10.0)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        test_data_input = list(test_data_input)
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = []
            max_context = 0
            for entry in batch:
                arr = np.array(entry["target"])
                if max_context < arr.shape[0]:
                    max_context = arr.shape[0]
                context.append(arr)

            max_context = ((max_context + self.tfm.model.p - 1) // self.tfm.model.p) * self.tfm.model.p
            self.tfm.compile(
                forecast_config=configs.ForecastConfig(
                    max_context=min(15360, max_context),
                    max_horizon=1024,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=128,
                ),
            )
            _, full_preds = self.tfm.forecast(
                horizon=self.prediction_length,
                inputs=context,
            )
            full_preds = full_preds[:, 0 : self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))

        forecast_outputs = np.concatenate(forecast_outputs)

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.quantiles)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts


def build_metrics():
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TimesFM-2.5 on GIFT-Eval.")
    parser.add_argument("--short-datasets", type=str, default="m4_weekly")
    parser.add_argument("--med-long-datasets", type=str, default="bizitobs_l2c/H")
    parser.add_argument("--model-name", type=str, default="TimesFM-2.5")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(REPO_ROOT / "logs" / "time_moe2"),
    )
    parser.add_argument(
        "--dataset-properties",
        type=str,
        default=str(Path(__file__).resolve().parent / "notebooks" / "dataset_properties.json"),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Auto-discover all datasets under GIFT_EVAL and evaluate them.",
    )
    parser.add_argument(
        "--gift-eval-root",
        type=str,
        default=None,
        help="Path to GIFT-Eval dataset root directory (same as GIFT_EVAL env var).",
    )
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


def is_hf_dataset_leaf(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "state.json").exists()
        and (path / "dataset_info.json").exists()
    )


def discover_datasets(gift_eval_root: Path) -> tuple[list[str], list[str]]:
    all_datasets = []

    for entry in sorted(gift_eval_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        if is_hf_dataset_leaf(entry):
            all_datasets.append(entry.name)
            continue

        has_child_dataset = False
        for child in sorted(entry.iterdir()):
            if is_hf_dataset_leaf(child):
                ds_name = f"{entry.name}/{child.name}"
                all_datasets.append(ds_name)
                has_child_dataset = True

        if not has_child_dataset and is_hf_dataset_leaf(entry):
            all_datasets.append(entry.name)

    discovered = sorted(set(all_datasets))
    canonical_to_discovered = {d.lower(): d for d in discovered}

    med_long_datasets = []
    for ds in DEFAULT_MED_LONG_DATASETS.split():
        key = ds.lower()
        if key in canonical_to_discovered:
            med_long_datasets.append(canonical_to_discovered[key])

    return discovered, sorted(set(med_long_datasets))


def main():
    args = parse_args()
    load_dotenv()

    if args.gift_eval_root:
        os.environ["GIFT_EVAL"] = str(Path(args.gift_eval_root).expanduser().resolve())

    if not os.getenv("GIFT_EVAL"):
        raise ValueError(
            "GIFT_EVAL is not set. Use --gift-eval-root or set it in .env / environment."
        )

    gift_eval_root = Path(os.getenv("GIFT_EVAL")).expanduser().resolve()
    if not gift_eval_root.exists():
        raise FileNotFoundError(f"GIFT_EVAL path does not exist: {gift_eval_root}")

    dataset_properties_map = json.load(open(args.dataset_properties))

    if args.all_datasets:
        all_datasets, med_long_dataset_list = discover_datasets(gift_eval_root)
    else:
        med_long_dataset_list = args.med_long_datasets.split()
        all_datasets = list(set(args.short_datasets.split() + med_long_dataset_list))

    if not all_datasets:
        raise ValueError("No datasets found to evaluate.")

    metrics = build_metrics()

    gts_logger = logging.getLogger("gluonts.model.forecast")
    gts_logger.addFilter(
        WarningFilter("The mean prediction is not stored in the forecast data")
    )

    tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    load_tfm_checkpoint_compatible(
        tfm=tfm,
        checkpoint_path=checkpoint_path,
        torch_compile=args.torch_compile,
    )

    output_dir = args.output_dir or str(Path(__file__).resolve().parent / "results" / args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, "all_results.csv")

    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    metric_key_map = {
        "eval_metrics/MSE[mean]": "MSE[mean]",
        "eval_metrics/MSE[0.5]": "MSE[0.5]",
        "eval_metrics/MAE[0.5]": "MAE[0.5]",
        "eval_metrics/MASE[0.5]": "MASE[0.5]",
        "eval_metrics/MAPE[0.5]": "MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]": "sMAPE[0.5]",
        "eval_metrics/MSIS": "MSIS",
        "eval_metrics/RMSE[mean]": "RMSE[mean]",
        "eval_metrics/NRMSE[mean]": "NRMSE[mean]",
        "eval_metrics/ND[0.5]": "ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss": "mean_weighted_sum_quantile_loss",
    }
    metric_means = {k: 0.0 for k in metric_key_map.keys()}
    evaluated_configs = 0

    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
            ]
        )

    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        terms = ["short", "medium", "long"]
        for term in terms:
            if (term == "medium" or term == "long") and ds_name not in med_long_dataset_list:
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            to_univariate = (
                False
                if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                else True
            )
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)
            print(f"Dataset size: {len(dataset.test_data)}")

            predictor = TimesFmPredictor(
                tfm=tfm,
                prediction_length=dataset.prediction_length,
            )

            try:
                res = evaluate_model(
                    predictor,
                    test_data=dataset.test_data,
                    metrics=metrics,
                    batch_size=args.batch_size,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season_length,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Evaluation failed on dataset={ds_name}, term={term}, "
                    f"prediction_length={dataset.prediction_length}, freq={dataset.freq}"
                ) from e

            with open(csv_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        ds_config,
                        args.model_name,
                        res["MSE[mean]"][0],
                        res["MSE[0.5]"][0],
                        res["MAE[0.5]"][0],
                        res["MASE[0.5]"][0],
                        res["MAPE[0.5]"][0],
                        res["sMAPE[0.5]"][0],
                        res["MSIS"][0],
                        res["RMSE[mean]"][0],
                        res["NRMSE[mean]"][0],
                        res["ND[0.5]"][0],
                        res["mean_weighted_sum_quantile_loss"][0],
                        dataset_properties_map[ds_key]["domain"],
                        dataset_properties_map[ds_key]["num_variates"],
                    ]
                )

            for metric_col, res_key in metric_key_map.items():
                metric_means[metric_col] += float(res[res_key][0])
            evaluated_configs += 1

            print(f"Results for {ds_name} have been written to {csv_file_path}")

    if evaluated_configs > 0:
        metric_means = {k: v / evaluated_configs for k, v in metric_means.items()}
        summary = {
            "model": args.model_name,
            "num_configs": evaluated_configs,
            "means": metric_means,
        }
        mean_file_path = os.path.join(output_dir, "mean_metrics.json")
        with open(mean_file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Mean metrics written to {mean_file_path}")


if __name__ == "__main__":
    main()
