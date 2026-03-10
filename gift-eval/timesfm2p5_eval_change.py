import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
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
from safetensors.torch import load_file
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
    parser = argparse.ArgumentParser(description="Original notebook-style TimesFM-2.5 evaluation script.")
    parser.add_argument("--short-datasets", type=str, default="m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H")
    parser.add_argument("--med-long-datasets", type=str, default="electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H")
    parser.add_argument("--model-name", type=str, default="TimesFM-2.5")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(REPO_ROOT / "logs" / "time_moe3"),
    )
    parser.add_argument(
        "--dataset-properties",
        type=str,
        default=str(Path(__file__).resolve().parent / "notebooks" / "dataset_properties.json"),
    )
    parser.add_argument("--output-dir", type=str, default="results/timesfm_change")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--gift-eval-root", type=str, default=None)
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


def main():
    args = parse_args()
    load_dotenv()

    if args.gift_eval_root:
        os.environ["GIFT_EVAL"] = str(Path(args.gift_eval_root).expanduser().resolve())

    if not os.getenv("GIFT_EVAL"):
        raise ValueError("GIFT_EVAL is not set. Use --gift-eval-root or set it in .env / environment.")

    dataset_properties_map = json.load(open(args.dataset_properties))
    all_datasets = list(set(args.short_datasets.split() + args.med_long_datasets.split()))

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

    output_dir = args.output_dir or str(Path(__file__).resolve().parent / "results" / f"{args.model_name}_original")
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, "all_results.csv")

    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

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
            if (term == "medium" or term == "long") and ds_name not in args.med_long_datasets.split():
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

            print(f"Results for {ds_name} have been written to {csv_file_path}")


if __name__ == "__main__":
    main()
