import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dotenv import load_dotenv
from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
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


# ──────────────────────────────────────────────
# Logging filter
# ──────────────────────────────────────────────
class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record: logging.LogRecord) -> bool:
        return self.text_to_filter not in record.getMessage()


# ──────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────
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

            max_context = (
                (max_context + self.tfm.model.p - 1) // self.tfm.model.p
            ) * self.tfm.model.p
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


# ──────────────────────────────────────────────
# ★ 新增：可视化函数
# ──────────────────────────────────────────────
def save_forecast_plot(
    test_data_input,
    test_data_label,
    forecasts: List[QuantileForecast],
    ds_config: str,
    plot_dir: str,
    max_variates: int = 4,
    context_len_to_show: int = 200,
):
    """
    为一个数据集保存一张预测对比图。

    Parameters
    ----------
    test_data_input  : 测试集输入（历史序列），iterable of dict
    test_data_label  : 测试集标签（真实未来值），iterable of dict
    forecasts        : TimesFmPredictor.predict() 返回的 QuantileForecast 列表
    ds_config        : 数据集配置字符串，用于标题和文件名，如 "electricity/H/short"
    plot_dir         : 图片保存目录
    max_variates     : 最多展示几个 variate（子图行数）
    context_len_to_show : 历史段最多展示多少个时间步（避免图太密）
    """
    os.makedirs(plot_dir, exist_ok=True)

    # 只取第一条样本做展示（代表性够用，避免图太多）
    sample_input = next(iter(test_data_input))
    sample_label = next(iter(test_data_label))
    sample_forecast: QuantileForecast = forecasts[0]

    # 兼容单变量 (1-D) 和多变量 (2-D, shape=[variates, time])
    context_target = np.array(sample_input["target"])
    label_target = np.array(sample_label["target"])

    if context_target.ndim == 1:
        context_target = context_target[np.newaxis, :]  # (1, T)
        label_target = label_target[np.newaxis, :]

    n_variates = min(context_target.shape[0], max_variates)
    pred_len = sample_forecast.prediction_length

    fig, axes = plt.subplots(
        n_variates, 1,
        figsize=(14, 3.5 * n_variates),
        squeeze=False,
    )
    fig.suptitle(f"Forecast vs Ground Truth\n{ds_config}", fontsize=13, fontweight="bold")

    for i in range(n_variates):
        ax = axes[i][0]

        # ── 历史数据（截取末尾 context_len_to_show 步）──
        ctx = context_target[i]
        ctx_show = ctx[-context_len_to_show:]
        ctx_x = np.arange(-len(ctx_show), 0)

        # ── 预测未来 ──
        fut_x = np.arange(0, pred_len)
        q50_key = "0.5"
        q10_key = "0.1"
        q90_key = "0.9"
        q20_key = "0.2"
        q80_key = "0.8"

        # QuantileForecast.forecast_array shape: (n_quantiles, pred_len)
        # forecast_keys 对应 ["0.1","0.2",...,"0.9"]
        key_to_idx = {k: idx for idx, k in enumerate(sample_forecast.forecast_keys)}

        pred_median = sample_forecast.forecast_array[key_to_idx[q50_key], :]
        pred_q10    = sample_forecast.forecast_array[key_to_idx[q10_key], :]
        pred_q90    = sample_forecast.forecast_array[key_to_idx[q90_key], :]
        pred_q20    = sample_forecast.forecast_array[key_to_idx[q20_key], :]
        pred_q80    = sample_forecast.forecast_array[key_to_idx[q80_key], :]

        # ── 真实未来 ──
        gt = label_target[i][:pred_len]  # 对齐长度

        # ── 绘图 ──
        ax.plot(ctx_x, ctx_show, color="#4C72B0", linewidth=1.2, label="History")
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        # 置信区间（80% 和 60%）
        ax.fill_between(fut_x, pred_q10, pred_q90,
                        alpha=0.15, color="#DD8452", label="80% CI (Q10–Q90)")
        ax.fill_between(fut_x, pred_q20, pred_q80,
                        alpha=0.25, color="#DD8452", label="60% CI (Q20–Q80)")

        # 预测中位数
        ax.plot(fut_x, pred_median, color="#DD8452", linewidth=1.8,
                linestyle="-", label="Forecast (Q50)")

        # 真实值
        ax.plot(fut_x, gt, color="#55A868", linewidth=1.8,
                linestyle="-", label="Ground Truth")

        # 连接历史与预测的衔接点（视觉连贯）
        ax.plot([ctx_x[-1], fut_x[0]], [ctx_show[-1], pred_median[0]],
                color="#DD8452", linewidth=1.8, linestyle="-")
        ax.plot([ctx_x[-1], fut_x[0]], [ctx_show[-1], gt[0]],
                color="#55A868", linewidth=1.8, linestyle="-")

        variate_label = f"Variate {i + 1}" if n_variates > 1 else "Series"
        ax.set_ylabel(variate_label, fontsize=10)
        ax.set_xlabel("Time Steps (0 = forecast start)", fontsize=9)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # 文件名：把 "/" 替换为 "_"，避免路径问题
    safe_name = ds_config.replace("/", "_")
    save_path = os.path.join(plot_dir, f"{safe_name}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot saved → {save_path}")


# ──────────────────────────────────────────────
# Checkpoint loader
# ──────────────────────────────────────────────
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
                import torch
                tfm.model = torch.compile(tfm.model)
            tfm.model.eval()
            print("Loaded checkpoint with auto-remap: stripped 'backbone.' prefix.")
            return

        raise first_error


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="TimesFM-2.5 evaluation script.")
    parser.add_argument("--short-datasets", type=str, default="m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H")
    parser.add_argument("--med-long-datasets", type=str, default="electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H")
    parser.add_argument("--model-name", type=str, default="TimesFM-2.5")
    parser.add_argument("--checkpoint-path", type=str, default=str(REPO_ROOT / "logs" / "time_moe3"))
    parser.add_argument("--dataset-properties", type=str, default=str(Path(__file__).resolve().parent / "notebooks" / "dataset_properties.json"))
    parser.add_argument("--output-dir", type=str, default="results/timesfm_change")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--gift-eval-root", type=str, default=None)
    # ★ 新增参数
    parser.add_argument("--save-plots", action="store_true", help="是否保存预测对比图")
    parser.add_argument("--plot-context-len", type=int, default=200, help="图中展示的历史步数")
    parser.add_argument("--plot-max-variates", type=int, default=4, help="图中最多展示几个 variate")
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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    load_dotenv()

    if args.gift_eval_root:
        os.environ["GIFT_EVAL"] = str(Path(args.gift_eval_root).expanduser().resolve())

    if not os.getenv("GIFT_EVAL"):
        raise ValueError("GIFT_EVAL is not set.")

    dataset_properties_map = json.load(open(args.dataset_properties))
    all_datasets = list(set(args.short_datasets.split() + args.med_long_datasets.split()))

    metrics = build_metrics()

    gts_logger = logging.getLogger("gluonts.model.forecast")
    gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))

    tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    load_tfm_checkpoint_compatible(tfm=tfm, checkpoint_path=checkpoint_path, torch_compile=args.torch_compile)

    output_dir = args.output_dir or str(Path(__file__).resolve().parent / "results" / f"{args.model_name}_original")
    os.makedirs(output_dir, exist_ok=True)

    # ★ 图片保存目录
    plot_dir = os.path.join(output_dir, "plots")

    csv_file_path = os.path.join(output_dir, "all_results.csv")

    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "dataset", "model",
            "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
            "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
            "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
            "eval_metrics/MSIS", "eval_metrics/RMSE[mean]",
            "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain", "num_variates",
        ])

    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        terms = ["short", "medium", "long"]
        for term in terms:
            if (term == "medium" or term == "long") and ds_name not in args.med_long_datasets.split():
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0].lower()
                ds_freq = ds_name.split("/")[1]
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

            predictor = TimesFmPredictor(tfm=tfm, prediction_length=dataset.prediction_length)

            # ★ 需要同时拿到 forecasts 列表才能画图，所以先手动 predict
            test_input_list  = list(dataset.test_data.input)
            test_label_list  = list(dataset.test_data.label)

            forecasts: List[QuantileForecast] = predictor.predict(
                test_input_list, batch_size=args.batch_size
            )

            # ── 评估 ──
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

            # ── ★ 保存图片 ──
            if args.save_plots:
                save_forecast_plot(
                    test_data_input=test_input_list,
                    test_data_label=test_label_list,
                    forecasts=forecasts,
                    ds_config=ds_config,
                    plot_dir=plot_dir,
                    max_variates=args.plot_max_variates,
                    context_len_to_show=args.plot_context_len,
                )

            with open(csv_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ds_config, args.model_name,
                    res["MSE[mean]"][0], res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0], res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0], res["sMAPE[0.5]"][0],
                    res["MSIS"][0], res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0], res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                ])

            print(f"Results for {ds_name} have been written to {csv_file_path}")


if __name__ == "__main__":
    main()
