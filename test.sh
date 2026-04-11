#!/usr/bin/env bash
# 运行 gift-eval/timesfm2p5_eval_change.py
# 在下方「手动配置」里改参数；留空的项使用 Python 脚本里的 argparse 默认值。
set -e

# ========== 手动配置（改这里）==========
# 字符串：留空 "" 表示不传该参数（用 Python 默认）
EVAL_SHORT_DATASETS="m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
EVAL_MED_LONG_DATASETS="electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
EVAL_CHECKPOINT_PATH="/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/logs_2/time300b_less_m4_elc_flict/best_model"           # 例: "${REPO_ROOT}/logs/time_moe3"
EVAL_OUTPUT_DIR="results_2/timesfm_change_m4_elc_flict"                # 例: "results/timesfm_change"
EVAL_BATCH_SIZE="8192"                # 例: 4096
EVAL_GIFT_EVAL_ROOT="/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/datasets/gift-eval"            # 数据根目录，一般不用设
EVAL_PLOT_CONTEXT_LEN="200"          # 例: 200
EVAL_PLOT_MAX_VARIATES="4"         # 例: 4
EVAL_SAVE_PLOTS=1
# ======================================

source /root/miniconda3/bin/activate moe
cd /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/gift-eval

_eval_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ARGS=()
[ -n "$EVAL_SHORT_DATASETS" ]       && ARGS+=(--short-datasets "$EVAL_SHORT_DATASETS")
[ -n "$EVAL_MED_LONG_DATASETS" ]   && ARGS+=(--med-long-datasets "$EVAL_MED_LONG_DATASETS")
[ -n "$EVAL_MODEL_NAME" ]           && ARGS+=(--model-name "$EVAL_MODEL_NAME")
[ -n "$EVAL_CHECKPOINT_PATH" ]     && ARGS+=(--checkpoint-path "$EVAL_CHECKPOINT_PATH")
[ -n "$EVAL_DATASET_PROPERTIES" ]  && ARGS+=(--dataset-properties "$EVAL_DATASET_PROPERTIES")
[ -n "$EVAL_OUTPUT_DIR" ]          && ARGS+=(--output-dir "$EVAL_OUTPUT_DIR")
[ -n "$EVAL_BATCH_SIZE" ]          && ARGS+=(--batch-size "$EVAL_BATCH_SIZE")
[ -n "$EVAL_GIFT_EVAL_ROOT" ]      && ARGS+=(--gift-eval-root "$EVAL_GIFT_EVAL_ROOT")
[ -n "$EVAL_PLOT_CONTEXT_LEN" ]    && ARGS+=(--plot-context-len "$EVAL_PLOT_CONTEXT_LEN")
[ -n "$EVAL_PLOT_MAX_VARIATES" ]   && ARGS+=(--plot-max-variates "$EVAL_PLOT_MAX_VARIATES")
_eval_truthy "$EVAL_TORCH_COMPILE" && ARGS+=(--torch-compile)
_eval_truthy "$EVAL_SAVE_PLOTS"    && ARGS+=(--save-plots)

python timesfm2p5_eval_change.py "${ARGS[@]}"
