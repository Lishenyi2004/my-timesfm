#!/usr/bin/env bash
set -e

# ===================================================
# 1. 激活环境
# ===================================================
source /root/miniconda3/bin/activate MOE

# ===================================================
# 1.1 修复 CUDA 动态库优先级（避免加载 cublasLt stub）
# ===================================================
sanitize_ld_library_path() {
    local original_ld="${LD_LIBRARY_PATH:-}"
    local sanitized_ld=""
    local IFS=':'

    for p in $original_ld; do
        [ -z "$p" ] && continue
        case "$p" in
            *"/stubs"|*"/stubs/")
                ;;
            *)
                if [ -z "$sanitized_ld" ]; then
                    sanitized_ld="$p"
                else
                    sanitized_ld="${sanitized_ld}:$p"
                fi
                ;;
        esac
    done

    local preferred_cuda_paths=""
    for p in /usr/local/cuda/lib64 /usr/local/cuda/compat /usr/lib/x86_64-linux-gnu; do
        if [ -d "$p" ]; then
            if [ -z "$preferred_cuda_paths" ]; then
                preferred_cuda_paths="$p"
            else
                preferred_cuda_paths="${preferred_cuda_paths}:$p"
            fi
        fi
    done

    if [ -n "$sanitized_ld" ] && [ -n "$preferred_cuda_paths" ]; then
        export LD_LIBRARY_PATH="${preferred_cuda_paths}:${sanitized_ld}"
    elif [ -n "$preferred_cuda_paths" ]; then
        export LD_LIBRARY_PATH="$preferred_cuda_paths"
    else
        export LD_LIBRARY_PATH="$sanitized_ld"
    fi
}

sanitize_ld_library_path

echo ">>> 平台注入信息:"
echo "NODE_RANK: ${NODE_RANK}"
echo "NODE_COUNT: ${NODE_COUNT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
# echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# 检查必要变量
if [ -z "${NODE_RANK}" ] || [ -z "${MASTER_ADDR}" ]; then
    echo "Error: 平台环境变量缺失！请确保使用了 -e DISTRIBUTED_JOB=true"
    env | grep -E "NODE|MASTER"
    exit 1
fi

# 2.1 设置端口 (平台未注入，需手动指定)
export MASTER_PORT=29500

# 2.2 设置 PyTorch/NCCL 必要的变量
if [ -n "${NCCL_SOCKET_IFNAME}" ]; then
    export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME%%,*} # 取第一个网卡
    export TP_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME%%,*}
fi

# 2.3 显式导出给 Python 脚本使用的变量
export RANK=${NODE_RANK}       # 很多旧脚本习惯用 RANK 代表节点 ID
export WORLD_SIZE=${NODE_COUNT} # 这里指节点数，具体看你的 python 脚本逻辑
export NNODES=${NODE_COUNT}

# 2.4 NCCL/PyTorch 通信稳定性参数（降低长时间 allreduce 超时风险）
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO

# ===================================================
# 3. 启动训练
# ===================================================
cd /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE
DATA_PATH="/mnt/shared-storage-gpfs2/speechllm-share/data/Time-300B/datasets--Maple728--Time-300B/snapshots/b52d0ca9de8da5202f73a5057681fca6b48906fb"

echo "---------------------------------------"
echo "启动命令: python torch_dist_run.py main.py"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Node Rank: $NODE_RANK / $NODE_COUNT"
echo "---------------------------------------"

# 使用 exec 让 Python 接管进程，确保信号传递
# 并读取环境变量 MASTER_ADDR, MASTER_PORT, NODE_RANK
exec python torch_dist_run.py main.py \
    -d "$DATA_PATH" \
    --from_scratch \
    --global_batch_size 65536 \
    --micro_batch_size 1024 \
    --deepspeed ds_config_zero1.json \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_strategy steps \
    --save_steps 2000 \
    --load_best_model_at_end \
    --dataloader_num_workers 16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine  \
    --cosine_num_cycles 0.5 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-5 \
    --weight_decay 0.05 \
    --ddp_timeout 7200 \
    --precision bf16 \
    --num_train_epochs 15 \
    --output_path logs_2/time_moe4 \
    --attn_implementation flash_attention_2 \
    --ddp_find_unused_parameters
