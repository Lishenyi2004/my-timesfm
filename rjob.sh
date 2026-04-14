JOB_NAME="train-time300b-m4"
WORKDIR="/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE"
TRAIN_SCRIPT="${WORKDIR}/train.sh"

# 资源配置
GPU_PER_NODE=4
NUM_NODES=1
NAMESPACE="ailab-brainllm"
CHARGED_GROUP="brainllm_gpu"

# 计算资源
GPU=$GPU_PER_NODE
CPU=$((GPU * 14))
MEMORY=$((10 * 120000))
MASTER_PORT=$((20000 + RANDOM % 20000))

echo "Using MASTER_PORT=${MASTER_PORT}"

rjob submit \
    --gpu=$GPU \
    --name=$JOB_NAME \
    --memory=$MEMORY \
    --cpu=$CPU \
    -P $NUM_NODES \
    --namespace=$NAMESPACE \
    --charged-group=$CHARGED_GROUP \
    -e GROUP=$CHARGED_GROUP \
    -e DISTRIBUTED_JOB=true \
    -e MASTER_PORT=$MASTER_PORT \
    --private-machine=group \
    --image=registry.h.pjlab.org.cn/ailab-speechllm-speechllm_cpu/lsy:moe-fast \
    --mount=gpfs://gpfs1/lishenyi:/mnt/shared-storage-user/lishenyi \
    --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
    --mount=gpfs://gpfs2/speechllm-share:/mnt/shared-storage-gpfs2/speechllm-share \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources brainpp.cn/fuse=1 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --host-network=true \
    --preemptible=no \
    --gang-start=true \
    --auto-restart=false \
    -- bash "$TRAIN_SCRIPT"
