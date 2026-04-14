# TimesFM 小模型蒸馏（独立入口）

本仓库新增了独立蒸馏入口，不修改原有 `main.py/train.sh` 训练链路。

## 新增文件

- `distill_main.py`：蒸馏训练入口
- `time_moe/trainer/distill_trainer.py`：蒸馏 Trainer（监督 + 蒸馏损失）
- `train_distill.sh`：分布式启动脚本模板

## 设计说明

1. `teacher` 使用已有权重：
   - 默认路径：`/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm-2.5-200m-pytorch`
2. `student` 通过裁剪 transformer 层数变小：
   - `--student_num_layers 8` 表示从 20 层裁剪为 8 层
3. 蒸馏总损失：
   - `L = a * L_supervised + b * L_point_distill + c * L_quantile_distill`
4. 为减少 teacher/student 随机切窗不一致，默认启用：
   - `--enable_overfit_fixed_window`

## 启动方式

直接使用：

```bash
bash train_distill.sh
```

或手动：

```bash
python torch_dist_run.py distill_main.py \
  -d <DATA_PATH> \
  --teacher_model_path /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm-2.5-200m-pytorch \
  --student_num_layers 8 \
  --init_student_from_teacher \
  --distill_supervised_weight 1.0 \
  --distill_point_weight 1.0 \
  --distill_quantile_weight 0.5 \
  --output_path logs_2/time300b_distill_example
```

## 关键可调参数

- `--student_num_layers`：student 大小（越小越快）
- `--distill_supervised_weight`：原始监督损失权重
- `--distill_point_weight`：点预测蒸馏损失权重
- `--distill_quantile_weight`：分位数蒸馏损失权重
- `--init_student_from_teacher/--no-init_student_from_teacher`：是否用 teacher 同形状参数 warm-start
