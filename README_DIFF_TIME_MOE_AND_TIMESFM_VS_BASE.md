# `time_moe/` 与 `timesfm/src/timesfm/` 相对 base 的差异索引

便于逐文件对照检查。**行号均指当前仓库中的文件**（非 base 副本）。

**对比路径约定**

| 当前仓库 | Base（对照） |
|----------|----------------|
| `time_moe/` | `timemoe_base/Time-MoE/time_moe/` |
| `timesfm/src/timesfm/` | `timesfm_base/timesfm/src/timesfm/` |

---

## 一、`time_moe/`

### 1. 仅当前仓库有（base 无对应文件）

| 文件 | 说明 |
|------|------|
| `time_moe/models/modeling_timesfm_2p5.py` | 全文为新增（约 L1–629）：`TimesFM2p5ForTraining`、`TimesFM2p5TrainingConfig`、与 HF Trainer 对接的 forward/loss、从 Hub/本地加载 `safetensors` 等。 |

---

### 2. `time_moe/datasets/time_moe_window_dataset.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L59** | 过滤条件由 `n_points < 2` 改为 `n_points < 481`：仅保留长度 ≥481 的序列；注释仍写 “fewer than 2 points”，与代码不一致。 |
| **L84–L86** | 不足 `window_size_plus_one` 时：`np.pad` 由**尾部** `(0, n_pad)` 改为**头部** `(n_pad, 0)`；`loss_mask` 同步改为左填充。改变时间轴上 padding 与有效 token 的对齐方式。 |

---

### 3. `time_moe/datasets/time_moe_dataset.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L13–L23** | `__init__` 增加参数 `max_sequences`；校验 `max_sequences > 0`。 |
| **L45–L46** | `os.walk` 时对 `dirs`、`files` 做 `sort`，遍历顺序确定。 |
| **L66–L71** | 引入 `total_num_sequences`；`num_sequences` 在设置 `max_sequences` 时为 `min(总数, max)`。 |
| **L76–L77** | `__getitem__` 上界由 `cumsum_lengths[-1]` 改为 `num_sequences`。 |
| **L87–L88** | `get_sequence_length_by_idx` 同样改为与 `num_sequences` 比较。 |
| **L101–L106** | `get_num_tokens()`：若序列被截断，改为按前 `num_sequences` 条逐条累加长度，而非对各子数据集 `get_num_tokens()` 全量求和。 |

---

### 4. `time_moe/models/modeling_time_moe.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L414** | 注意力里 KV 长度：`past_key_value.get_seq_length(...)` → `get_usable_length(kv_seq_len, self.layer_idx)`。 |
| **L509** | 另一处注意力分支同样改为 `get_usable_length`。 |
| **L825** | `past_key_values_length`：`get_seq_length()` → `get_usable_length(seq_length)`。 |
| **L1111–L1120**（约） | `prepare_inputs_for_generation`：`DynamicCache` 时用 `seen_tokens` 作为 `past_length`；`max_cache_length` 由 `getattr(..., 'max_cache_len')` 改为 `past_key_values.get_max_length()`；并增加简短注释。 |

**性质**：主要为适配较新 `transformers` 的 KV Cache API；影响带 cache 的生成/解码路径，与标准 HF `Trainer` 里无 cache 的前向是否一致取决于版本，但差异集中在 cache 相关逻辑。

---

### 5. `time_moe/models/ts_generation_mixin.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L6** | 去掉 `GenerationConfig` 的 import。 |
| **L14–L30** | 生成入口由新版 `_sample(..., generation_config=...)` 改为旧式 **`_greedy_search`**，参数为显式 `max_length`、`pad_token_id`、`eos_token_id` 等。 |
| **L37–L74** | 在 `_greedy_search` 内自行初始化 `logits_processor` / `stopping_criteria`、`eos` 处理，并从 `self.generation_config` 取默认开关。 |
| **L99、L104、L114** 等 | 循环内注释风格与 base 不同；`synced_gpus` 分支保留 `continue`。 |
| **L141–L143** | `next_tokens` 不再 `argmax`，直接使用 `logits_processor` 输出（与连续值/TimeSeries 头一致）。 |
| **L155** | 去掉对 `input_ids` 的 `unsqueeze(-1)`，直接 `cat` 扩展序列维。 |
| **L208–L211** | `_update_model_kwargs_for_generation`：`past_key_values` 改为通过 `_extract_past_from_model_output` 写入。 |
| **L236–L238** | `cache_position` 更新为 `[-1:] + horizon_length`，并保留一行注释掉的 `+ 1` 写法。 |

**性质**：主要影响 **`.generate()` / 自回归解码** 路径，与 HF Trainer 常规训练步无直接对应关系。

---

### 6. `time_moe/trainer/hf_trainer.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L23–L24** | `__init__` 末尾增加 `_extra_loss_sums`、`_extra_loss_counts`。 |
| **L26–L71** | 新增 `_extract_metric_value`、`_accumulate_extra_metric`、重写 **`compute_loss`**（支持 `num_items_in_batch`）、重写 **`log`**：在训练步聚合 `train_loss`、`quantile_loss_sum` 并写入 logs。 |
| **L93** | `create_scheduler` 中 cosine 调度调用增加 `num_cycles=self.args.cosine_num_cycles`。 |
| **L127–L129** | `TimeMoETrainingArguments` 增加字段 **`cosine_num_cycles`**（默认 `0.5`）。 |
| **文件末尾** | 最后一行无换行符（`\ No newline at end of file`），与 base 仅有格式差异。 |

---

### 7. `time_moe/runner.py`（改动面最大）

按逻辑块与**当前文件行号**对照：

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L4–L7、L11** | 增加 `inspect`、`json`、`re`；`torch.utils.data.random_split`。 |
| **L16、L26–L33** | 引入 `TimesFM2p5ForTraining`；`TimeMoeRunner` 默认 `output_path='logs/time_moe2'`、增加 **`model_family`**（默认 `'timesfm_2p5'`）。 |
| **L35–L102** | **`load_model` 重写**：`timesfm_2p5` 分支构造/加载 `TimesFM2p5ForTraining`；原 **`TimeMoeForPrediction` + flash/eager 注意力** 整段注释掉，`time_moe` 分支除 `ValueError` 外无可执行加载逻辑。 |
| **L104–L120** | **`train_model` 开头**：按 `RANK` 设置 `model_seed = seed + rank` 并 `setup_seed`；打印分布式环境信息。 |
| **L173–L175** | 根据 Transformers 版本选择 `evaluation_strategy` 或 `eval_strategy` 参数名。 |
| **L177–L197** | **`ddp_find_unused_parameters`** 默认规则（TimesFM 为 True）；**验证集比例** `validation_split_ratio`、`enable_validation_split`；`load_best_model_at_end`、`metric_for_best_model`、`greater_is_better` 推断。 |
| **L199–L246** | **`TimeMoETrainingArguments` 构造**：增加 `logging_strategy`、`cosine_num_cycles`、`report_to`、`seed=model_seed`（与 `data_seed` 分离）、`ddp_timeout`、`load_best_model_at_end` 相关字段等。 |
| **L248–L267** | 加载模型时 **`model_family` 写死为 `"timesfm_2p5"`**（忽略上一行取到的 `model_family` 变量）；传入大量 TimesFM 训练相关 kwargs。 |
| **L276–L278** | 去掉打印完整 `training_args` 的一行，改为稍后打印。 |
| **L286–L324** | 数据改为 **`get_train_val_datasets`**；若有 `eval_ds` 则自动调整 `eval_strategy`/`save_strategy`/`eval_steps`/`save_steps`；再实例化 `TimeMoETrainingArguments` 并 `log`。 |
| **L326–L376** | **`resume_from_checkpoint`**（含 `auto`）、`trainer.train(resume_from_checkpoint=...)`；根据 `best_model_checkpoint` 另存 **`best_model/`** 与 **`best_model_info.json`**。 |
| **L383–L388** | 保留原 **`get_train_dataset`**（行为与 base 接近），但训练主路径已改用 `get_train_val_datasets`。 |
| **L390–L432** | **新增 `get_train_val_datasets`**：`TimeMoEDataset(..., max_sequences=...)` + `TimeMoEWindowDataset` + `random_split` 划分 train/val。 |

---

## 二、`timesfm/src/timesfm/`

与 base 相比，**仅有以下 3 个文件内容不同**（忽略 `__pycache__`）。

---

### 1. `timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_base.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L96–L128**（约） | 数据类 **`TimesFM_2p5_200M_Definition`** 中：`tokenizer` 与 `stacked_transformers.transformer` 的 **`hidden_dims` / `model_dims` 由 1280 改为 1920**，**`num_heads` 由 16 改为 24**；`output_projection_*` 的 **`input_dims` / `hidden_dims` 同步改为 1920**（`output_dims` 数值与 base 一致处仍为点预测/分位数输出维度）。**这是与官方 200M 公开结构不一致的硬差异。** |
| **L224–L226** | 文档字符串中 **`xreg_mode` 两种模式的文字说明**与 base 对调修正（哪一种是 residual on TimesFM forecast、哪一种是先 xreg 再 TimesFM），**仅注释/文档，不改计算**。 |

---

### 2. `timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L21、L25** | `typing` 增加 `Dict`；`PyTorchModelHubMixin` → **`ModelHubMixin`**。 |
| **L41–L114** | **`TimesFM_2p5_200M_torch_module.__init__`**：增加可选 **`num_layers` / `num_heads` / `model_dims`**；当与默认 Definition 不一致时用 `dataclasses.replace` 重写 `tokenizer`、`stacked_transformers`、`output_projection_*` 的 config；否则沿用类级 `config`。 |
| **L338–L407** | **`TimesFM_2p5_200M_torch` 包装类**：去掉 `PyTorchModelHubMixin` 的 repo 元数据装饰；`model` 改为类注解 + 默认模块实例；**`_from_pretrained`**：`force_download` 默认 **`True`**；先 **`cls(**model_kwargs)`** 再下载；固定对 **`google/timesfm-2.5-200m-pytorch` 的 `config.json`** 调用 `hf_hub_download(..., force_download=True)` 及 **`print("Downloaded.")`**；权重路径与 `load_checkpoint(..., **model_kwargs)` 行为与 base 的 Hub 封装不同。 |

**说明**：若训练走 `time_moe` 里 `TimesFM2p5ForTraining.from_pretrained` 直接读 `safetensors`，可能**不经过**本文件中的 `TimesFM_2p5_200M_torch._from_pretrained`，但 **`TimesFM_2p5_200M_torch_module` 仍依赖 `timesfm_2p5_base` 的 Definition**（见上一节 1920/24 问题）。

---

### 3. `timesfm/src/timesfm/torch/transformer.py`

| 行号（当前） | 与 base 的差异要点 |
|--------------|-------------------|
| **L107–L108**（约） | RoPE 分支：对 **`sin` / `cos` 做 `.to(inputs.dtype)`**，与输入 dtype 对齐。 |
| **L139–L158**（约） | **`_dot_product_attention`**：在调用 `F.scaled_dot_product_attention` 前，若 `mask` 存在则构造 **`safe_mask`**，对「**某行全被 mask**」的情况打开一个 dummy key，避免 fused SDPA 产生 NaN；可选地将 **q/k 转为 `value.dtype`**；调用后用 **`masked_fill` 将全 mask 行输出置 0**。 |
| **L167–L168**（约） | `scaled_dot_product_attention` 使用 **`safe_mask`** 替代原 `mask`。 |

**性质**：数值/稳定性与 dtype 行为与 base 可能有微小差别，主要规避全 mask 行与 dtype 混用问题。

---

## 三、快速核对命令（可选）

在仓库根目录自行执行可得到与本文一致的机械 diff（仅辅助，非生成本文的依据）：

```bash
diff -u timemoe_base/Time-MoE/time_moe/datasets/time_moe_window_dataset.py time_moe/datasets/time_moe_window_dataset.py
diff -u timemoe_base/Time-MoE/time_moe/runner.py time_moe/runner.py
diff -u timesfm_base/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_base.py timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_base.py
# …其余文件同理
```

---

*文档根据当前工作区与 `timemoe_base`、`timesfm_base` 快照对照整理；若你后续改行号，以 `diff -u` 为准更新本节行号即可。*
