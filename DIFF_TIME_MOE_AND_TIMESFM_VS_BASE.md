# time_moe / timesfm 与 baseline 差异索引

对照目录：
- `time_moe/` ↔ `timemoe_base/Time-MoE/time_moe/`
- `timesfm/src/timesfm/` ↔ `timesfm_base/timesfm/src/timesfm/`

说明：`@@` hunk 行为 **连续行区间**（unified diff 标准）。仅列 Python 文件（`.py`）；`__pycache__` 已忽略。

---

## 1. `time_moe/`

| 文件 | 说明 | base 行号区间 | 当前仓库行号区间 |
|------|------|---------------|------------------|
| `datasets/time_moe_dataset.py` | 共 **5** 处 hunk | base: L10-21, L39-44, L57-70, L77-84, L88-94 | 当前: L10-25, L43-50, L63-80, L87-94, L98-107 |
| `datasets/time_moe_window_dataset.py` | 共 **2** 处 hunk | base: L56-62, L81-88 | 当前: L56-62, L81-88 |
| `models/modeling_time_moe.py` | 共 **4** 处 hunk | base: L411-417, L506-512, L822-828, L1108-1118 | 当前: L411-417, L506-512, L822-828, L1108-1123 |
| `models/modeling_timesfm_2p5.py` | 仅当前仓库存在（base 无） | 当前约 **628** 行 | — |
| `models/ts_generation_mixin.py` | 共 **7** 处 hunk | base: L3-9, L11-56, L59-68, L72-83, L97-114, L162-182, L186-190 | 当前: L3-9, L11-93, L96-107, L111-124, L138-157, L205-231, L235-240 |
| `runner.py` | 共 **8** 处 hunk | base: L1-14, L18-67, L114-134, L143-167, L174-180, L183-200, L207-212, L247-250 | 当前: L1-19, L23-123, L170-217, L226-268, L275-280, L283-380, L387-436, L471-474 |
| `trainer/hf_trainer.py` | 共 **4** 处 hunk | base: L20-25, L30-35, L63-68, L91-94 | 当前: L20-85, L90-96, L124-132, L155-158 |

### 1.1 `time_moe/` 各文件 hunk 摘要（diff 上下文）

#### `datasets/time_moe_dataset.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timemoe_base/Time-MoE/time_moe/datasets/time_moe_dataset.py	2026-04-11 12:12:39.699310000 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/time_moe/datasets/time_moe_dataset.py	2026-03-20 13:59:44.654626000 +0800
@@ -10,12 +10,16 @@
 
 class TimeMoEDataset(TimeSeriesDataset):
 
-    def __init__(self, data_folder, normalization_method=None):
+    def __init__(self, data_folder, normalization_method=None, max_sequences=None):
         self.data_folder = data_folder
         self.normalization_method = normalization_method
+        self.max_sequences = None if max_sequences is None else int(max_sequences)
         self.datasets = []
         self.num_tokens = None
 
+        if self.max_sequences is not None and self.max_sequences <= 0:
+            raise ValueError(f'max_sequences should be positive, but got {self.max_sequences}')
+
         if normalization_method is None:
             self.normalization_method = None
         elif isinstance(normalization_method, str):
@@ -39,6 +43,8 @@
         else:
             # walk through the data_folder
             for root, dirs, files in os.walk(self.data_folder):
+                dirs.sort()
+                files.sort()
                 for file in files:
                     fn_path = os.path.join(root, file)
                     if file != BinaryDataset.meta_file_name and GeneralDataset.is_valid_path(fn_path):
@@ -57,14 +63,18 @@
             self.cumsum_lengths.append(
                 self.cumsum_lengths[-1] + len(ds)
             )
-        self.num_sequences = self.cumsum_lengths[-1]
+        self.total_num_sequences = self.cumsum_lengths[-1]
+        if self.max_sequences is None:
+            self.num_sequences = self.total_num_sequences
+        else:
+            self.num_sequences = min(self.total_num_sequences, self.max_sequences)
 
     def __len__(self):
         return self.num_sequences
 
     def __getitem__(self, seq_idx):
-        if seq_idx >= self.cumsum_lengths[-1]:
-            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
+        if seq_idx >= self.num_sequences:
+            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.num_sequences}')
         elif seq_idx < 0:
             raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')
 
@@ -77,8 +87,8 @@
         return seq
 
     def get_sequence_length_by_idx(self, seq_idx):
-        if seq_idx >= self.cumsum_lengths[-1]:
-            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
+        if seq_idx >= self.num_sequences:
+            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.num_sequences}')
         elif seq_idx < 0:
             raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')
 
@@ -88,7 +98,10 @@
 
     def get_num_tokens(self):
         if self.num_tokens is None:
-            self.num_tokens = sum([ds.get_num_tokens() for ds in self.datasets])
+            if self.num_sequences == self.total_num_sequences:
+                self.num_tokens = sum([ds.get_num_tokens() for ds in self.datasets])
+            else:
+                self.num_tokens = sum([self.get_sequence_length_by_idx(i) for i in range(self.num_sequences)])
 
         return self.num_tokens
```

#### `datasets/time_moe_window_dataset.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timemoe_base/Time-MoE/time_moe/datasets/time_moe_window_dataset.py	2026-04-11 12:12:39.699961000 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/time_moe/datasets/time_moe_window_dataset.py	2026-04-09 18:56:20.511663000 +0800
@@ -56,7 +56,7 @@
         for seq_idx in iterator:
             n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
             # Skip sequences with fewer than 2 points
-            if n_points < 2:
+            if n_points < 481:
                 continue
             self.sub_seq_indexes.append((seq_idx, 0))
             for offset_idx in range(
@@ -81,8 +81,8 @@
         loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
         n_pad = self.window_size_plus_one - len(seq)
         if n_pad > 0:
-            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
-            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)
+            seq = np.pad(seq, (n_pad, 0), 'constant', constant_values=0)
+            loss_mask = np.pad(loss_mask, (n_pad, 0), 'constant', constant_values=0)
 
         return {
             'input_ids': seq[:-1],
```

#### `models/modeling_time_moe.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timemoe_base/Time-MoE/time_moe/models/modeling_time_moe.py	2026-04-11 12:12:39.703317523 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/time_moe/models/modeling_time_moe.py	2026-02-12 17:51:25.873963533 +0800
@@ -411,7 +411,7 @@
                     "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                     "with a layer index."
                 )
-            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
+            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
 
@@ -506,7 +506,7 @@
                     "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                     "with a layer index."
                 )
-            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
+            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
         rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
         cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
@@ -822,7 +822,7 @@
             use_legacy_cache = not isinstance(past_key_values, Cache)
             if use_legacy_cache:
                 past_key_values = DynamicCache.from_legacy_cache(past_key_values)
-            past_key_values_length = past_key_values.get_seq_length()
+            past_key_values_length = past_key_values.get_usable_length(seq_length)
 
         if position_ids is None:
             device = input_ids.device if input_ids is not None else inputs_embeds.device
@@ -1108,11 +1108,16 @@
     def prepare_inputs_for_generation(
             self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
     ):
+        # Omit tokens covered by past_key_values
         if past_key_values is not None:
             if isinstance(past_key_values, Cache):
                 cache_length = past_key_values.get_seq_length()
-                past_length = cache_length
-                max_cache_length = getattr(past_key_values, 'max_cache_len', None)
+                if isinstance(past_key_values, DynamicCache):
+                    past_length = past_key_values.seen_tokens
+                else:
+                    past_length = cache_length
+
+                max_cache_length = past_key_values.get_max_length()
             else:
                 cache_length = past_length = past_key_values[0][0].shape[2]
                 max_cache_length = None
```

#### `models/ts_generation_mixin.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timemoe_base/Time-MoE/time_moe/models/ts_generation_mixin.py	2026-04-11 12:12:39.704095000 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/time_moe/models/ts_generation_mixin.py	2026-02-12 17:51:25.878142000 +0800
@@ -3,7 +3,7 @@
 
 import torch
 
-from transformers import GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
+from transformers import GenerationMixin, LogitsProcessorList, StoppingCriteriaList
 from transformers.generation import validate_stopping_criteria, EosTokenCriteria
 from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
 from transformers.utils import ModelOutput
@@ -11,46 +11,83 @@
 
 class TSGenerationMixin(GenerationMixin):
 
-    def _sample(
+    def _greedy_search(
             self,
-            input_ids: torch.LongTensor,
-            logits_processor: LogitsProcessorList,
-            stopping_criteria: StoppingCriteriaList,
-            generation_config: GenerationConfig,
+            input_ids: torch.Tensor,
+            logits_processor: Optional[LogitsProcessorList] = None,
+            stopping_criteria: Optional[StoppingCriteriaList] = None,
+            max_length: Optional[int] = None,
+            pad_token_id: Optional[int] = None,
+            eos_token_id: Optional[Union[int, List[int]]] = None,
+            output_attentions: Optional[bool] = None,
+            output_hidden_states: Optional[bool] = None,
+            output_scores: Optional[bool] = None,
+            output_logits: Optional[bool] = None,
+            return_dict_in_generate: Optional[bool] = None,
             synced_gpus: bool = False,
             streamer: Optional["BaseStreamer"] = None,
             **model_kwargs,
-    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
+    ) -> Union[GenerateNonBeamOutput, torch.Tensor]:
         input_ids_origin_device = input_ids.device
         input_ids = input_ids.to(self.device)
         if len(input_ids.shape) == 2:
             batch_size, cur_len = input_ids.shape
         else:
             raise ValueError('Input shape must be: [batch_size, seq_len]')
-        
-        pad_token_id = generation_config._pad_token_tensor
-        output_attentions = generation_config.output_attentions
-        output_hidden_states = generation_config.output_hidden_states
-        output_scores = generation_config.output_scores
-        output_logits = generation_config.output_logits
-        return_dict_in_generate = generation_config.return_dict_in_generate
-        
-        eos_token_id = generation_config._eos_token_tensor
+        # init values
+        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
+        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
+        if max_length is not None:
+            warnings.warn(
+                "`max_length` is deprecated in this function, use"
+                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
+                UserWarning,
+            )
+            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
+        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
+        if eos_token_id is not None:
+            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
+        else:
+            # remove when the method is totally private
+            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
+            eos_token_id = [
+                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
+            ]
+            eos_token_id = eos_token_id[0] if eos_token_id else None
+            if eos_token_id is None and self.generation_config.eos_token_id is not None:
+                eos_token_id = self.generation_config.eos_token_id
+                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
+
         if isinstance(eos_token_id, int):
             eos_token_id = [eos_token_id]
+        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
+        output_attentions = (
+            output_attentions if output_attentions is not None else self.generation_config.output_attentions
+        )
+        output_hidden_states = (
+            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
+        )
+        return_dict_in_generate = (
+            return_dict_in_generate
+            if return_dict_in_generate is not None
+            else self.generation_config.return_dict_in_generate
+        )
 
+        # init attention / hidden states / scores tuples
         raw_logits = () if (return_dict_in_generate and output_logits) else None
         scores = () if (return_dict_in_generate and output_scores) else None
         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
 
+        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
         if return_dict_in_generate and self.config.is_encoder_decoder:
             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
             encoder_hidden_states = (
                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
             )
 
+        # keep track of which sequences are already finished
         if "inputs_embeds" in model_kwargs:
             cur_len = model_kwargs["inputs_embeds"].shape[1]
         this_peer_finished = False
@@ -59,10 +96,12 @@
 
         max_length = stopping_criteria.max_length
         while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
+            # prepare model inputs
             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
 
             input_length = input_ids.shape[1]
 
+            # forward pass to get next token
             outputs = self(
                 **model_inputs,
                 return_dict=True,
@@ -72,12 +111,14 @@
             )
 
             if synced_gpus and this_peer_finished:
-                continue
+                continue  # don't waste resources running the code we don't need
 
             next_token_logits = outputs.logits[:, -1, :]
 
+            # pre-process distribution
             next_tokens_scores = logits_processor(input_ids, next_token_logits)
 
+            # Store scores, attentions and hidden_states when required
             if return_dict_in_generate:
                 if output_scores:
                     scores += (next_tokens_scores,)
@@ -97,18 +138,20 @@
                         else (outputs.hidden_states,)
                     )
 
+            # argmax
+            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
             next_tokens = next_tokens_scores
 
+            # finished sentences should have their next token be a padding token
             if eos_token_id is not None:
                 if pad_token_id is None:
                     raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                 next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
 
+            # update generated ids, model inputs, and length for next step
             next_tokens = next_tokens.reshape(batch_size, -1, self.config.input_size)
             horizon_length = next_tokens.shape[1]
 
-            if input_ids.ndim == 2:
-                input_ids = input_ids.unsqueeze(-1)
             input_ids = torch.cat([input_ids, next_tokens], dim=-2)
             if streamer is not None:
                 streamer.put(next_tokens.cpu())
@@ -162,21 +205,27 @@
             is_encoder_decoder: bool = False,
             standardize_cache_format: bool = False,
     ) -> Dict[str, Any]:
-        model_kwargs["past_key_values"] = outputs.past_key_values
+        # update past_key_values
+        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
+            outputs, standardize_cache_format=standardize_cache_format
+        )
         if getattr(outputs, "state", None) is not None:
             model_kwargs["state"] = outputs.state
 
+        # update token_type_ids with last value
         if "token_type_ids" in model_kwargs:
             token_type_ids = model_kwargs["token_type_ids"]
             model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
 
         if not is_encoder_decoder:
+            # update attention mask
             if "attention_mask" in model_kwargs:
                 attention_mask = model_kwargs["attention_mask"]
                 model_kwargs["attention_mask"] = torch.cat(
                     [attention_mask, attention_mask.new_ones((attention_mask.shape[0], horizon_length))], dim=-1
                 )
         else:
+            # update decoder attention mask
             if "decoder_attention_mask" in model_kwargs:
                 decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                 model_kwargs["decoder_attention_mask"] = torch.cat(
@@ -186,5 +235,6 @@
 
         if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
             model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + horizon_length
+            # model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
 
         return model_kwargs
```

#### `runner.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timemoe_base/Time-MoE/time_moe/runner.py	2026-04-11 12:12:39.704696000 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/time_moe/runner.py	2026-04-10 21:16:01.858710000 +0800
@@ -1,14 +1,19 @@
 import os
 import math
 import random
+import inspect
+import json
+import re
 from functools import reduce
 from operator import mul
 
 import torch
+from torch.utils.data import random_split
 
 from time_moe.datasets.time_moe_dataset import TimeMoEDataset
 from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
 from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
+from time_moe.models.modeling_timesfm_2p5 import TimesFM2p5ForTraining
 from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
 from time_moe.utils.dist_util import get_world_size
 from time_moe.utils.log_util import logger, log_in_local_rank_0
@@ -18,50 +23,101 @@
     def __init__(
             self,
             model_path: str = None,
-            output_path: str = 'logs/time_moe',
-            seed: int = 9899
+            output_path: str = 'logs/time_moe2',
+            seed: int = 9899,
+            model_family: str = 'timesfm_2p5',
     ):
         self.model_path = model_path
         self.output_path = output_path
         self.seed = seed
+        self.model_family = model_family
 
     def load_model(self, model_path: str = None, from_scatch: bool = False, **kwargs):
         if model_path is None:
             model_path = self.model_path
-        attn = kwargs.pop('attn_implementation', None)
-        if attn is None:
-            attn = 'eager'
-        elif attn == 'auto':
-            # try to use flash-attention
-            try:
-                from flash_attn import flash_attn_func, flash_attn_varlen_func
-                from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
-                attn = 'flash_attention_2'
-            except:
-                log_in_local_rank_0('Flash attention import failed, switching to eager attention.', type='warn')
-                attn = 'eager'
-
-        if attn == 'eager':
-            log_in_local_rank_0('Use Eager Attention')
-        elif attn == 'flash_attention_2':
-            log_in_local_rank_0('Use Flash Attention 2')
-        else:
-            raise ValueError(f'Unknown attention method: {attn}')
-        kwargs['attn_implementation'] = attn
 
-        if from_scatch:
-            config = TimeMoeConfig.from_pretrained(model_path, _attn_implementation=attn)
-            model = TimeMoeForPrediction(config)
-        else:
-            model = TimeMoeForPrediction.from_pretrained(model_path, **kwargs)
-        return model
+        model_family = kwargs.pop('model_family', None) or self.model_family
+
+        if model_family == 'timesfm_2p5':
+            if from_scatch:
+                model = TimesFM2p5ForTraining(
+                    torch_dtype=kwargs.get('torch_dtype'),
+                    use_quantile_loss=kwargs.get('use_quantile_loss', True),
+                    quantile_loss_weight=kwargs.get('quantile_loss_weight', 1.0),
+                    timesfm_num_layers=kwargs.get('timesfm_num_layers'),
+                    timesfm_num_heads=kwargs.get('timesfm_num_heads'),
+                    timesfm_model_dims=kwargs.get('timesfm_model_dims'),
+                    use_revin_norm=kwargs.get('use_revin_norm', True),
+                    use_revin_denorm=kwargs.get('use_revin_denorm', True),
+                    enable_overfit_fixed_window=kwargs.get('enable_overfit_fixed_window', False),
+                    overfit_hist_length=kwargs.get('overfit_hist_length', 384),
+                    overfit_gt_length=kwargs.get('overfit_gt_length', 128),
+                )
+            else:
+                model = TimesFM2p5ForTraining.from_pretrained(
+                    model_path=model_path,
+                    torch_dtype=kwargs.get('torch_dtype'),
+                    use_quantile_loss=kwargs.get('use_quantile_loss', True),
+                    quantile_loss_weight=kwargs.get('quantile_loss_weight', 1.0),
+                    timesfm_num_layers=kwargs.get('timesfm_num_layers'),
+                    timesfm_num_heads=kwargs.get('timesfm_num_heads'),
+                    timesfm_model_dims=kwargs.get('timesfm_model_dims'),
+                    use_revin_norm=kwargs.get('use_revin_norm', True),
+                    use_revin_denorm=kwargs.get('use_revin_denorm', True),
+                    enable_overfit_fixed_window=kwargs.get('enable_overfit_fixed_window', False),
+                    overfit_hist_length=kwargs.get('overfit_hist_length', 384),
+                    overfit_gt_length=kwargs.get('overfit_gt_length', 128),
+                )
+            return model
+
+        if model_family != 'time_moe':
+            raise ValueError(f'Unknown model_family: {model_family}')
+
+        # attn = kwargs.pop('attn_implementation', None)
+        # if attn is None:
+        #     attn = 'eager'
+        # elif attn == 'auto':
+        #     # try to use flash-attention
+        #     try:
+        #         from flash_attn import flash_attn_func, flash_attn_varlen_func
+        #         from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
+        #         attn = 'flash_attention_2'
+        #     except:
+        #         log_in_local_rank_0('Flash attention import failed, switching to eager attention.', type='warn')
+        #         attn = 'eager'
+
+        # if attn == 'eager':
+        #     log_in_local_rank_0('Use Eager Attention')
+        # elif attn == 'flash_attention_2':
+        #     log_in_local_rank_0('Use Flash Attention 2')
+        # else:
+        #     raise ValueError(f'Unknown attention method: {attn}')
+        # kwargs['attn_implementation'] = attn
+
+        # if from_scatch:
+        #     config = TimeMoeConfig.from_pretrained(model_path, _attn_implementation=attn)
+        #     model = TimeMoeForPrediction(config)
+        # else:
+        #     model = TimeMoeForPrediction.from_pretrained(model_path, **kwargs)
+        # return model
 
     def train_model(self, from_scratch: bool = False, **kwargs):
-        setup_seed(self.seed)
+        rank_for_seed = int(os.getenv('RANK', '0'))
+        model_seed = int(self.seed) + rank_for_seed
+        setup_seed(model_seed)
 
         train_config = kwargs
 
         num_devices = get_world_size()
+        log_in_local_rank_0(
+            'Distributed env:',
+            f'RANK={os.getenv("RANK")}',
+            f'LOCAL_RANK={os.getenv("LOCAL_RANK")}',
+            f'WORLD_SIZE={os.getenv("WORLD_SIZE")}',
+            f'LOCAL_WORLD_SIZE={os.getenv("LOCAL_WORLD_SIZE")}',
+            f'Detected num_devices={num_devices}',
+            f'Model seed per rank={model_seed} (base_seed={self.seed}, rank_offset={rank_for_seed})',
+        )
 
         global_batch_size = train_config.get('global_batch_size', None)
         micro_batch_size = train_config.get('micro_batch_size', None)
@@ -114,21 +170,48 @@
         log_in_local_rank_0(f'Set precision to {precision}')
         log_in_local_rank_0(f'Set normalization to {train_config["normalization_method"]}')
 
-        training_args = TimeMoETrainingArguments(
+        eval_strategy_arg_name = 'evaluation_strategy'
+        if 'eval_strategy' in inspect.signature(TimeMoETrainingArguments.__init__).parameters:
+            eval_strategy_arg_name = 'eval_strategy'
+
+        ddp_find_unused_parameters = train_config.get('ddp_find_unused_parameters')
+        if ddp_find_unused_parameters is None:
+            ddp_find_unused_parameters = (train_config.get('model_family') or self.model_family) == 'timesfm_2p5'
+        log_in_local_rank_0(f'Set ddp_find_unused_parameters to {bool(ddp_find_unused_parameters)}')
+
+        # Validation split and effective strategies.
+        validation_split_ratio = float(train_config.get('validation_split_ratio', 0.01))
+        enable_validation_split = bool(train_config.get('enable_validation_split', True))
+
+        eval_strategy = train_config.get("evaluation_strategy", 'no')
+        save_strategy = train_config.get("save_strategy", "no")
+        eval_steps = _safe_float(train_config.get("eval_steps", None))
+        save_steps = _safe_float(train_config.get("save_steps", None))
+        load_best_model_at_end = bool(train_config.get('load_best_model_at_end', True))
+        metric_for_best_model = train_config.get('metric_for_best_model', 'eval_loss')
+        greater_is_better = train_config.get('greater_is_better')
+        if greater_is_better is None and metric_for_best_model.endswith('loss'):
+            greater_is_better = False
+
+        if enable_validation_split and validation_split_ratio <= 0:
+            enable_validation_split = False
+
+        training_args_kwargs = dict(
             output_dir=self.output_path,
             num_train_epochs=num_train_epochs,
             # use_cpu=True,
             max_steps=train_steps,
-            evaluation_strategy=train_config.get("evaluation_strategy", 'no'),
-            eval_steps=_safe_float(train_config.get("eval_steps", None)),
-            save_strategy=train_config.get("save_strategy", "no"),
-            save_steps=_safe_float(train_config.get("save_steps", None)),
+            logging_strategy=train_config.get('logging_strategy', 'steps'),
+            eval_steps=eval_steps,
+            save_strategy=save_strategy,
+            save_steps=save_steps,
             learning_rate=float(train_config.get("learning_rate", 1e-5)),
             min_learning_rate=float(train_config.get("min_learning_rate", 0)),
             adam_beta1=float(train_config.get("adam_beta1", 0.9)),
             adam_beta2=float(train_config.get("adam_beta2", 0.95)),
             adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
             lr_scheduler_type=train_config.get("lr_scheduler_type", 'constant'),
+            cosine_num_cycles=float(train_config.get("cosine_num_cycles", 0.5)),
             warmup_ratio=float(train_config.get("warmup_ratio") or 0.0),
             warmup_steps=int(train_config.get("warmup_steps", 0)),
             weight_decay=float(train_config.get("weight_decay", 0.1)),
@@ -143,25 +226,43 @@
             logging_first_step=True,
             log_on_each_node=False,
             logging_steps=int(train_config.get('logging_steps', 1)),
-            seed=self.seed,
+            report_to=train_config.get('report_to', ['tensorboard']),
+            seed=model_seed,
             data_seed=self.seed,
             max_grad_norm=train_config.get('max_grad_norm', 1.0),
             optim=train_config.get('optim', 'adamw_torch'),
             torch_compile=train_config.get('torch_compile', False),
             dataloader_num_workers=train_config.get('dataloader_num_workers') or 2,
-            ddp_find_unused_parameters=False,
+            ddp_timeout=int(train_config.get('ddp_timeout', 1800)),
+            ddp_find_unused_parameters=bool(ddp_find_unused_parameters),
 
             logging_dir=os.path.join(self.output_path, 'tb_logs'),
             save_only_model=train_config.get('save_only_model', True),
             save_total_limit=train_config.get('save_total_limit'),
+            load_best_model_at_end=load_best_model_at_end,
+            metric_for_best_model=metric_for_best_model,
+            greater_is_better=greater_is_better,
         )
+        training_args_kwargs[eval_strategy_arg_name] = eval_strategy
 
         model_path = train_config.pop('model_path', None) or self.model_path
+        model_family = train_config.get('model_family') or self.model_family
         if model_path is not None:
             model = self.load_model(
                 model_path=model_path,
                 from_scatch=from_scratch,
                 torch_dtype=torch_dtype,
+                model_family="timesf
... [truncated, 请本地运行: diff -u timemoe_base/Time-MoE/time_moe/runner.py time_moe/runner.py]
```

#### `trainer/hf_trainer.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timemoe_base/Time-MoE/time_moe/trainer/hf_trainer.py	2026-04-11 12:12:39.710149000 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/time_moe/trainer/hf_trainer.py	2026-04-02 10:28:48.915627000 +0800
@@ -20,6 +20,66 @@
         self.tokenizer = kwargs.get("tokenizer", None)
         self.label_column = label_column
         self.loss_mask_column = loss_mask_column
+        self._extra_loss_sums = {}
+        self._extra_loss_counts = {}
+
+    @staticmethod
+    def _extract_metric_value(outputs, key: str):
+        value = None
+        if isinstance(outputs, dict):
+            value = outputs.get(key)
+        else:
+            value = getattr(outputs, key, None)
+
+        if value is None:
+            return None
+        if isinstance(value, torch.Tensor):
+            if value.numel() == 0:
+                return None
+            return value.detach().float().mean().item()
+        try:
+            return float(value)
+        except (TypeError, ValueError):
+            return None
+
+    def _accumulate_extra_metric(self, name: str, value):
+        if value is None:
+            return
+        self._extra_loss_sums[name] = self._extra_loss_sums.get(name, 0.0) + float(value)
+        self._extra_loss_counts[name] = self._extra_loss_counts.get(name, 0) + 1
+
+    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
+        if num_items_in_batch is not None:
+            outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
+        else:
+            outputs = model(**inputs)
+
+        if isinstance(outputs, dict):
+            loss = outputs["loss"]
+        else:
+            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
+
+        if model.training:
+            self._accumulate_extra_metric('train_loss', self._extract_metric_value(outputs, 'train_loss'))
+            self._accumulate_extra_metric('quantile_loss_sum', self._extract_metric_value(outputs, 'quantile_loss_sum'))
+
+        if return_outputs:
+            return loss, outputs
+        return loss
+
+    def log(self, logs, start_time=None):
+        if self._extra_loss_sums:
+            logs = dict(logs)
+            for metric_name, metric_sum in self._extra_loss_sums.items():
+                metric_count = self._extra_loss_counts.get(metric_name, 0)
+                if metric_count > 0:
+                    logs[metric_name] = metric_sum / metric_count
+            self._extra_loss_sums.clear()
+            self._extra_loss_counts.clear()
+
+        if start_time is None:
+            return super().log(logs)
+        return super().log(logs, start_time=start_time)
 
     def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
         optimizer = self.optimizer if optimizer is None else optimizer
@@ -30,6 +90,7 @@
                     optimizer=optimizer,
                     num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                     num_training_steps=num_training_steps,
+                    num_cycles=self.args.cosine_num_cycles,
                     min_lr_ratio=min_lr_ratio,
                 )
             else:
@@ -63,6 +124,9 @@
     min_learning_rate: float = field(
         default=0, metadata={"help": "Minimum learning rate for cosine_schedule"}
     )
+    cosine_num_cycles: float = field(
+        default=0.5, metadata={"help": "Number of cosine cycles when using cosine scheduler"}
+    )
 
 
 def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
@@ -91,4 +155,4 @@
         num_cycles=num_cycles,
         min_lr_ratio=min_lr_ratio,
     )
-    return LambdaLR(optimizer, lr_lambda, last_epoch)
+    return LambdaLR(optimizer, lr_lambda, last_epoch)
\ No newline at end of file
```

---

## 2. `timesfm/src/timesfm/`

| 文件 | 说明 | base 行号区间 | 当前仓库行号区间 |
|------|------|---------------|------------------|
| `timesfm_2p5/timesfm_2p5_base.py` | 共 **3** 处 hunk | base: L95-111, L116-130, L221-228 | 当前: L95-111, L116-130, L221-228 |
| `timesfm_2p5/timesfm_2p5_torch.py` | 共 **3** 处 hunk | base: L18-28, L38-46, L263-352 | 当前: L18-28, L38-118, L335-409 |
| `torch/transformer.py` | 共 **3** 处 hunk | base: L105-110, L135-140, L143-149 | 当前: L105-112, L137-162, L165-174 |

### 2.1 `timesfm/src/timesfm/` 各文件 hunk 摘要

#### `timesfm_2p5/timesfm_2p5_base.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm_base/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_base.py	2026-04-11 12:18:26.883864000 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_base.py	2026-03-30 19:11:08.251258000 +0800
@@ -95,17 +95,17 @@
   decode_index: int = 5
   tokenizer: ResidualBlockConfig = ResidualBlockConfig(
     input_dims=64,
-    hidden_dims=1280,
-    output_dims=1280,
+    hidden_dims=1920,
+    output_dims=1920,
     use_bias=True,
     activation="swish",
   )
   stacked_transformers: StackedTransformersConfig = StackedTransformersConfig(
     num_layers=20,
     transformer=TransformerConfig(
-      model_dims=1280,
-      hidden_dims=1280,
-      num_heads=16,
+      model_dims=1920,
+      hidden_dims=1920,
+      num_heads=24,
       attention_norm="rms",
       feedforward_norm="rms",
       qk_norm="rms",
@@ -116,15 +116,15 @@
     ),
   )
   output_projection_point: ResidualBlockConfig = ResidualBlockConfig(
-    input_dims=1280,
-    hidden_dims=1280,
+    input_dims=1920,
+    hidden_dims=1920,
     output_dims=1280,
     use_bias=False,
     activation="swish",
   )
   output_projection_quantiles: ResidualBlockConfig = ResidualBlockConfig(
-    input_dims=1280,
-    hidden_dims=1280,
+    input_dims=1920,
+    hidden_dims=1920,
     output_dims=10240,
     use_bias=False,
     activation="swish",
@@ -221,8 +221,8 @@
       dynamic_categorical_covariates: A dict of dynamic categorical covariates.
       static_numerical_covariates: A dict of static numerical covariates.
       static_categorical_covariates: A dict of static categorical covariates.
-      xreg_mode: one of "xreg + timesfm" or "timesfm + xreg". "timesfm + xreg"
-        fits a model on the residuals of the TimesFM forecast. "xreg + timesfm"
+      xreg_mode: one of "xreg + timesfm" or "timesfm + xreg". "xreg + timesfm"
+        fits a model on the residuals of the TimesFM forecast. "timesfm + xreg"
         fits a model on the targets then forecasts on the residuals via TimesFM.
       normalize_xreg_target_per_input: whether to normalize the xreg target per
         input in the given batch.
```

#### `timesfm_2p5/timesfm_2p5_torch.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm_base/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py	2026-04-11 12:18:26.885879510 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py	2026-03-03 16:22:31.946357000 +0800
@@ -18,11 +18,11 @@
 import math
 import os
 from pathlib import Path
-from typing import Optional, Sequence, Union
+from typing import Dict, Optional, Sequence, Union
 
 import numpy as np
 import torch
-from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
+from huggingface_hub import ModelHubMixin, hf_hub_download
 from safetensors.torch import load_file, save_file
 from torch import nn
 
@@ -38,9 +38,81 @@
 
   config = timesfm_2p5_base.TimesFM_2p5_200M_Definition()
 
-  def __init__(self):
+  def __init__(
+    self,
+    num_layers: Optional[int] = None,
+    num_heads: Optional[int] = None,
+    model_dims: Optional[int] = None,
+  ):
     super().__init__()
 
+    base_config = self.config
+    resolved_num_layers = (
+      num_layers
+      if num_layers is not None
+      else base_config.stacked_transformers.num_layers
+    )
+    resolved_num_heads = (
+      num_heads
+      if num_heads is not None
+      else base_config.stacked_transformers.transformer.num_heads
+    )
+    resolved_model_dims = (
+      model_dims
+      if model_dims is not None
+      else base_config.stacked_transformers.transformer.model_dims
+    )
+    if resolved_model_dims % resolved_num_heads != 0:
+      raise ValueError(
+        "model_dims must be divisible by num_heads: "
+        f"{resolved_model_dims} % {resolved_num_heads} != 0"
+      )
+
+    if (
+      resolved_num_layers != base_config.stacked_transformers.num_layers
+      or resolved_num_heads
+      != base_config.stacked_transformers.transformer.num_heads
+      or resolved_model_dims
+      != base_config.stacked_transformers.transformer.model_dims
+    ):
+      updated_transformer = dataclasses.replace(
+        base_config.stacked_transformers.transformer,
+        model_dims=resolved_model_dims,
+        hidden_dims=resolved_model_dims,
+        num_heads=resolved_num_heads,
+      )
+      updated_stacked = dataclasses.replace(
+        base_config.stacked_transformers,
+        num_layers=resolved_num_layers,
+        transformer=updated_transformer,
+      )
+      updated_tokenizer = dataclasses.replace(
+        base_config.tokenizer,
+        hidden_dims=resolved_model_dims,
+        output_dims=resolved_model_dims,
+      )
+      updated_output_projection_point = dataclasses.replace(
+        base_config.output_projection_point,
+        input_dims=resolved_model_dims,
+        hidden_dims=resolved_model_dims,
+        output_dims=base_config.output_patch_len * (len(base_config.quantiles) + 1),
+      )
+      updated_output_projection_quantiles = dataclasses.replace(
+        base_config.output_projection_quantiles,
+        input_dims=resolved_model_dims,
+        hidden_dims=resolved_model_dims,
+        output_dims=base_config.output_quantile_len * (len(base_config.quantiles) + 1),
+      )
+      self.config = dataclasses.replace(
+        base_config,
+        tokenizer=updated_tokenizer,
+        stacked_transformers=updated_stacked,
+        output_projection_point=updated_output_projection_point,
+        output_projection_quantiles=updated_output_projection_quantiles,
+      )
+    else:
+      self.config = base_config
+
     # Names constants.
     self.p = self.config.input_patch_len  # 32
     self.o = self.config.output_patch_len  # 128
@@ -263,90 +335,75 @@
     return outputs
 
 
-class TimesFM_2p5_200M_torch(
-  timesfm_2p5_base.TimesFM_2p5,
-  PyTorchModelHubMixin,
-  library_name="timesfm",
-  repo_url="https://github.com/google-research/timesfm",
-  paper_url="https://arxiv.org/abs/2310.10688",
-  docs_url="https://github.com/google-research/timesfm",
-  license="apache-2.0",
-  pipeline_tag="time-series-forecasting",
-  tags=["pytorch", "timeseries", "forecasting", "timesfm-2.5"],
-):
+class TimesFM_2p5_200M_torch(timesfm_2p5_base.TimesFM_2p5, ModelHubMixin):
   """PyTorch implementation of TimesFM 2.5 with 200M parameters."""
 
-  DEFAULT_REPO_ID = "google/timesfm-2.5-200m-pytorch"
-  WEIGHTS_FILENAME = "model.safetensors"
-
-  def __init__(
-    self,
-    torch_compile: bool = True,
-    config: Optional[dict] = None,
-  ):
-    self.model = TimesFM_2p5_200M_torch_module()
-    self.torch_compile = torch_compile
-    if config is not None:
-      self._hub_mixin_config = config
+  model: nn.Module = TimesFM_2p5_200M_torch_module()
 
   @classmethod
   def _from_pretrained(
     cls,
     *,
-    model_id: str = DEFAULT_REPO_ID,
+    model_id: str = "google/timesfm-2.5-200m-pytorch",
     revision: Optional[str],
     cache_dir: Optional[Union[str, Path]],
-    force_download: bool = False,
+    force_download: bool = True,
+    proxies: Optional[Dict] = None,
+    resume_download: Optional[bool] = None,
     local_files_only: bool,
     token: Optional[Union[str, bool]],
-    config: Optional[dict] = None,
     **model_kwargs,
   ):
     """
     Loads a PyTorch safetensors TimesFM model from a local path or the Hugging
     Face Hub. This method is the backend for the `from_pretrained` class
-    method provided by `PyTorchModelHubMixin`.
+    method provided by `ModelHubMixin`.
     """
+    # Create an instance of the model wrapper class.
+    instance = cls(**model_kwargs)
+    # Download the config file for hf tracking.
+    _ = hf_hub_download(
+      repo_id="google/timesfm-2.5-200m-pytorch",
+      filename="config.json",
+      force_download=True,
+    )
+    print("Downloaded.")
+
     # Determine the path to the model weights.
     model_file_path = ""
     if os.path.isdir(model_id):
       logging.info("Loading checkpoint from local directory: %s", model_id)
-      model_file_path = os.path.join(model_id, cls.WEIGHTS_FILENAME)
+      model_file_path = os.path.join(model_id, "model.safetensors")
       if not os.path.exists(model_file_path):
-        raise FileNotFoundError(
-          f"{cls.WEIGHTS_FILENAME} not found in directory {model_id}"
-        )
+        raise FileNotFoundError(f"model.safetensors not found in directory {model_id}")
     else:
       logging.info("Downloading checkpoint from Hugging Face repo %s", model_id)
       model_file_path = hf_hub_download(
         repo_id=model_id,
-        filename=cls.WEIGHTS_FILENAME,
+        filename="model.safetensors",
         revision=revision,
         cache_dir=cache_dir,
         force_download=force_download,
+        proxies=proxies,
+        resume_download=resume_download,
         token=token,
         local_files_only=local_files_only,
       )
 
-    # Create an instance of the model wrapper class.
-    instance = cls(config=config, **model_kwargs)
-
     logging.info("Loading checkpoint from: %s", model_file_path)
     # Load the weights into the model.
-    instance.model.load_checkpoint(
-      model_file_path, torch_compile=instance.torch_compile
-    )
+    instance.model.load_checkpoint(model_file_path, **model_kwargs)
     return instance
 
   def _save_pretrained(self, save_directory: Union[str, Path]):
     """
     Saves the model's state dictionary to a safetensors file. This method
-    is called by the `save_pretrained` method from `PyTorchModelHubMixin`.
+    is called by the `save_pretrained` method from `ModelHubMixin`.
     """
     if not os.path.exists(save_directory):
       os.makedirs(save_directory)
 
-    weights_path = os.path.join(save_directory, self.WEIGHTS_FILENAME)
+    weights_path = os.path.join(save_directory, "model.safetensors")
     save_file(self.model.state_dict(), weights_path)
 
   def compile(self, forecast_config: configs.ForecastConfig, **kwargs) -> None:
```

#### `torch/transformer.py`

```diff
--- /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm_base/timesfm/src/timesfm/torch/transformer.py	2026-04-11 12:18:26.891889000 +0800
+++ /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm/src/timesfm/torch/transformer.py	2026-03-18 14:58:40.851780000 +0800
@@ -105,6 +105,8 @@
     sinusoid_inp = position / timescale
     sin = torch.sin(sinusoid_inp)
     cos = torch.cos(sinusoid_inp)
+    sin = sin.to(inputs.dtype)
+    cos = cos.to(inputs.dtype)
     first_half, second_half = torch.chunk(inputs, 2, dim=-1)
     first_part = first_half * cos - second_half * sin
     second_part = second_half * cos + first_half * sin
@@ -135,6 +137,26 @@
   but using the fast and fused F.scaled_dot_product_attention kernel.
   """
 
+#   return _dot_product_attention(query, key, value, mask)
+
+  safe_mask = mask
+  fully_masked_rows = None
+  if mask is not None:
+    # F.scaled_dot_product_attention can emit NaNs, and more importantly NaN
+    # gradients, when any query row is fully masked. Keep the fused kernel by
+    # opening a single dummy key for those rows, then zero their outputs.
+    fully_masked_rows = ~mask.any(dim=-1, keepdim=True)
+    if fully_masked_rows.any():
+      dummy_key_mask = torch.zeros_like(mask)
+      dummy_key_mask[..., 0] = True
+      safe_mask = mask | (fully_masked_rows & dummy_key_mask)
+
+  attention_dtype = value.dtype
+  if query.dtype != attention_dtype:
+    query = query.to(attention_dtype)
+  if key.dtype != attention_dtype:
+    key = key.to(attention_dtype)
+
   # 1. Permute inputs from (B, L, H, D) to the expected (B, H, L, D)
   query = query.permute(0, 2, 1, 3)
   key = key.permute(0, 2, 1, 3)
@@ -143,7 +165,10 @@
   # 2. Call the fused attention kernel
   #    - Pass the mask to `attn_mask`.
   #    - Set `scale=1.0` to disable the default 1/sqrt(d_k) scaling.
-  output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=1.0)
+  output = F.scaled_dot_product_attention(query, key, value, attn_mask=safe_mask, scale=1.0)
+
+  if fully_masked_rows is not None and fully_masked_rows.any():
+    output = output.masked_fill(fully_masked_rows, 0.0)
 
   # 3. Permute the output back to the original (B, L, H, D) layout
   output = output.permute(0, 2, 1, 3)
```
