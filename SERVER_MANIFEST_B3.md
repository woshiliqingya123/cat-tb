# GPU 服务器执行清单 —— Stage B3（摘要器生成 M）

> 目标：在有空闲大显存的服务器上起 Qwen3.6-27B 的 vLLM 端点，对 B1/B2 产出的每个折叠点
> 生成结构化长期记忆 M（目标压缩比 ~30%）。这是整条流水线里第一处需要 GPU 的步骤。

## 0. 硬件 / 模型前置
- **显存**：Qwen3.6-27B (bf16) 权重约 **54GB** + KV cache。建议：
  - 1×80GB (A100/H100)：`--tensor-parallel-size 1`
  - 2×48GB：`--tensor-parallel-size 2`
  - 4×24GB(4090)：`--tensor-parallel-size 4`
- **模型权重**：Qwen3.6-27B 需在该服务器上（确认路径，下文记为 `$MODEL`）。
  本机路径是 `/data/liqingyang/models/Qwen3.6-27B`；若那台没有，需 scp 过去（~54GB）。

## 1. 环境（与评测 env 分开；可叫 `catgpu`）
```bash
conda create -y -n catgpu python=3.12
conda activate catgpu
pip install vllm            # 自带 torch；用于起端点
pip install openai transformers   # B3 客户端 + token 计数
```

## 2. 从本机要传过去的文件
| 文件 | 大小 | 用途 |
|---|---|---|
| `project/data/fold_plans/fold_plans.jsonl` | 小 (~数十 MB) | B1/B2 折叠计划（折叠点+切段元数据） |
| `project/data/raw/tbase_pool_min50.jsonl` | **5.3 GB** | 轨迹原文（B3 据此取可压缩段的原始消息） |
| `project/catgen/stage_b12_fold_points.py` | 16K | 提供 `load_steps`（B3 import 它） |
| `project/catgen/stage_b3_generate_memory.py` | 12K | B3 主脚本 |

```bash
# 在本机执行（示例，按你那台地址改）
scp project/data/fold_plans/fold_plans.jsonl  GPU:/path/cat/data/fold_plans/
scp project/data/raw/tbase_pool_min50.jsonl   GPU:/path/cat/data/raw/
scp project/catgen/stage_b12_fold_points.py   project/catgen/stage_b3_generate_memory.py  GPU:/path/cat/catgen/
```

## 3. 起 vLLM 端点（开一个终端，常驻）
```bash
conda activate catgpu
vllm serve $MODEL \
  --served-model-name Qwen3.6-27B \
  --tensor-parallel-size 2 \         # 按你的卡数改
  --max-model-len 16384 \            # 摘要段最长 ~7k，16k 足够
  --port 8000
# 验证：curl http://localhost:8000/v1/models
```

## 4. 跑 B3
```bash
conda activate catgpu
cd /path/cat
python catgen/stage_b3_generate_memory.py \
  --fold-plans data/fold_plans/fold_plans.jsonl \
  --pool       data/raw/tbase_pool_min50.jsonl \
  --out        data/fold_plans/fold_plans_with_M.jsonl \
  --tokenizer  $MODEL \
  --base-url   http://localhost:8000/v1 \
  --model      Qwen3.6-27B \
  --target-ratio 0.30
# 先小样验证： 加 --limit 20 跑通、看 avg compression ratio 接近 0.30 再全量
# 可断点续跑：脚本会跳过 out 文件里已完成的 trajectory_id
```

**验收**：`avg compression ratio` 接近 **0.30**（论文 Table 1：15585→4676 ≈ 30%）；
随机抽几条看 M 的四段结构（阶段目标/已尝试策略及结果/关键环境反馈/仍生效约束）是否完整、
是否保留了具体路径/报错/符号名。

## 5. 跑完传回本机
```bash
scp GPU:/path/cat/data/fold_plans/fold_plans_with_M.jsonl  project/data/fold_plans/
```
拿回来后我在本机做 **B4 缝合**（插入 `context` 工具调用 + Observation=M，重建 (Q,M,近期k)）→ 产出 CaT-Instruct。

---

## 后续还需 GPU 的步骤（之后另给清单，不在本次）
- **Stage D 训练**：CaT-Instruct / Base-Instruct 各 SFT 一个 Qwen3.6-27B。需 trl/deepspeed，全参或 LoRA。
- **Stage E/F 推理评测**：vLLM 部署训练好的模型 + Harbor(OpenHands) 跑 TB。

## 待我在本机解决的依赖（不需 GPU）
- OpenHands 0.54.0 的 condensation-request 工具/action 精确 schema（B4 用，决定 `context` 工具长相）。
