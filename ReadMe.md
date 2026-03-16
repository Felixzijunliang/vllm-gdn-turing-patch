# vLLM + Qwen3.5 适配 Turing 架构（SM75）补丁

在 Tesla T10 等 Turing 架构（SM75）显卡上部署 Qwen3.5 系列模型时，vLLM 会因底层算子不兼容而无法启动。本补丁通过算子降级 + 环境配置，实现在老架构上的正常运行。

## 问题背景

Qwen3.5 引入了 **Gated Delta Networks (GDN)** 混合架构。vLLM 部署时依赖 [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention) 库中的 Triton 加速算子和 FlashInfer 后端，这些组件**硬性要求 SM80（Ampere）及以上**的 GPU 计算能力。

在 SM75 上直接部署会遇到以下问题：

| 问题 | 表现 | 根因 |
|------|------|------|
| **Triton 算子崩溃** | FLA 内核编译失败或运行时报错 | Triton 编译产物使用了 SM80+ 独有指令（如 `cp.async`、`ldmatrix`），SM75 不支持 |
| **P2P 通信死锁** | 多卡启动时 NCCL 握手卡死 | 双卡 T10 无 NVLink，PCIe 直连的 P2P 通信触发 NCCL 死锁 |
| **显存 Profiling 假死** | `shm_broadcast` 持续超时警告（60s） | FlashInfer 后端在 SM75 上执行模拟推理时陷入底层死循环 |
| **CUDA Graph 捕获失败** | 替换为 Python 代码后 Graph 编译死循环 | 纯 PyTorch 实现含动态控制流（Python 循环/分支），CUDA Graph 无法捕获 |

## 解决思路

采用 **"算子降级 + 环境隔离 + 强制回退"** 策略：在数学上完全等价的前提下，用纯 PyTorch 矩阵运算替换 Triton 加速算子，并关闭不兼容的系统级特性。

具体措施：

1. **算子降级（核心）**：将 `fused_gdn_gating`、`fused_recurrent_gated_delta_rule`、`chunk_gated_delta_rule` 三个关键函数替换为纯 PyTorch 实现。数学运算完全等价，仅损失 Triton 的并行优化性能。
2. **注意力后端降级**：`VLLM_ATTENTION_BACKEND=XFORMERS`，用兼容性更好的 Xformers 替代 FlashInfer。
3. **禁用 P2P 通信**：`NCCL_P2P_DISABLE=1`，强制多卡数据交换通过主机内存中转。
4. **禁用 CUDA Graph**：`--enforce-eager`，以 Eager 模式逐行执行，规避动态控制流问题。

## 多卡并行策略：为什么选 Pipeline Parallel

Qwen3.5-9B 使用了 GQA（Grouped Query Attention）架构，模型有 16 个 Query 头但仅 8 个 KV 头。这一设计对多卡并行方案的选择有直接影响。

**Tensor Parallel（张量并行）** 会将每一层的权重矩阵横向切分到多张卡上。在 TP=2 时，每张卡分到 8 个 Query 头和 4 个 KV 头——虽然比例仍为 2:1，但 vLLM 对 Qwen3.5 的 GQA 实现在此配置下会出现 `einsum` 维度不匹配的问题，且张量并行要求频繁的卡间 all-reduce 通信，在无 NVLink 的双卡 T10 上开销很大。

**Pipeline Parallel（流水线并行）** 则按层纵向切分模型：卡 0 运行前半部分层，卡 1 运行后半部分层，每一层保持完整的 16 个 Query 头 + 8 个 KV 头，完全绕开了 GQA 头数分配问题。卡间只需在层边界传递一次激活值，通信量远低于张量并行，天然适合 PCIe 直连的老硬件。代价是存在流水线"气泡"（部分卡在等待上游数据），延迟略高于张量并行，但对于显存受限、缺少 NVLink 的 Turing 双卡环境，这是更稳妥的选择。

| 方案 | 层内切分 | 通信量 | GQA 兼容性 | 双卡 T10 适用 |
|------|----------|--------|------------|---------------|
| Tensor Parallel (TP=2) | 切分每层的头数 | 高（每层 all-reduce） | 可能触发维度不匹配 | 不推荐 |
| Pipeline Parallel (PP=2) | 每层完整保留 | 低（仅层边界传激活） | 完全兼容 | 推荐 |

## 文件说明

```
turing_fix/
├── apply_gdn_patch.py   # 补丁脚本：修改 vLLM 源码，注入 PyTorch fallback 逻辑
├── run_vllm.sh           # 启动脚本：配置环境变量 + vLLM 启动参数
└── ReadMe.md
```

`apply_gdn_patch.py` 会修改 vLLM 虚拟环境中的以下文件（自动备份 `.bak`）：

- `vllm/model_executor/layers/fla/ops/fused_recurrent.py` — 注入 recurrent 推理的 fallback 分支
- `vllm/model_executor/models/qwen3_next.py` — 注入 prefill 阶段的 fallback 分支 + gating 函数降级
- 新建 `vllm/model_executor/layers/fla/ops/pytorch_fallback.py` — 纯 PyTorch 的 GDN 算子实现

## 使用方法

### 前置条件

- 已安装 vLLM（虚拟环境路径默认为 `~/vllm-env`，可在脚本中修改）
- 已下载 Qwen3.5 模型（默认路径 `~/models/Qwen3.5-9B`）

### Step 1：应用补丁

```bash
source ~/vllm-env/bin/activate
python3 apply_gdn_patch.py
```

脚本会自动备份原文件（`.bak` 后缀），并注入 fallback 逻辑。如果提示 `already patched`，说明补丁已应用过，无需重复执行。

### Step 2：启动 vLLM

```bash
bash run_vllm.sh
```

`run_vllm.sh` 中的关键参数说明：

| 参数 / 环境变量 | 作用 |
|-----------------|------|
| `VLLM_GDN_PYTORCH_FALLBACK=1` | 激活 PyTorch fallback，核心开关 |
| `VLLM_ATTENTION_BACKEND=XFORMERS` | 注意力后端降级为 Xformers |
| `NCCL_P2P_DISABLE=1` | 禁用 GPU 间 P2P 直连通信 |
| `--enforce-eager` | 禁用 CUDA Graph，使用 Eager 模式 |
| `--disable-custom-all-reduce` | 禁用自定义 AllReduce，使用标准 NCCL |
| `--dtype float16` | 使用 FP16 精度（SM75 不支持 BF16） |
| `--pipeline-parallel-size 2` | 双卡流水线并行（见上文并行策略说明） |
| `--max-model-len 2048` | 限制最大序列长度以控制显存 |
| `--gpu-memory-utilization 0.85` | GPU 显存利用率上限 |

### 回滚补丁

如需恢复原始文件：

```bash
VENV=~/vllm-env
VLLM=$VENV/lib/python3.10/site-packages/vllm

cp $VLLM/model_executor/layers/fla/ops/fused_recurrent.py.bak \
   $VLLM/model_executor/layers/fla/ops/fused_recurrent.py

cp $VLLM/model_executor/models/qwen3_next.py.bak \
   $VLLM/model_executor/models/qwen3_next.py

rm $VLLM/model_executor/layers/fla/ops/pytorch_fallback.py
```

## 性能影响

- **推理延迟**：由于使用纯 PyTorch 逐步循环替代 Triton 并行内核，GDN 层的计算速度会显著下降，尤其在长序列 prefill 阶段
- **吞吐量**：禁用 CUDA Graph 和自定义 AllReduce 会进一步降低整体吞吐
- **精度**：数学上完全等价，不影响模型输出质量
- **适用场景**：适合对延迟不敏感、利用老硬件跑推理的场景（如内部测试、小规模服务）

## 适用环境

- GPU：Tesla T10 / RTX 2080 等 Turing 架构（SM75）显卡
- 模型：Qwen3.5 系列（使用 GDN 架构的模型）
- 框架：vLLM（已测试路径基于 `python3.10`，其他版本需修改 `apply_gdn_patch.py` 中的路径）

## 特别鸣谢

纸鸢随风、infamousgxy
