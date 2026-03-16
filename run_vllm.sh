export VLLM_GDN_PYTORCH_FALLBACK=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

python3 -m vllm.entrypoints.openai.api_server \
  --model ~/models/Qwen3.5-9B \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --port 8002