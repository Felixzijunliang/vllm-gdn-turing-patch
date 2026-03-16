#!/usr/bin/env python3
"""
Patch vLLM to add PyTorch fallback for GDN (Gated Delta Net) operations.
Enables Qwen3.5 to run on Turing (SM75) GPUs where Triton kernels may crash.

Usage:
    source ~/vllm-env/bin/activate
    python3 ~/apply_gdn_patch.py

Then add to run_vllm.sh:
    export VLLM_GDN_PYTORCH_FALLBACK=1
"""
import os
import shutil

VENV = os.path.expanduser("~/vllm-env")
VLLM = f"{VENV}/lib/python3.10/site-packages/vllm"
FLA_OPS = f"{VLLM}/model_executor/layers/fla/ops"
MODELS = f"{VLLM}/model_executor/models"


def create_pytorch_fallback():
    path = f"{FLA_OPS}/pytorch_fallback.py"
    code = (
        '# Pure PyTorch fallback for GDN ops (Turing SM75 compatibility)\n'
        'import torch\n'
        'import torch.nn.functional as F\n'
        '\n'
        '\n'
        'def pytorch_fused_recurrent_gated_delta_rule_fwd(\n'
        '    q, k, v, g, beta, scale, initial_state,\n'
        '    inplace_final_state=True, cu_seqlens=None,\n'
        '    ssm_state_indices=None, num_accepted_tokens=None,\n'
        '    use_qk_l2norm_in_kernel=False,\n'
        '):\n'
        '    B, T_total, H, K = q.shape\n'
        '    HV, V = v.shape[2], v.shape[3]\n'
        '    N = B if cu_seqlens is None else len(cu_seqlens) - 1\n'
        '    heads_per_kv = HV // H\n'
        '    o = torch.zeros_like(v)\n'
        '    if inplace_final_state:\n'
        '        final_state = initial_state\n'
        '    else:\n'
        '        final_state = initial_state.new_empty(T_total, HV, V, K)\n'
        '    for i_n in range(N):\n'
        '        if cu_seqlens is not None:\n'
        '            bos = cu_seqlens[i_n].item()\n'
        '            eos = cu_seqlens[i_n + 1].item()\n'
        '            b_idx = 0\n'
        '        else:\n'
        '            bos, eos = 0, T_total\n'
        '            b_idx = i_n\n'
        '        T = eos - bos\n'
        '        if T == 0:\n'
        '            continue\n'
        '        if ssm_state_indices is not None:\n'
        '            if num_accepted_tokens is not None:\n'
        '                init_tok = num_accepted_tokens[i_n].item() - 1\n'
        '            else:\n'
        '                init_tok = 0\n'
        '            if ssm_state_indices.ndim > 1:\n'
        '                state_idx = ssm_state_indices[i_n, init_tok].item()\n'
        '            else:\n'
        '                state_idx = ssm_state_indices[i_n].item()\n'
        '            if state_idx < 0:\n'
        '                continue\n'
        '        else:\n'
        '            state_idx = i_n\n'
        '        b_h = initial_state[state_idx].float().clone()\n'
        '        for i_t in range(T):\n'
        '            t = bos + i_t\n'
        '            b_q = q[b_idx, t].float()\n'
        '            b_k = k[b_idx, t].float()\n'
        '            if use_qk_l2norm_in_kernel:\n'
        '                b_q = F.normalize(b_q, p=2, dim=-1, eps=1e-6)\n'
        '                b_k = F.normalize(b_k, p=2, dim=-1, eps=1e-6)\n'
        '            b_q = b_q * scale\n'
        '            b_q = b_q.repeat_interleave(heads_per_kv, dim=0)\n'
        '            b_k = b_k.repeat_interleave(heads_per_kv, dim=0)\n'
        '            b_v = v[b_idx, t].float()\n'
        '            b_g = g[b_idx, t].float()\n'
        '            b_beta = beta[b_idx, t].float()\n'
        '            b_h = b_h * torch.exp(b_g).unsqueeze(-1).unsqueeze(-1)\n'
        '            delta_v = b_v - torch.einsum("hvk,hk->hv", b_h, b_k)\n'
        '            delta_v = delta_v * b_beta.unsqueeze(-1)\n'
        '            b_h = b_h + torch.einsum("hv,hk->hvk", delta_v, b_k)\n'
        '            b_o = torch.einsum("hvk,hk->hv", b_h, b_q)\n'
        '            o[b_idx, t] = b_o.to(o.dtype)\n'
        '            if inplace_final_state and ssm_state_indices is not None:\n'
        '                if ssm_state_indices.ndim > 1:\n'
        '                    f_idx = ssm_state_indices[i_n, i_t].item()\n'
        '                else:\n'
        '                    f_idx = ssm_state_indices[i_n].item()\n'
        '                if f_idx >= 0:\n'
        '                    final_state[f_idx] = b_h.to(final_state.dtype)\n'
        '            elif not inplace_final_state:\n'
        '                final_state[bos + i_t] = b_h.to(final_state.dtype)\n'
        '    return o, final_state\n'
        '\n'
        '\n'
        'def pytorch_chunk_gated_delta_rule(\n'
        '    q, k, v, g, beta, scale=None, initial_state=None,\n'
        '    output_final_state=False, cu_seqlens=None,\n'
        '    use_qk_l2norm_in_kernel=False,\n'
        '):\n'
        '    if use_qk_l2norm_in_kernel:\n'
        '        q = F.normalize(q, p=2, dim=-1, eps=1e-6)\n'
        '        k = F.normalize(k, p=2, dim=-1, eps=1e-6)\n'
        '    B, T_total, H, K = q.shape\n'
        '    V = v.shape[-1]\n'
        '    if scale is None:\n'
        '        scale = K ** -0.5\n'
        '    q = q * scale\n'
        '    N = B if cu_seqlens is None else len(cu_seqlens) - 1\n'
        '    output = torch.zeros_like(v)\n'
        '    final_states = []\n'
        '    for i_n in range(N):\n'
        '        if cu_seqlens is not None:\n'
        '            bos = cu_seqlens[i_n].item()\n'
        '            eos = cu_seqlens[i_n + 1].item()\n'
        '            b_idx = 0\n'
        '        else:\n'
        '            bos, eos = 0, T_total\n'
        '            b_idx = i_n\n'
        '        T = eos - bos\n'
        '        if T == 0:\n'
        '            if output_final_state:\n'
        '                final_states.append(torch.zeros(H, V, K, dtype=torch.float32, device=q.device))\n'
        '            continue\n'
        '        if initial_state is not None:\n'
        '            state = initial_state[i_n].float().clone()\n'
        '        else:\n'
        '            state = torch.zeros(H, V, K, dtype=torch.float32, device=q.device)\n'
        '        for i_t in range(T):\n'
        '            t = bos + i_t\n'
        '            q_t = q[b_idx, t].float()\n'
        '            k_t = k[b_idx, t].float()\n'
        '            v_t = v[b_idx, t].float()\n'
        '            g_t = g[b_idx, t].float()\n'
        '            beta_t = beta[b_idx, t].float()\n'
        '            state = state * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)\n'
        '            delta_v = v_t - torch.einsum("hvk,hk->hv", state, k_t)\n'
        '            delta_v = delta_v * beta_t.unsqueeze(-1)\n'
        '            state = state + torch.einsum("hv,hk->hvk", delta_v, k_t)\n'
        '            o_t = torch.einsum("hvk,hk->hv", state, q_t)\n'
        '            output[b_idx, t] = o_t.to(output.dtype)\n'
        '        if output_final_state:\n'
        '            final_states.append(state)\n'
        '    final_state = torch.stack(final_states, dim=0) if output_final_state else None\n'
        '    return output, final_state\n'
        '\n'
        '\n'
        'def pytorch_fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):\n'
        '    x = a.float() + dt_bias.float()\n'
        '    softplus_x = torch.where(\n'
        '        beta * x <= threshold,\n'
        '        (1.0 / beta) * torch.log1p(torch.exp(beta * x)),\n'
        '        x,\n'
        '    )\n'
        '    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(0)\n'
        '    beta_output = b.sigmoid().unsqueeze(0)\n'
        '    return g, beta_output\n'
    )
    with open(path, 'w') as f:
        f.write(code)
    print(f"  Created: {path}")


def patch_fused_recurrent():
    path = f"{FLA_OPS}/fused_recurrent.py"
    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  Backed up: {backup}")

    with open(path, 'r') as f:
        content = f.read()

    if "VLLM_GDN_PYTORCH_FALLBACK" in content:
        print("  fused_recurrent.py already patched, skipping")
        return

    if 'import os' not in content:
        content = content.replace('import torch', 'import os\nimport torch', 1)

    old = (
        '    o, final_state = FusedRecurrentFunction.apply(\n'
        '        q,\n'
        '        k,\n'
        '        v,\n'
        '        g,\n'
        '        beta,\n'
        '        scale,\n'
        '        initial_state,\n'
        '        inplace_final_state,\n'
        '        cu_seqlens,\n'
        '        ssm_state_indices,\n'
        '        num_accepted_tokens,\n'
        '        use_qk_l2norm_in_kernel,\n'
        '    )\n'
        '    return o, final_state'
    )

    new = (
        '    if os.environ.get("VLLM_GDN_PYTORCH_FALLBACK") == "1":\n'
        '        from .pytorch_fallback import pytorch_fused_recurrent_gated_delta_rule_fwd\n'
        '        o, final_state = pytorch_fused_recurrent_gated_delta_rule_fwd(\n'
        '            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),\n'
        '            g=g.contiguous(), beta=beta.contiguous(), scale=scale,\n'
        '            initial_state=initial_state, inplace_final_state=inplace_final_state,\n'
        '            cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,\n'
        '            num_accepted_tokens=num_accepted_tokens,\n'
        '            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n'
        '        )\n'
        '        return o, final_state\n'
        '\n'
        '    o, final_state = FusedRecurrentFunction.apply(\n'
        '        q,\n'
        '        k,\n'
        '        v,\n'
        '        g,\n'
        '        beta,\n'
        '        scale,\n'
        '        initial_state,\n'
        '        inplace_final_state,\n'
        '        cu_seqlens,\n'
        '        ssm_state_indices,\n'
        '        num_accepted_tokens,\n'
        '        use_qk_l2norm_in_kernel,\n'
        '    )\n'
        '    return o, final_state'
    )

    if old in content:
        content = content.replace(old, new)
        print("  Patched: fused_recurrent_gated_delta_rule()")
    else:
        print("  WARNING: target not found in fused_recurrent.py")

    with open(path, 'w') as f:
        f.write(content)


def patch_qwen3_next():
    path = f"{MODELS}/qwen3_next.py"
    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  Backed up: {backup}")

    with open(path, 'r') as f:
        content = f.read()

    if "VLLM_GDN_PYTORCH_FALLBACK" in content:
        print("  qwen3_next.py already patched, skipping")
        return

    if 'import os\n' not in content:
        content = content.replace(
            'from collections.abc import Iterable',
            'import os\nfrom collections.abc import Iterable',
            1
        )

    # 1. Patch ChunkGatedDeltaRule.__init__
    old_init = (
        '    def __init__(self) -> None:\n'
        '        super().__init__()\n'
        '        if current_platform.is_cuda() and current_platform.is_device_capability(90):\n'
        '            logger.info_once(\n'
        '                "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"\n'
        '            )\n'
        '            self._forward_method = self.forward_cuda\n'
        '        else:\n'
        '            self._forward_method = self.forward_native'
    )
    new_init = (
        '    def __init__(self) -> None:\n'
        '        super().__init__()\n'
        '        if os.environ.get("VLLM_GDN_PYTORCH_FALLBACK") == "1":\n'
        '            logger.info_once(\n'
        '                "Using PyTorch fallback for GDN (VLLM_GDN_PYTORCH_FALLBACK=1)"\n'
        '            )\n'
        '            self._forward_method = self.forward_pytorch\n'
        '        elif current_platform.is_cuda() and current_platform.is_device_capability(90):\n'
        '            logger.info_once(\n'
        '                "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"\n'
        '            )\n'
        '            self._forward_method = self.forward_cuda\n'
        '        else:\n'
        '            self._forward_method = self.forward_native'
    )
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("  Patched: ChunkGatedDeltaRule.__init__()")
    else:
        print("  WARNING: ChunkGatedDeltaRule.__init__ target not found")

    # 2. Add forward_pytorch method
    old_native_end = (
        '        return fla_chunk_gated_delta_rule(\n'
        '            q=q,\n'
        '            k=k,\n'
        '            v=v,\n'
        '            g=g,\n'
        '            beta=beta,\n'
        '            initial_state=initial_state,\n'
        '            output_final_state=output_final_state,\n'
        '            cu_seqlens=cu_seqlens,\n'
        '            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n'
        '        )\n'
        '\n'
        '\n'
        'class Qwen3NextSparseMoeBlock'
    )
    new_native_end = (
        '        return fla_chunk_gated_delta_rule(\n'
        '            q=q,\n'
        '            k=k,\n'
        '            v=v,\n'
        '            g=g,\n'
        '            beta=beta,\n'
        '            initial_state=initial_state,\n'
        '            output_final_state=output_final_state,\n'
        '            cu_seqlens=cu_seqlens,\n'
        '            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n'
        '        )\n'
        '\n'
        '    def forward_pytorch(\n'
        '        self,\n'
        '        q: torch.Tensor,\n'
        '        k: torch.Tensor,\n'
        '        v: torch.Tensor,\n'
        '        g: torch.Tensor,\n'
        '        beta: torch.Tensor,\n'
        '        initial_state: torch.Tensor,\n'
        '        output_final_state: bool,\n'
        '        cu_seqlens: torch.LongTensor | None = None,\n'
        '        use_qk_l2norm_in_kernel: bool = True,\n'
        '    ):\n'
        '        from vllm.model_executor.layers.fla.ops.pytorch_fallback import (\n'
        '            pytorch_chunk_gated_delta_rule,\n'
        '        )\n'
        '        return pytorch_chunk_gated_delta_rule(\n'
        '            q=q, k=k, v=v, g=g, beta=beta,\n'
        '            initial_state=initial_state,\n'
        '            output_final_state=output_final_state,\n'
        '            cu_seqlens=cu_seqlens,\n'
        '            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n'
        '        )\n'
        '\n'
        '\n'
        'class Qwen3NextSparseMoeBlock'
    )
    if old_native_end in content:
        content = content.replace(old_native_end, new_native_end)
        print("  Patched: added forward_pytorch()")
    else:
        print("  WARNING: forward_native end target not found")

    # 3. Patch fused_gdn_gating
    old_gating = (
        '    TODO maybe use torch.compile to replace this triton kernel\n'
        '    """\n'
        '    batch, num_heads = a.shape'
    )
    new_gating = (
        '    TODO maybe use torch.compile to replace this triton kernel\n'
        '    """\n'
        '    if os.environ.get("VLLM_GDN_PYTORCH_FALLBACK") == "1":\n'
        '        from vllm.model_executor.layers.fla.ops.pytorch_fallback import (\n'
        '            pytorch_fused_gdn_gating,\n'
        '        )\n'
        '        return pytorch_fused_gdn_gating(A_log, a, b, dt_bias, beta, threshold)\n'
        '    batch, num_heads = a.shape'
    )
    if old_gating in content:
        content = content.replace(old_gating, new_gating)
        print("  Patched: fused_gdn_gating()")
    else:
        print("  WARNING: fused_gdn_gating target not found")

    with open(path, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    print("=" * 50)
    print("Applying PyTorch GDN fallback patch for vLLM")
    print("=" * 50)
    print()

    if not os.path.exists(FLA_OPS):
        print(f"ERROR: {FLA_OPS} not found")
        exit(1)

    print("[1/3] Creating pytorch_fallback.py ...")
    create_pytorch_fallback()

    print("[2/3] Patching fused_recurrent.py ...")
    patch_fused_recurrent()

    print("[3/3] Patching qwen3_next.py ...")
    patch_qwen3_next()

    print()
    print("Done! To use, add to run_vllm.sh:")
    print("  export VLLM_GDN_PYTORCH_FALLBACK=1")
    print()
    print("To revert:")
    print(f"  cp {FLA_OPS}/fused_recurrent.py.bak {FLA_OPS}/fused_recurrent.py")
    print(f"  cp {MODELS}/qwen3_next.py.bak {MODELS}/qwen3_next.py")
    print(f"  rm {FLA_OPS}/pytorch_fallback.py")
