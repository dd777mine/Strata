# NVIDIA GeForce RTX 2060 12 GB (sm_75): Fused IQ MoE (IQ3_S, IQ2, IQ4), BF16 Tensor Cores & GDN Keyhead (2026-10-18)

One NVIDIA GeForce RTX 2060 12 GB (TU106, sm_75) with Flash-Next IQ3_S (3.5-bit, 262K native context).

In Strata v0.1.45, `ASTAGES_DEV = 2` was added for `__CUDA_ARCH__ < 800`, but Fused IQ MoE was restricted to Q2_0 due to a comment in `moe_fused_iq.cu`:
```cpp
// Turing: the canonical Q2_0 pack's kernels (the codebook formats' do not fit 64 KB at 4 warps a row)
```
This change addresses the root cause of that comment: while `IQ2_S` and `IQ2_XS` at `WW=4` (4 warps a row) require 67.5 KB SMEM (exceeding Turing's 64 KB ceiling), **`IQ3_S` at `WW=4` requires only 61.6 KB** and fits cleanly under 64 KB. For `IQ2_S` and `IQ2_XS`, clamping to `WW=2` reduces SMEM to 43.0 KB.

With all formats fitting under 64 KB SMEM, Fused MoE is enabled for all supported formats on Turing (`sm_75`), alongside FP16 Tensor Core emulation for BF16 and GDN keyhead recurrence.

| What | Was on sm_75 | Now on sm_75 | Switch |
| --- | --- | --- | --- |
| Fused MoE Prefill (`moe_fused_iq.cu`) | Disabled for codebook formats (`native_supported` returned false) | Enabled for all supported formats (`setup_one_turing`, `WW=2` for IQ2_S/XS) | `STRATA_PF_FUSED=1` or `STRATA_PF_FUSED_SM75=1` |
| BF16 GEMMs (`gemm.cu`) | Scalar / excluded from tensor cores (`cc < 75 ? 1 : 0`) | FP16 Tensor Core emulation (`cc <= 75 ? 1 : 0`) | `STRATA_BF16_TC=0` to disable |
| GDN Keyhead Recurrence (`kernels.cu`) | Disabled (`major >= 8`) | Enabled for compute capability 7.5 | on by default |
| IQ3_S Grid Staging (`iq_kernels.cu`) | Staged from global memory | Staged in shared memory (`kStageIqGrid<21>`) + IMAD sign unpacker | on by default |

## Measured Results (Single RTX 2060 12 GB @ 200W VBIOS)

Through the server (`http://127.0.0.1:8080/v1/chat/completions`), greedy, temperature 0.0, Flash-Next IQ3_S (83.6 GB):

| Prompt / Workload | Stock Strata (`v0.1.45`) | This Change (sm_75) | Delta |
| --- | ---: | ---: | ---: |
| 169-token warmup | 2.43 s (69.6 tok/s) | **2.23 s (75.9 tok/s)** | -0.20 s (-8.2%) |
| 343-token prompt | 3.12 s (110.0 tok/s) | **2.86 s (120.1 tok/s)** | -0.26 s (-8.3%) |
| 1,213-token prompt | 7.69 s (157.8 tok/s) | **6.85 s (177.1 tok/s)** | -0.84 s (-10.9%) |
| 2,953-token prompt | 9.81 s (301.1 tok/s) | **7.69 s (383.9 tok/s)** | -2.12 s (-21.6%) |
| 5,853-token prompt | 14.11 s (414.8 tok/s) | **10.11 s (578.8 tok/s)** | **-4.00 s (-28.3% TTFT latency)** |
| 17,453-token prompt (OpenCode context) | 43.51 s (401.1 tok/s) | **32.68 s (534.1 tok/s)** | **-10.83 s (-24.9% TTFT latency)** |
| Clean Delta Prefill (~5.5K tokens) | 501.3 tok/s | **759.4 tok/s** | **+258.1 tok/s (+51.5% speedup)** |
| Decode (400 tokens, warm KV hit) | 12.69 s (31.53 tok/s) | **12.64 s (31.66 tok/s)** | Parity (31.7 tok/s) |
| Engine pure decode speed | 31.9 tok/s | **32.0 tok/s** | Parity (32.0 tok/s) |
| MTP speculative draft acceptance | 82.2% | **73.9% - 82.2%** | High speculative accuracy |

## Rig

- GPU: 1x NVIDIA GeForce RTX 2060 12 GB (TU106, sm_75, PCIe 3.0 x16), 200W VBIOS, VRAM at 8000 MHz (384 GB/s GDDR6 bandwidth).
- CPU: Intel Xeon E5-2650 v3 (10C/20T @ 2.30 GHz), 64 GB DDR4.
- OS: Ubuntu 24.04 (Linux 7.0), CUDA 12.4, GCC 13.
- Model: `Qwen3.8-Flash-Next` IQ3_S (3.5-bit GGUF, 83.6 GB, 262K native context limit).
- Config: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.50 --mtp --max-context 262144 --kv int8 --kv-resident 32768`.

## Standalone Reproduction Script

Provided in `tools/bench_turing_prefill.py` (pure standard library Python, no external dependencies):
```bash
python tools/bench_turing_prefill.py --url http://127.0.0.1:8080/v1/chat/completions --heavy
```
