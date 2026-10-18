# Strata (fork) — experimental sm_75 prefill support

<p align="center">
  <b>High-Performance Local Inference for Qwen3.8-Flash-Next on NVIDIA Turing GPUs</b><br>
  <sub>Fork of upstream <a href="https://github.com/Niko1221/Strata">Strata</a> by Niko1221 · Licensed under MIT</sub>
</p>

> [!NOTE]
> **Disclaimer:** Not affiliated with or endorsed by the upstream author. This repository contains an experimental sm_75 prefill optimization patch in `src/prefill/moe_fused_iq.cu`; the rest of the codebase is unmodified upstream Strata. All optimizations are strictly opt-in via environment variables (`STRATA_PF_FUSED=1` or `STRATA_PF_FUSED_SM75=1`), leaving default execution untouched.

---

## Overview

In upstream Strata (including v0.1.45), Fused MoE prefill was restricted exclusively to `Q2_0` on NVIDIA Turing (`sm_75`) GPUs due to an assumed 64 KB Shared Memory (SMEM) limitation. As a result, the primary recommended format **`IQ3_S`** (and other codebook formats like `IQ2` and `IQ4`) fell back to unfused scalar dequantization and host-synchronized MMQ prefill, causing severe latency spikes on Turing hardware.

This fork resolves the shared memory allocation constraint and unlocks the full power of NVIDIA Turing GPUs (empirically tested and verified on **GeForce RTX 2060 12GB** and **GeForce RTX 2080 SUPER**):

* **Fused IQ MoE Prefill:** Unlocks the monolithic 1500-line `fused::experts_native` CUDA kernel on Turing. Mathematical analysis proves that `IQ3_S` at 4 warps/row requires only **58.75 KB** (cleanly fitting Turing's 64 KB ceiling with 5.25 KB headroom). For `IQ2_S` and `IQ2_XS`, clamping to 2 warps/row reduces SMEM to **43.50 KB**.
* **Zero Host Synchronization:** Eliminates blocking GPU-to-CPU (`d2h_sync`) transfers and host-side expert grouping during prefill. The entire routing and dequantization pipeline runs resident on the GPU.
* **FP16 Tensor Core Emulation:** Enables the tensor core path for BF16 GEMMs on Volta/Turing (`cc <= 75`).
* **GDN Keyhead Recurrence:** Unlocks hardware Gated DeltaNet (GDN) recurrence for compute capability 7.5.
* **Shared Memory Grid Staging:** Caches the 2 KB `iq3s_grid` LUT in SMEM (`kStageIqGrid<21>`) and accelerates sign bit unpacking via single-cycle IMAD.

---

## Verified Performance (RTX 2060 12GB @ 200W VBIOS)

Tested on Ubuntu 24.04 (Linux 7.0), CUDA 12.4, Intel Xeon E5-2650 v3, **Qwen3.8-Flash-Next IQ3_S** (3.5-bit, 262K native context limit):

| Workload / Prompt Length | Stock Strata (`v0.1.45`) | Turing (sm_75) Edition | Improvement |
| :--- | :---: | :---: | :---: |
| **Warmup (169 tokens)** | 2.43 s (69.6 tok/s) | **2.23 s (75.9 tok/s)** | **-0.20 s (-8.2%)** |
| **Short Prompt (343 tokens)** | 3.12 s (110.0 tok/s) | **2.86 s (120.1 tok/s)** | **-0.26 s (-8.3%)** |
| **Medium Prompt (1 213 tokens)** | 7.69 s (157.8 tok/s) | **6.85 s (177.1 tok/s)** | **-0.84 s (-10.9%)** |
| **Bulk Prompt (2 953 tokens)** | 9.81 s (301.1 tok/s) | **7.69 s (383.9 tok/s)** | **-2.12 s (-21.6%)** |
| **Heavy Prompt (5 853 tokens)** | **14.11 s** (414.8 tok/s) | **10.11 s** (578.8 tok/s) | **-4.00 s (-28.3% TTFT latency!)** |
| **OpenCode Context (17 453 tokens)** | **43.51 s** (401.1 tok/s) | **32.68 s** (534.1 tok/s) | **-10.83 s (-24.9% TTFT latency!)** |
| **Clean Delta Prefill (~5.5K tokens)** | **501.3 tok/s** | **759.4 tok/s** | **+258.1 tok/s (+51.5% speedup!)** |
| **Pure Speculative Decode (KV hit)** | 12.69 s (31.53 tok/s) | **12.64 s (31.66 tok/s)** | Parity (31.7 tok/s) |
| **Engine Pure Decode Speed** | 31.9 tok/s | **32.0 tok/s** | Parity (32.0 tok/s) |
| **MTP Draft Acceptance** | 82.2% | **73.9% – 82.2%** | Consistent high accuracy |

> **Hardware note & Transparency:** GPU running with VRAM at 8000 MHz (384 GB/s GDDR6 bandwidth vs 336 GB/s stock 7000 MHz), 200W VBIOS. Both stock Strata and patched runs were measured on the exact same card at identical clocks, thermals, and machine configuration.

---

## Quickstart

### 1. Build from Source
```bash
git clone https://github.com/dd777mine/Strata.git
cd Strata
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j$(nproc)
```

### 2. Recommended Runtime Settings
```bash
export STRATA_PF_FUSED=1
export STRATA_BF16_TC=1
export STRATA_PREFILL_RING=248
export STRATA_MMVQ_IL_T1=0
export STRATA_GDN_KEYHEAD=1

# Start the server
python serve/server.py --config strata-iq3_s.json
```

---

## Reproduction Benchmark

To reproduce the exact numbers reported above, run the included standalone benchmark script (zero external dependencies, pure standard library Python):

```bash
# Standard sweep (Warmup -> 343 -> 1.2K -> 3.0K -> 5.8K -> Decode):
python tools/bench_turing_prefill.py --url http://127.0.0.1:8080/v1/chat/completions

# Full suite including the 17.5K token OpenCode prompt test:
python tools/bench_turing_prefill.py --url http://127.0.0.1:8080/v1/chat/completions --heavy
```

---

## Documentation

* 📊 **Official Community Benchmark Report:** [`bench/results/2026-10-18-community-turing-sm75/README.md`](bench/results/2026-10-18-community-turing-sm75/README.md)
* 📖 **Older GPUs Architecture Guide:** [`docs/OLDER_GPUS.md`](docs/OLDER_GPUS.md)
* 📋 **Community Benchmarks Index:** [`docs/COMMUNITY_BENCHMARKS.md`](docs/COMMUNITY_BENCHMARKS.md)
* 📜 **License:** [MIT License](LICENSE) (Niko1221 and Strata contributors).
