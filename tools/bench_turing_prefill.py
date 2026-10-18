#!/usr/bin/env python3
"""
bench_turing_prefill.py - Empirical prefill & decode benchmark for Strata

Reproduces the official Turing (sm_75) benchmarks across prompt sizes:
  - Warmup (169 tokens)
  - Short prompts (343 tokens, 1 213 tokens)
  - Medium / Bulk prompts (2 953 tokens, 5 853 tokens)
  - Heavy OpenCode IDE prompt (17 453 tokens, with --heavy)
  - Pure speculative decode (200 tokens KV hit)

Usage:
  python tools/bench_turing_prefill.py
  python tools/bench_turing_prefill.py --url http://127.0.0.1:8080/v1/chat/completions --heavy
"""

import argparse
import json
import sys
import time
import urllib.request

# Base payload text for constructing realistic prompts
BASE_TEXT = (
    "Разработка распределенных высокопроизводительных систем машинного обучения требует "
    "глубокого понимания архитектуры современных графических процессоров, протоколов "
    "межсоединений NVLink и PCIe Gen4, а также эффективного управления иерархией памяти VRAM и системной RAM. "
)

def query(url: str, model: str, prompt: str, max_tokens: int = 1, api_key: str = "", timeout: int = 300):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    t1 = time.time()
    
    usage = res.get("usage", {})
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)
    return t1 - t0, p_tok, c_tok, res

def main():
    parser = argparse.ArgumentParser(description="Strata Turing (sm_75) Prefill & Decode Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions", help="Completions endpoint URL")
    parser.add_argument("--model", default="qwen3.8-flash-next-iq3_s", help="Model name")
    parser.add_argument("--api-key", default="", help="Optional API key")
    parser.add_argument("--heavy", action="store_true", help="Include heavy 17.5K token OpenCode prompt test")
    args = parser.parse_args()

    print("=" * 70)
    print("      STRATA TURING (sm_75) PREFILL & DECODE BENCHMARK")
    print("=" * 70)
    print(f"Target URL: {args.url}")
    print(f"Model:      {args.model}\n")

    # 1. Warmup
    print("[1/4] Running Warmup (169 tokens)...")
    try:
        dt, p_tok, _, _ = query(args.url, args.model, BASE_TEXT * 2, max_tokens=1, api_key=args.api_key)
        print(f"      Warmup complete: {p_tok} prompt tokens in {dt:.2f}s ({p_tok / dt:.1f} tok/s)\n")
    except Exception as e:
        print(f"ERROR connecting to Strata server at {args.url}: {e}")
        sys.exit(1)

    # 2. Sweep across prompt sizes
    multipliers = [5, 20, 50, 100]
    if args.heavy:
        multipliers.append(300) # ~17.5K tokens
        
    results = []
    print("[2/4] Measuring Time-to-First-Token (TTFT) across prompt sizes...")
    print(f"  {'Prompt Tokens':>14} | {'TTFT (sec)':>10} | {'Throughput (tok/s)':>18}")
    print("  " + "-" * 48)
    
    for m in multipliers:
        prompt = BASE_TEXT * m
        dt, p_tok, _, _ = query(args.url, args.model, prompt, max_tokens=1, api_key=args.api_key)
        tps = p_tok / dt if dt > 0 else 0
        results.append((p_tok, dt, tps))
        print(f"  {p_tok:14d} | {dt:9.2f}s | {tps:17.1f} tok/s")
        time.sleep(0.5)

    # 3. Clean Delta Prefill (steady state derivative between ~340 and ~5850 tokens)
    if len(results) >= 4:
        p_base, t_base, _ = results[0]
        p_5k, t_5k, _ = results[3]
        delta_p = p_5k - p_base
        delta_t = t_5k - t_base
        if delta_t > 0:
            delta_tps = delta_p / delta_t
            print("  " + "-" * 48)
            print(f"  >>> Clean Bulk Delta Prefill: {delta_tps:.1f} tok/s (over {delta_p} tokens)\n")

    # 4. Pure Speculative Decode (KV hit)
    print("[3/4] Measuring Pure Decode Speed (200 tokens generation)...")
    short_prompt = "Explain quicksort algorithm in Python with code examples and time complexity analysis."
    # Warm prompt in KV cache
    query(args.url, args.model, short_prompt, max_tokens=1, api_key=args.api_key)
    # Generate 200 tokens
    dt, p_tok, c_tok, res = query(args.url, args.model, short_prompt, max_tokens=200, api_key=args.api_key)
    decode_tps = c_tok / dt if dt > 0 else 0
    print(f"      KV Hit: {p_tok} prompt tokens | Generated: {c_tok} tokens in {dt:.2f}s")
    print(f"      Generation Speed (End-to-End): {decode_tps:.2f} tok/s\n")

    print("[4/4] Benchmark completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
