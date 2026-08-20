"""GSPO step-time benchmark: fused 1-pass vs 2-pass custom-loss pattern.

Measures wall-clock per training step for the same frozen batch on the same
server, two ways:

  A (native, 1 pass):  forward_backward(datums, loss_fn="gspo")
  B (custom, 2 pass):  forward(datums)  ->  compute sequence-level GSPO
                       weights client-side  ->  forward_backward(
                       reweighted datums, loss_fn="importance_sampling")

Path B reproduces the round-trip pattern hosted Tinker requires for losses it
doesn't serve natively (fetch logprobs, compute loss client-side, ship back).
The client-side math computes real sequence-level clipped ratios; gradient
equivalence to fused GSPO is approximate, but timing — the quantity measured —
is exactly the 2-pass pattern's cost.

The batch is REAL rollout data: prompts sampled once from the model at the
start (variable lengths), then frozen and reused for every step of both paths.

Usage:
  python benchmarks/bench_gspo.py --base-url http://... --model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")

import tinker  # noqa: E402
import torch  # noqa: E402
from tinker import types  # noqa: E402
from tinker.types.tensor_data import TensorData  # noqa: E402

PROMPTS = [
    "Q: A baker makes 24 rolls and sells 3/4 of them. How many are left? A:",
    "Q: Tom has 5 boxes of 12 pencils and gives away 17. How many remain? A:",
    "Q: A train travels 60 mph for 2.5 hours. How far does it go? A:",
    "Q: Sara reads 45 pages a day for 6 days. Total pages? A:",
    "Q: A shirt costs $28 after a 30% discount. What was the original price? A:",
    "Q: There are 8 teams of 11 players. A third are injured. How many can play? A:",
    "Q: A tank holds 240L and drains at 15L/hour. Hours until empty? A:",
    "Q: Mia saves $7 a week. After how many weeks does she have $91? A:",
    "Q: A recipe needs 3 cups per batch. Cups needed for 7 batches? A:",
    "Q: A car uses 8L per 100km. Liters for a 350km trip? A:",
    "Q: 120 students split into groups of 9, remainder walk. How many walk? A:",
    "Q: A ladder of 13ft leans with base 5ft from the wall. Wall height reached? A:",
    "Q: Movie starts 7:45pm and runs 128 minutes. When does it end? A:",
    "Q: A worker earns $18/hour plus 50% overtime after 40 hours. Pay for 46 hours? A:",
    "Q: A garden 12m by 9m needs fencing. Total meters of fence? A:",
    "Q: If 4 machines make 4 widgets in 4 minutes, how long for 8 machines to make 8? A:",
]

GSPO_CLIP_LOW = 0.8
GSPO_CLIP_HIGH = 1.2


def build_frozen_batch(training_client, tokenizer, max_tokens: int):
    """Sample once, build rl_loop-style datums with real variable lengths."""
    sampling_client = training_client.save_weights_and_get_sampling_client()
    params = types.SamplingParams(max_tokens=max_tokens, temperature=1.0)
    futures = []
    prompts_enc = []
    for text in PROMPTS:
        model_input = types.ModelInput.from_ints(tokenizer.encode(text))
        prompts_enc.append(model_input)
        futures.append(
            sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=params)
        )
    datums, lengths = [], []
    for i, fut in enumerate(futures):
        seq = fut.result().sequences[0]
        sampled_tokens = list(seq.tokens)
        logprobs = [float(x) for x in seq.logprobs]
        prompt = prompts_enc[i]
        ob_len = prompt.length - 1
        model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
        target_tokens = [0] * ob_len + sampled_tokens
        padded_logprobs = [0.0] * ob_len + logprobs
        advantage = 1.0 if i % 2 == 0 else -1.0  # synthetic; shapes/lengths are what matter
        padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
        datums.append(
            types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                },
            )
        )
        lengths.append(len(sampled_tokens))
    return datums, lengths


def step_native(training_client, datums, adam):
    fb = training_client.forward_backward(
        datums,
        loss_fn="gspo",
        loss_fn_config={"clip_low_threshold": GSPO_CLIP_LOW, "clip_high_threshold": GSPO_CLIP_HIGH},
    )
    opt = training_client.optim_step(adam)
    fb.result()
    opt.result()


def step_custom_2pass(training_client, datums, adam):
    # Pass 1: fetch current per-token logprobs.
    fwd = training_client.forward(datums, loss_fn="cross_entropy").result()
    new_lp_per_datum = [out["logprobs"].to_torch() for out in fwd.loss_fn_outputs]

    # Client-side GSPO: sequence-level clipped importance ratios.
    reweighted = []
    for datum, new_lp in zip(datums, new_lp_per_datum):
        old_lp = datum.loss_fn_inputs["logprobs"].to_torch()
        adv = datum.loss_fn_inputs["advantages"].to_torch()
        mask = adv != 0
        n = int(mask.sum().item()) or 1
        s = math.exp(float((new_lp[mask] - old_lp[mask]).mean().item()))
        inside = GSPO_CLIP_LOW <= s <= GSPO_CLIP_HIGH
        new_adv = adv * (s if inside else 0.0)
        reweighted.append(
            types.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": datum.loss_fn_inputs["target_tokens"],
                    "logprobs": TensorData.from_torch(new_lp),  # ratio starts at 1
                    "advantages": TensorData.from_torch(new_adv),
                },
            )
        )

    # Pass 2: backward through importance sampling with the reweighted batch.
    fb = training_client.forward_backward(reweighted, loss_fn="importance_sampling")
    opt = training_client.optim_step(adam)
    fb.result()
    opt.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--lora-rank", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    service = tinker.ServiceClient(base_url=args.base_url)
    training_client = service.create_lora_training_client(
        base_model=args.model, rank=args.lora_rank
    )
    adam = types.AdamParams(learning_rate=1e-5)

    print("Sampling frozen batch ...", flush=True)
    datums, lengths = build_frozen_batch(training_client, tokenizer, args.max_tokens)
    total_target_tokens = sum(d.model_input.length for d in datums)
    print(f"batch: {len(datums)} seqs, sampled lengths min/mean/max = "
          f"{min(lengths)}/{sum(lengths)/len(lengths):.0f}/{max(lengths)}, "
          f"total tokens/step = {total_target_tokens}", flush=True)

    results = {}
    for name, step_fn in [("native_1pass", step_native), ("custom_2pass", step_custom_2pass)]:
        step_fn(training_client, datums, adam)  # warmup (JIT/caches), untimed
        times = []
        for i in range(args.steps):
            t0 = time.perf_counter()
            step_fn(training_client, datums, adam)
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"[{name}] step {i + 1}/{args.steps}: {dt:.2f}s", flush=True)
        mean = statistics.mean(times)
        results[name] = {
            "mean_s": round(mean, 3),
            "stdev_s": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
            "tokens_per_s": round(total_target_tokens / mean, 1),
            "times": [round(t, 3) for t in times],
        }

    a, b = results["native_1pass"], results["custom_2pass"]
    results["summary"] = {
        "model": args.model,
        "batch_seqs": len(datums),
        "total_tokens_per_step": total_target_tokens,
        "step_time_reduction_pct": round(100 * (1 - a["mean_s"] / b["mean_s"]), 1),
        "throughput_gain_pct": round(100 * (a["tokens_per_s"] / b["tokens_per_s"] - 1), 1),
    }
    print("BENCH_RESULT " + json.dumps(results["summary"]), flush=True)
    with open("benchmarks/last_gspo_bench.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
