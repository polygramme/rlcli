"""Render launch assets: GSM8K reward curve + GSPO 1-pass-vs-2-pass bars.

Usage:
  python benchmarks/plot_assets.py --run-dir ~/.rlcli/runs/rl-... \
      --bench benchmarks/last_gspo_bench.json --out benchmarks/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1B2328"
BLUE = "#3B6FD4"
TAN = "#C9BFA8"


def plot_curve(run_dir: Path, out: Path) -> Path | None:
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None
    steps, correct = [], []
    for line in metrics_file.read_text().splitlines():
        row = json.loads(line)
        if "env/all/correct" in row:
            steps.append(row["step"])
            correct.append(100 * row["env/all/correct"])
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.plot(steps, correct, color=BLUE, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("GSM8K train accuracy (%)")
    ax.set_title("Qwen3-0.6B · GSM8K · rlcli train rl --loss gspo · 1×L4 GPU")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = out / "curve_gsm8k_gspo.png"
    fig.savefig(path)
    return path


def plot_bench(bench_file: Path, out: Path) -> Path | None:
    if not bench_file.exists():
        return None
    d = json.loads(bench_file.read_text())
    a, b, s = d["native_1pass"], d["custom_2pass"], d["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=200)
    for ax, vals, unit, better in [
        (axes[0], (b["mean_s"], a["mean_s"]), "step time (s)", "lower"),
        (axes[1], (b["tokens_per_s"], a["tokens_per_s"]), "tokens / s", "higher"),
    ]:
        bars = ax.barh(["2-pass custom\n(hosted pattern)", "fused 1-pass\nforward_backward(\"gspo\")"],
                       vals, color=[TAN, BLUE], height=0.55)
        for bar, v in zip(bars, vals):
            ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:g}",
                    va="center", fontsize=9, color=INK)
        ax.set_xlabel(f"{unit} · {better} is better")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"GSPO: {s['step_time_reduction_pct']}% faster steps, "
        f"+{s['throughput_gain_pct']}% throughput — {s['model']}, same A100, same batch",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / "bench_gspo_bars.png"
    fig.savefig(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--bench", type=Path, default=Path("benchmarks/last_gspo_bench.json"))
    ap.add_argument("--out", type=Path, default=Path("benchmarks"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for result in (plot_curve(args.run_dir.expanduser(), args.out),
                   plot_bench(args.bench, args.out)):
        if result:
            print("wrote", result)
