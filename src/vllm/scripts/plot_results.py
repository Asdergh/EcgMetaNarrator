"""scripts/plot_results.py — Шаг 26 плана.

Строит три графика (по одному файлу каждый, плюс комбинированный):
  Г1: GPU memory used % (и util %) vs число ботов
  Г2: output tokens/sec vs число ботов (точка насыщения)
  Г3: TTFT p95 vs число боты + линия бюджета (2000 мс)

Вход:
  --aggregated  results/aggregated.csv   (выход шага 22, может содержать обе модели)
  --gpu-summary results/gpu_summary.json (опционально; выход шага 19 по concurrency)
  --budget-ms   2000

Использование:
  python scripts/plot_results.py --aggregated results/aggregated.csv \
      --gpu-memory results/gpu_memory_by_concurrency.csv --outdir reports/figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_agg(path: Path) -> dict[str, dict[int, dict]]:
    """model -> concurrency -> row (усредняем по бакетам, чтобы график был читаемым)."""
    data: dict[str, dict[int, dict]] = defaultdict(dict)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_model_conc: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_model_conc[(r["model"], int(r["concurrency"]))].append(r)

    for (model, conc), rs in by_model_conc.items():
        def avg(key: str) -> float | None:
            vals = [float(x[key]) for x in rs if x.get(key) not in (None, "", "None")]
            return sum(vals) / len(vals) if vals else None

        data[model][conc] = {
            "ttft_p95": avg("ttft_p95_ms"),
            "throughput": avg("output_throughput_tok_s"),
            "rps": avg("request_throughput_rps"),
        }
    return data


def read_gpu_mem(path: Path | None) -> dict[str, dict[int, float]]:
    """model -> concurrency -> peak memory used % (если файл есть)."""
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        out[r["model"]][int(r["concurrency"])] = float(r["peak_memory_used_pct"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregated", required=True)
    ap.add_argument("--gpu-memory", default=None)
    ap.add_argument("--outdir", default="reports/figures")
    ap.add_argument("--budget-ms", type=float, default=2000.0)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = read_agg(Path(args.aggregated))
    gpu = read_gpu_mem(Path(args.gpu_memory) if args.gpu_memory else None)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Г1: GPU memory used % vs concurrency ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, by_conc in gpu.items():
        xs = sorted(by_conc)
        ys = [by_conc[c] for c in xs]
        ax.plot(xs, ys, marker="o", label=model)
    ax.set_xlabel("concurrency (bots)")
    ax.set_ylabel("peak GPU memory used, % of total")
    ax.set_title("GPU memory used vs concurrency")
    ax.grid(alpha=0.3)
    if gpu:
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig1_gpu_memory.png", dpi=150)
    plt.close(fig)

    # --- Г2: throughput vs concurrency ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, by_conc in agg.items():
        xs = sorted(by_conc)
        ys = [by_conc[c]["throughput"] for c in xs]
        ax.plot(xs, ys, marker="o", label=model)
    ax.set_xlabel("concurrency (bots)")
    ax.set_ylabel("output tokens / sec")
    ax.set_title("Throughput vs concurrency (saturation point)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig2_throughput.png", dpi=150)
    plt.close(fig)

    # --- Г3: TTFT p95 vs concurrency + бюджет ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, by_conc in agg.items():
        xs = sorted(by_conc)
        ys = [by_conc[c]["ttft_p95"] for c in xs]
        ax.plot(xs, ys, marker="o", label=model)
    ax.axhline(args.budget_ms, color="red", linestyle="--", label=f"budget {args.budget_ms:.0f} ms")
    ax.set_xlabel("concurrency (bots)")
    ax.set_ylabel("TTFT p95, ms")
    ax.set_title("TTFT p95 vs concurrency (budget breach point)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig3_ttft_p95.png", dpi=150)
    plt.close(fig)

    print(f"figures -> {outdir} (fig1_gpu_memory.png, fig2_throughput.png, fig3_ttft_p95.png)")


if __name__ == "__main__":
    main()
