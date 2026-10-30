"""scripts/plot_matrix.py — графики по результатам матрицы (целевой режим 1K/2K).

Три графика:
  fig1: GPU memory used % vs concurrency (3 модели)
  fig2: throughput tok/s vs concurrency (3 модели)
  fig3: TTFT p95 vs concurrency, с бюджетом 2000 мс, лог-шкала Y
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_agg(path):
    data = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        model = r["model"]
        c = int(r["concurrency"])
        def f(k):
            v = r.get(k, "")
            try:
                return float(v) if v not in ("", None) else None
            except ValueError:
                return None
        data[model][c] = {
            "ttft_p95": f("ttft_p95_ms"),
            "throughput": f("output_throughput_tok_s"),
            "tokens": f("output_tokens_mean"),
        }
    return data


def read_gpu(path):
    out = defaultdict(dict)
    if not Path(path).exists():
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            out[r["model"]][int(r["concurrency"])] = float(r["peak_memory_used_pct"])
        except (ValueError, KeyError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregated", required=True)
    ap.add_argument("--gpu-memory", default=None)
    ap.add_argument("--outdir", default="reports/figures")
    ap.add_argument("--budget-ms", type=float, default=2000.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = read_agg(args.aggregated)
    gpu = read_gpu(args.gpu_memory) if args.gpu_memory else {}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    colors = {"qwen25-3b-mx": "#e53e3e", "qwen25-1.5b-mx": "#3182ce", "rusmind-malysh-mx": "#38a169"}
    labels = {"qwen25-3b-mx": "Qwen 3B", "qwen25-1.5b-mx": "Qwen 1.5B", "rusmind-malysh-mx": "Malysh 0.8B"}

    def style(model):
        return colors.get(model, "gray"), labels.get(model, model)

    # --- fig1: GPU memory ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, by_c in gpu.items():
        xs = sorted(by_c)
        ys = [by_c[c] for c in xs]
        color, label = style(model)
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=8, color=color, label=label)
    ax.axhline(95, color="red", linestyle="--", linewidth=1, label="опасно (95%)")
    ax.set_xlabel("concurrency (агентов)")
    ax.set_ylabel("peak GPU memory, % от 8188 МБ")
    ax.set_title("Пик VRAM vs конкурентность (целевой режим 1K/2K, окно 3072)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig1_gpu_memory.png", dpi=150)
    plt.close(fig)

    # --- fig2: throughput ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, by_c in agg.items():
        xs = [c for c in sorted(by_c) if by_c[c]["throughput"] is not None]
        ys = [by_c[c]["throughput"] for c in xs]
        color, label = style(model)
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=8, color=color, label=label)
    ax.set_xlabel("concurrency (агентов)")
    ax.set_ylabel("output tokens / sec")
    ax.set_title("Throughput vs конкурентность (1K промпт / 2K ответ)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig2_throughput.png", dpi=150)
    plt.close(fig)

    # --- fig3: TTFT p95 с лог-шкалой ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, by_c in agg.items():
        xs = [c for c in sorted(by_c) if by_c[c]["ttft_p95"] is not None]
        ys = [by_c[c]["ttft_p95"] for c in xs]
        color, label = style(model)
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=8, color=color, label=label)
        for x, y in zip(xs, ys):
            if y > 3000:
                ax.annotate(f"{y/1000:.1f}с", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, color=color)
    ax.axhline(args.budget_ms, color="red", linestyle="--", linewidth=2, label=f"бюджет {args.budget_ms:.0f} мс")
    ax.set_yscale("log")
    ax.set_xlabel("concurrency (агентов)")
    ax.set_ylabel("TTFT p95, мс (лог-шкала)")
    ax.set_title("TTFT p95 vs конкурентность (лог-шкала; пунктир — бюджет)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig3_ttft_p95.png", dpi=150)
    plt.close(fig)

    print(f"figures -> {outdir}")


if __name__ == "__main__":
    main()
