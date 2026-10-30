"""scripts/queue_limit.py — Шаг 28 плана.

Для каждой модели находит «верхнюю границу очереди»:
  max_safe_concurrency = максимальная конкурентность, при которой
      (a) TTFT p95 <= budget_ms, И
      (b) throughput ещё растёт (прирост от предыдущей точки >= min_gain_frac)

Правило роста: gain(c) = (T(c) - T(prev)) / T(prev) >= min_gain_frac (по умолчанию 5%).
Если насыщение throughput наступает РАНЬШЕ нарушения TTFT — границей считается
точка насыщения (дальше добавлять ботов бессмысленно). Если TTFT нарушается раньше —
границей считается последняя точка в бюджете.

Выход summary.csv: model, max_safe_concurrency, limiting_factor,
                   throughput_at_limit, ttft_p95_at_limit, peak_gpu_memory_pct

Использование:
  python scripts/queue_limit.py --aggregated results/aggregated.csv \
      --gpu-memory results/gpu_memory_by_concurrency.csv \
      --budget-ms 2000 --out results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FIELDS = [
    "model",
    "max_safe_concurrency",
    "limiting_factor",
    "throughput_at_limit",
    "ttft_p95_at_limit",
    "peak_gpu_memory_pct",
    "notes",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregated", required=True)
    ap.add_argument("--gpu-memory", default=None)
    ap.add_argument("--budget-ms", type=float, default=2000.0)
    ap.add_argument("--min-gain-frac", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # model -> concurrency -> усреднённые метрики по бакетам
    by_model_conc: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(Path(args.aggregated).open(encoding="utf-8")):
        by_model_conc[r["model"]][int(r["concurrency"])].append(r)

    gpu: dict[str, dict[int, float]] = defaultdict(dict)
    if args.gpu_memory and Path(args.gpu_memory).exists():
        for r in csv.DictReader(Path(args.gpu_memory).open(encoding="utf-8")):
            raw = (r.get("peak_memory_used_pct") or "").strip()
            if not raw:
                continue  # метрика не снималась — не выдумываем значение
            gpu[r["model"]][int(r["concurrency"])] = float(raw)

    summary = []
    for model, by_conc in by_model_conc.items():
        points = []
        for conc in sorted(by_conc):
            rows = by_conc[conc]
            ttft = _avg(rows, "ttft_p95_ms")
            thr = _avg(rows, "output_throughput_tok_s")
            points.append({"c": conc, "ttft": ttft, "thr": thr})

        limit = None
        factor = "none"
        prev_thr = None
        for i, p in enumerate(points):
            if p["ttft"] is None or p["thr"] is None:
                continue
            if p["ttft"] > args.budget_ms:
                factor = "ttft_budget"
                break
            if prev_thr is not None:
                gain = (p["thr"] - prev_thr) / prev_thr if prev_thr > 0 else 1.0
                if gain < args.min_gain_frac:
                    factor = "throughput_saturation"
                    limit = points[i - 1]
                    break
            limit = p
            prev_thr = p["thr"]

        if limit is None:
            summary.append(
                {f: None for f in FIELDS} | {"model": model, "notes": "no valid points"}
            )
            continue

        summary.append(
            {
                "model": model,
                "max_safe_concurrency": limit["c"],
                "limiting_factor": factor if factor != "none" else "last_point",
                "throughput_at_limit": round(limit["thr"], 2),
                "ttft_p95_at_limit": round(limit["ttft"], 2),
                "peak_gpu_memory_pct": gpu.get(model, {}).get(limit["c"]),
                "notes": "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    print(f"wrote {len(summary)} rows -> {out}")
    for r in summary:
        print(f"  {r['model']}: max_safe_c={r['max_safe_concurrency']} ({r['limiting_factor']})")


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
    return sum(vals) / len(vals) if vals else None


if __name__ == "__main__":
    main()
