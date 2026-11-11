"""scripts/aggregate_gpu.py — Шаг 19 плана.

Агрегирует gpu_timeseries.csv (НЕ только конечную точку, а весь ряд):
  - peak_memory_used_mb, peak_memory_used_pct
  - mean/median utilization_gpu_pct
  - доля сэмплов, где утилизация > 95% (признак насыщения)
  - момент первого достижения порога memory_used_pct >= gpu_memory_utilization*100
  - coverage: доля времени, где GPU реально нагружен (>50% util)

Использование:
  python scripts/aggregate_gpu.py \
      --timeseries results/smoke/gpu_timeseries.csv \
      --gpu-mem-util 0.90 \
      --out results/smoke/gpu_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def as_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--saturation-util", type=float, default=95.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.timeseries).open(encoding="utf-8")))
    if not rows:
        raise SystemExit("empty timeseries")

    mem_used = [x for x in (as_float(r["memory_used_mb"]) for r in rows) if x is not None]
    mem_pct = [x for x in (as_float(r["memory_used_pct"]) for r in rows) if x is not None]
    util = [x for x in (as_float(r["utilization_gpu_pct"]) for r in rows) if x is not None]

    if not mem_used or not util:
        raise SystemExit("timeseries не содержит memory_used_mb / utilization_gpu_pct")

    total_mb = as_float(rows[0].get("memory_total_mb"))
    threshold_pct = args.gpu_mem_util * 100.0

    first_hit_ts = None
    for r in rows:
        p = as_float(r.get("memory_used_pct"))
        if p is not None and p >= threshold_pct:
            first_hit_ts = r.get("timestamp_iso")
            break

    saturated = sum(1 for u in util if u >= args.saturation_util)
    active = sum(1 for u in util if u >= 50.0)

    summary = {
        "run_id": rows[0].get("run_id"),
        "concurrency": rows[0].get("concurrency"),
        "samples": len(rows),
        "window_start": rows[0].get("timestamp_iso"),
        "window_end": rows[-1].get("timestamp_iso"),
        "memory_total_mb": total_mb,
        "peak_memory_used_mb": max(mem_used),
        "peak_memory_used_pct": max(mem_pct) if mem_pct else None,
        "mean_memory_used_mb": round(statistics.fmean(mem_used), 1),
        "memory_threshold_pct": threshold_pct,
        "threshold_reached_at": first_hit_ts,
        "threshold_reached": first_hit_ts is not None,
        "util_mean_pct": round(statistics.fmean(util), 2),
        "util_median_pct": round(statistics.median(util), 2),
        "util_p95_pct": round(sorted(util)[min(len(util) - 1, int(0.95 * len(util)))], 2),
        "saturated_fraction": round(saturated / len(util), 4),
        "active_fraction_ge50": round(active / len(util), 4),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
