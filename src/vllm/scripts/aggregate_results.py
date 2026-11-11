"""scripts/aggregate_results.py — Шаг 22 плана.

Читает per_request.csv (возможно несколько файлов) и считает по каждой комбинации
(run_id, model, bucket, concurrency):
  - output_throughput_tok_s = sum(output_tokens) / wall_duration
  - request_throughput_rps  = count / wall_duration
  - TTFT p50/p95/p99, E2E p50/p95/p99, TPOT median
  - error_rate, warmup_excluded, server_metrics_used_count

Правила:
  - is_warmup=True  -> исключается из агрегации (но считается отдельно)
  - status=error    -> не идёт в latency/throughput, но идёт в error_rate

Использование:
  python scripts/aggregate_results.py \
      --inputs results/a/c10/per_request.csv results/a/c20/per_request.csv \
      --model qwen25-14b-awq --out results/a/aggregated.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

OUT_FIELDS = [
    "run_id",
    "model",
    "bucket",
    "concurrency",
    "n_ok",
    "n_error",
    "n_warmup",
    "error_rate",
    "wall_duration_s",
    "output_throughput_tok_s",
    "request_throughput_rps",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "tpot_median_ms",
    "output_tokens_mean",
    "server_metrics_used",
    "wall_source",
]


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 3)
    # linear interpolation между соседними порядковыми статистиками
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    frac = pos - lo
    return round(s[lo] * (1 - frac) + s[hi] * frac, 3)


def fnum(v: str | None) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups: dict[tuple, dict] = defaultdict(
        lambda: {
            "ok": [],
            "err": 0,
            "warmup": 0,
            "server_used": 0,
            "run_id": None,
        }
    )

    for path in args.inputs:
        with Path(path).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                conc = int(row["concurrency"])
                bucket = row["bucket"]
                run_id = row["run_id"]
                key = (run_id, bucket, conc)
                g = groups[key]
                g["run_id"] = run_id

                if row["is_warmup"] in ("True", "true", "1"):
                    g["warmup"] += 1
                    continue
                if row["status"] != "ok":
                    g["err"] += 1
                    continue

                g["ok"].append(
                    {
                        "ttft": fnum(row["ttft_client_ms"]),
                        "e2e": fnum(row["e2e_ms"]),
                        "tpot": fnum(row["tpot_ms"]),
                        "tok": fnum(row["output_tokens"]) or 0.0,
                        "start": fnum(row.get("started_at")),
                        "finish": fnum(row.get("finished_at")),
                    }
                )
                if row["server_metrics_used"] in ("True", "true", "1"):
                    g["server_used"] += 1

    out_rows = []
    for (run_id, bucket, conc), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        ok = g["ok"]
        ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
        e2es = [r["e2e"] for r in ok if r["e2e"] is not None]
        tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
        toks = [r["tok"] for r in ok]

        # throughput по реальному окну нагрузки: max(finished_at) - min(started_at).
        # Fallback на max(e2e) — только если таймстемпов нет (старые CSV); даёт НИЖНЮЮ оценку.
        starts = [r["start"] for r in ok if r["start"] is not None]
        finishes = [r["finish"] for r in ok if r["finish"] is not None]
        if starts and finishes and max(finishes) > min(starts):
            wall = max(finishes) - min(starts)
            wall_source = "timestamps"
        else:
            wall = max(e2es) / 1000.0 if e2es else None
            wall_source = "fallback_max_e2e"
        total_tok = sum(toks)
        out_rows.append(
            {
                "run_id": run_id,
                "model": args.model,
                "bucket": bucket,
                "concurrency": conc,
                "n_ok": len(ok),
                "n_error": g["err"],
                "n_warmup": g["warmup"],
                "error_rate": round(g["err"] / max(1, g["err"] + len(ok)), 4),
                "wall_duration_s": round(wall, 3) if wall else None,
                "output_throughput_tok_s": round(total_tok / wall, 2) if wall else None,
                "request_throughput_rps": round(len(ok) / wall, 4) if wall else None,
                "ttft_p50_ms": pct(ttfts, 0.50),
                "ttft_p95_ms": pct(ttfts, 0.95),
                "ttft_p99_ms": pct(ttfts, 0.99),
                "e2e_p50_ms": pct(e2es, 0.50),
                "e2e_p95_ms": pct(e2es, 0.95),
                "e2e_p99_ms": pct(e2es, 0.99),
                "tpot_median_ms": round(statistics.median(tpots), 3) if tpots else None,
                "output_tokens_mean": round(statistics.fmean(toks), 1) if toks else None,
                "server_metrics_used": g["server_used"],
                "wall_source": wall_source,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"wrote {len(out_rows)} aggregated rows -> {out}")
    for r in out_rows:
        print(
            f"  {r['bucket']:5s} c={r['concurrency']:>3}  ttft_p95={r['ttft_p95_ms']}  "
            f"tok/s={r['output_throughput_tok_s']}  ok={r['n_ok']} err={r['n_error']}"
        )


if __name__ == "__main__":
    main()
