"""scripts/combine_aggregates.py — объединение aggregated.csv от нескольких моделей.

plot_results.py и queue_limit.py ожидают один CSV с колонкой model.
Этот скрипт склеивает агрегаты по моделям, сохраняя заголовок один раз.

Использование:
  python scripts/combine_aggregates.py \
      --inputs results/qwen25-3b-awq/aggregated.csv results/qwen25-1.5b-awq/aggregated.csv \
      --out results/aggregated_all.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    fields: list[str] | None = None
    for path in args.inputs:
        with Path(path).open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = reader.fieldnames or []
            for r in reader:
                rows.append(r)

    if not rows or fields is None:
        raise SystemExit("no input rows")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    models = sorted({r["model"] for r in rows})
    print(f"combined {len(rows)} rows from {len(args.inputs)} files -> {out}")
    print(f"models: {models}")


if __name__ == "__main__":
    main()
