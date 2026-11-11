"""scripts/validate_prompts.py — Шаг 9 плана.

Проверяет prompts.jsonl:
  1) sha256 файла совпадает с configs/run_config.yaml -> prompts.corpus_sha256
  2) каждая длина промпта (токенизатор целевой модели) попадает в свой бакет
  3) нет дубликатов id, схема записей полная
  4) печатает распределение длин по бакетам (min/median/max)

Использование:
  python scripts/validate_prompts.py \
      --tokenizer Qwen/Qwen2.5-14B-Instruct-AWQ \
      --prompts prompts/prompts.jsonl \
      --config configs/run_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REQUIRED_KEYS = {
    "id",
    "prompt",
    "prompt_len_bucket",
    "target_prompt_len",
    "max_tokens",
    "seed",
}
BUCKET_RANGES = {"short": (256, 512), "long": (1536, 2048)}


def load_expected_sha(config_path: Path) -> str | None:
    """Минимальный парсер YAML: берём только нужную строку, чтобы не тянуть pyyaml-зависимость."""
    for line in config_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("corpus_sha256:"):
            return s.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", default="prompts/prompts.jsonl")
    ap.add_argument("--config", default="configs/run_config.yaml")
    args = ap.parse_args()

    prompts_path = Path(args.prompts)
    config_path = Path(args.config)

    errors: list[str] = []

    # --- 1. sha256 ---
    actual_sha = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    expected_sha = load_expected_sha(config_path)
    print(f"sha256 actual  : {actual_sha}")
    print(f"sha256 expected: {expected_sha}")
    if expected_sha is None:
        errors.append("corpus_sha256 не найден в config")
    elif expected_sha.lower() != actual_sha.lower():
        errors.append("sha256 не совпадает с run_config.yaml")

    # --- 2. структура ---
    records = []
    for i, line in enumerate(prompts_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"строка {i}: невалидный JSON ({e})")
            continue
        if set(r) != REQUIRED_KEYS:
            errors.append(f"строка {i} ({r.get('id')}): схема полей {sorted(set(r) ^ REQUIRED_KEYS)}")
        records.append(r)

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        dups = sorted({x for x in ids if ids.count(x) > 1})
        errors.append(f"дубликаты id: {dups[:10]}")

    # --- 3. длины по токенизатору целевой модели ---
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    by_bucket: dict[str, list[int]] = {b: [] for b in BUCKET_RANGES}
    for r in records:
        bucket = r["prompt_len_bucket"]
        if bucket not in BUCKET_RANGES:
            errors.append(f"{r['id']}: неизвестный бакет '{bucket}'")
            continue
        n_tok = len(tok.encode(r["prompt"], add_special_tokens=False))
        lo, hi = BUCKET_RANGES[bucket]
        if not (lo <= n_tok <= hi):
            errors.append(f"{r['id']}: {n_tok} ток. вне [{lo},{hi}]")
        by_bucket[bucket].append(n_tok)

    print("\n--- распределение длин (токены, токенизатор целевой модели) ---")
    for bucket, lens in by_bucket.items():
        if not lens:
            errors.append(f"бакет {bucket}: пусто")
            continue
        print(
            f"{bucket:5s}: n={len(lens):3d} min={min(lens):4d} "
            f"median={int(statistics.median(lens)):4d} max={max(lens):4d}"
        )

    print()
    if errors:
        print(f"FAIL: {len(errors)} ошибок")
        for e in errors[:30]:
            print("  -", e)
        return 1

    print(f"OK: {len(records)} промптов прошли валидацию")
    return 0


if __name__ == "__main__":
    sys.exit(main())
