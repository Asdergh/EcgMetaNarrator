"""scripts/generate_prompts.py — Шаг 7 плана.

Генерирует prompts/prompts.jsonl с промптами двух бакетов длины:
  short: ~256-512 токенов
  long : ~1536-2048 токенов

Поля записи: id, prompt, prompt_len_bucket, target_prompt_len, max_tokens, seed.
Длины считаются токенизатором ЦЕЛЕВОЙ модели (не tiktoken), чтобы шаг 9 (валидатор)
и корпус говорили на одном языке.

Использование:
  python scripts/generate_prompts.py \
      --tokenizer Qwen/Qwen2.5-14B-Instruct-AWQ \
      --n-short 60 --n-long 60 --out prompts/prompts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SHORT_RANGE = (256, 512)
LONG_RANGE = (1536, 2048)

# Содержательные «затравки»: темы разные, чтобы корпус не был однообразным
# (однообразие искажает кэш префиксов и TTFT).
TOPICS = [
    "объясни разницу между TCP и UDP для разработчика",
    "опиши, как работает индексация в PostgreSQL",
    "сравни батчевую и стриминговую обработку данных",
    "расскажи про устройство attention в трансформерах",
    "объясни, что такое идемпотентность HTTP-методов",
    "опиши жизненный цикл запроса в веб-фреймворке",
    "сравни AWQ и GPTQ квантизацию LLM",
    "объясни, как устроен CUDA-граф и зачем он нужен",
    "опиши стратегии кэширования в распределённых системах",
    "расскажи про backpressure в асинхронных пайплайнах",
]

FILLER = (
    "Приведи конкретные примеры, числа и граничные случаи. "
    "Структурируй ответ: сначала краткое резюме, затем детали, в конце — вывод. "
    "Не повторяйся, избегай общих фраз; где уместно — сравни варианты в таблице. "
)


def build_prompt(rng: random.Random, target_len: int, tokenizer) -> str:
    """Набирает текст из затравки + варьирующихся блоков, пока длина не достигнет цели."""
    parts = [rng.choice(TOPICS) + ".\n"]
    while True:
        candidate = "\n".join(parts)
        n_tok = len(tokenizer.encode(candidate, add_special_tokens=False))
        if n_tok >= target_len:
            break
        # добавляем уникальный по содержанию блок (номер + случайный подпункт)
        parts.append(f"Пункт {len(parts)}. {FILLER} Уточнение #{rng.randint(1, 10**6)}.")
    return "\n".join(parts)


def truncate_to_range(text: str, target: int, tokenizer) -> str:
    """Обрезает по токенам ровно до target и декодирует обратно."""
    ids = tokenizer.encode(text, add_special_tokens=False)[:target]
    return tokenizer.decode(ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF id токенизатора целевой модели")
    ap.add_argument("--n-short", type=int, default=60)
    ap.add_argument("--n-long", type=int, default=60)
    ap.add_argument("--out", default="prompts/prompts.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for bucket, n, (lo, hi) in (
        ("short", args.n_short, SHORT_RANGE),
        ("long", args.n_long, LONG_RANGE),
    ):
        for i in range(n):
            target = rng.randint(lo, hi)
            text = build_prompt(rng, target, tok)
            text = truncate_to_range(text, target, tok)
            real_len = len(tok.encode(text, add_special_tokens=False))
            if not (lo <= real_len <= hi):
                raise RuntimeError(
                    f"bucket {bucket}: длина {real_len} вне [{lo},{hi}] — генератор нестабилен"
                )
            records.append(
                {
                    "id": f"{bucket}-{i:03d}",
                    "prompt": text,
                    "prompt_len_bucket": bucket,
                    "target_prompt_len": target,
                    "max_tokens": 256,
                    "seed": args.seed,
                }
            )

    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {len(records)} prompts -> {out}")
    print(f"sha256: {digest}")
    print("Внесите sha256 в configs/run_config.yaml -> prompts.corpus_sha256")


if __name__ == "__main__":
    main()
