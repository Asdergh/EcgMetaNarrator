"""scripts/inference_manager.py — Шаги 11-13 плана.

Пул async-ботов с asyncio.Semaphore(N), сессия = N ботов × repeats повторов.
Основной замер TTFT — КЛИЕНТСКИЙ stream (шаг 12): t0 -> первый непустой delta.content -> t1.
Серверные per-request метрики (metrics.time_to_first_token_ms) — только fallback (шаг 13).
Warm-up (шаг 15): первые warmup_requests запросов помечаются is_warmup=True и в агрегацию не идут.

Использование:
  python scripts/inference_manager.py \
      --base-url http://127.0.0.1:8000/v1 \
      --model qwen25-14b-awq \
      --prompts prompts/prompts.jsonl \
      --concurrency 1 --repeats 1 --max-tokens 128 \
      --run-id smoke --out results/smoke/per_request.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import httpx

FIELDS = [
    "run_id",
    "prompt_id",
    "bucket",
    "concurrency",
    "slot",
    "repeat",
    "prompt_len_chars",
    "started_at",
    "finished_at",
    "ttft_client_ms",
    "e2e_ms",
    "output_tokens",
    "tpot_ms",
    "ttft_server_ms",
    "server_metrics_used",
    "is_warmup",
    "status",
    "error",
]


def load_prompts(path: Path, buckets: list[str] | None = None) -> list[dict]:
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if buckets:
        allowed = {b.strip() for b in buckets if b.strip()}
        recs = [r for r in recs if r["prompt_len_bucket"] in allowed]
    return recs


def parse_server_ttft_ms(body: dict) -> float | None:
    """Шаг 13: fallback-парсер серверных per-request метрик.

    Корректно обрабатывает: отсутствие metrics, null, и n>1 (берём первый choice).
    """
    choices = body.get("choices") or []
    if not choices:
        return None
    metrics = choices[0].get("metrics") or {}
    val = metrics.get("time_to_first_token_ms")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    rec: dict,
    max_tokens: int,
) -> dict:
    """Один запрос со stream-замером TTFT (основной способ, шаг 12)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": rec["prompt"]}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t0 = time.perf_counter()
    started_at = time.time()
    ttft_client_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens = 0
    server_ttft_ms: float | None = None
    server_body_for_fallback: dict | None = None

    try:
        async with client.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "replace")[:500]
                return _err_row(rec, f"HTTP {resp.status_code}: {text}")

            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")

                if content and ttft_client_ms is None:
                    ttft_client_ms = (time.perf_counter() - t0) * 1000.0

                if content:
                    output_tokens += 1

                # сервер может прислать metrics в последнем чанке с usage/choices
                m = parse_server_ttft_ms(chunk)
                if m is not None:
                    server_ttft_ms = m
                server_body_for_fallback = chunk

            e2e_ms = (time.perf_counter() - t0) * 1000.0

    except Exception as e:  # noqa: BLE001 — хотим записать любую сетевую ошибку в CSV
        return _err_row(rec, f"{type(e).__name__}: {e}")

    if ttft_client_ms is None:
        return _err_row(rec, "no content chunk received (ttft undefined)")

    tpot_ms = None
    if output_tokens > 1 and e2e_ms is not None:
        tpot_ms = (e2e_ms - ttft_client_ms) / (output_tokens - 1)

    return {
        "started_at": started_at,
        "finished_at": started_at + (e2e_ms / 1000.0 if e2e_ms is not None else 0.0),
        "ttft_client_ms": round(ttft_client_ms, 3),
        "e2e_ms": round(e2e_ms, 3) if e2e_ms is not None else None,
        "output_tokens": output_tokens,
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "ttft_server_ms": round(server_ttft_ms, 3) if server_ttft_ms is not None else None,
        "server_metrics_used": server_ttft_ms is not None,
        "status": "ok",
        "error": "",
    }


def _err_row(rec: dict, err: str) -> dict:
    now = time.time()
    return {
        "started_at": now,
        "finished_at": now,
        "ttft_client_ms": None,
        "e2e_ms": None,
        "output_tokens": 0,
        "tpot_ms": None,
        "ttft_server_ms": None,
        "server_metrics_used": False,
        "status": "error",
        "error": err,
    }


async def run_session(args) -> list[dict]:
    prompts = load_prompts(Path(args.prompts), args.buckets)
    if not prompts:
        raise SystemExit(f"empty prompts selection (buckets={args.buckets})")
    print(f"loaded {len(prompts)} prompts (buckets={args.buckets or 'all'})", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    total = args.concurrency * args.repeats
    counter = {"n": 0}

    limits = httpx.Limits(max_connections=args.concurrency + 4, max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(connect=10.0, read=args.request_timeout_s, write=30.0, pool=10.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # прогреваем соединение и CUDA-графы ОТДЕЛЬНЫМ запросом, не входящим в агрегацию
        for _ in range(args.warmup_requests):
            rec = prompts[0]
            res = await one_request(client, args.base_url, args.model, rec, min(args.max_tokens, 16))
            rows.append(_row(args, rec, slot=0, repeat=-1, res=res, is_warmup=True))

        async def worker(slot: int, repeat: int, rec: dict) -> None:
            async with sem:
                res = await one_request(client, args.base_url, args.model, rec, args.max_tokens)
                rows.append(_row(args, rec, slot=slot, repeat=repeat, res=res, is_warmup=False))
                counter["n"] += 1
                if counter["n"] % 10 == 0 or counter["n"] == total:
                    print(f"  progress {counter['n']}/{total}", file=sys.stderr)

        tasks = []
        for repeat in range(args.repeats):
            for slot in range(args.concurrency):
                rec = prompts[(slot + repeat * args.concurrency) % len(prompts)]
                tasks.append(asyncio.create_task(worker(slot, repeat, rec)))
        await asyncio.gather(*tasks)

    return rows


def _row(args, rec: dict, slot: int, repeat: int, res: dict, is_warmup: bool) -> dict:
    return {
        "run_id": args.run_id,
        "prompt_id": rec["id"],
        "bucket": rec["prompt_len_bucket"],
        "concurrency": args.concurrency,
        "slot": slot,
        "repeat": repeat,
        "prompt_len_chars": len(rec["prompt"]),
        "started_at": res["started_at"],
        "finished_at": res["finished_at"],
        "ttft_client_ms": res["ttft_client_ms"],
        "e2e_ms": res["e2e_ms"],
        "output_tokens": res["output_tokens"],
        "tpot_ms": res["tpot_ms"],
        "ttft_server_ms": res["ttft_server_ms"],
        "server_metrics_used": res["server_metrics_used"],
        "is_warmup": is_warmup,
        "status": res["status"],
        "error": res["error"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="prompts/prompts.jsonl")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup-requests", type=int, default=1)
    ap.add_argument("--request-timeout-s", type=float, default=300.0)
    ap.add_argument("--buckets", nargs="*", default=None,
                   help="ограничить бакеты, например: --buckets short (для 8 ГБ VRAM)")
    ap.add_argument("--run-id", default="manual")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = asyncio.run(run_session(args))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = sum(1 for r in rows if r["status"] == "ok" and not r["is_warmup"])
    err = sum(1 for r in rows if r["status"] == "error" and not r["is_warmup"])
    used_server = sum(1 for r in rows if r["server_metrics_used"])
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"ok={ok} error={err} warmup={sum(1 for r in rows if r['is_warmup'])}")
    print(f"server_metrics_available_in={used_server} rows (fallback channel)")


if __name__ == "__main__":
    main()
