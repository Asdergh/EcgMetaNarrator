import csv
import json
import statistics
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct-AWQ")

print("=== КОРПУС prompts_matrix_1k.jsonl ===")
recs = [json.loads(l) for l in (ROOT / "prompts/prompts_matrix_1k.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
lens = [len(tok.encode(r["prompt"], add_special_tokens=False)) for r in recs]
print(f"промптов: {len(recs)}")
print(f"длина промпта (токены): min={min(lens)} median={int(statistics.median(lens))} max={max(lens)}")

print("\n=== OUTPUT TOKENS по прогонам ===")
for path in [
    "results/matrix/qwen25-1.5b-mx/c2/per_request.csv",
    "results/matrix/qwen25-1.5b-mx/c10/per_request.csv",
    "results/matrix/qwen25-1.5b-mx/c15/per_request.csv",
    "results/matrix/rusmind-malysh-mx/c2/per_request.csv",
    "results/matrix/rusmind-malysh-mx/c10_mem/per_request.csv",
]:
    p = ROOT / path
    if not p.exists():
        print(f"MISSING {path}")
        continue
    rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r["status"] == "ok" and r["is_warmup"] == "False"]
    if not rows:
        print(f"EMPTY {path}")
        continue
    t = [int(r["output_tokens"]) for r in rows]
    c = rows[0]["concurrency"]
    model = path.split("/")[-3]
    print(f"{model} c={c}: n={len(rows)} output_tokens min={min(t)} med={int(statistics.median(t))} max={max(t)}")
