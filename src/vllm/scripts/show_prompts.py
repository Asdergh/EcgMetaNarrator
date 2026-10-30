import json
from pathlib import Path

for path in ["prompts/prompts.jsonl", "prompts/prompts_long_gen.jsonl"]:
    p = Path(path)
    if not p.exists():
        continue
    print(f"=== {path} ===")
    recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    short = [r for r in recs if r["prompt_len_bucket"] == "short"][:3]
    longg = [r for r in recs if r["prompt_len_bucket"] == "long"][:3]
    for label, subset in (("short", short), ("long", longg)):
        for r in subset:
            text = r["prompt"].replace("\n", " ")
            print(f"--- {r['id']} [{label}] target_len={r.get('target_prompt_len')}")
            print(text[:280] + ("..." if len(text) > 280 else ""))
        print()
