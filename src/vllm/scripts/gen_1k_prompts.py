import json
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct-AWQ")

TOPICS = [
    ("архитектура современных LLM", "разбери transformer, attention, KV-cache, квантизацию, батчинг, vLLM, PagedAttention, speculative decoding"),
    ("проектирование распределённых баз данных", "разбери шардирование, репликацию, консенсус (Raft/Paxos), транзакции, CAP, согласованность, отказоустойчивость"),
    ("оптимизация Python для высоких нагрузок", "разбери GIL, asyncio, multiprocessing, Cython, numba, профилирование, память, пулы процессов"),
    ("устройство Kubernetes изнутри", "разбери control plane, etcd, scheduler, kubelet, CNI, CSI, operators, HPA, сетевую модель"),
    ("системы рекомендаций в продакшене", "разбери candidate generation, ranking, feature store, online learning, A/B, метрики, холодный старт"),
    ("безопасность веб-приложений", "разбери OWASP Top 10, SQL-инъекции, XSS, CSRF, SSRF, аутентификацию, JWT, OAuth2, rate limiting"),
    ("компиляторы и оптимизация кода", "разбери SSA, register allocation, loop unrolling, vectorization, JIT, LLVM pipeline, профилирование"),
    ("финансовое моделирование рисков", "разбери VaR, CVaR, Monte Carlo, стресс-тесты, корреляции, копулы, backtesting, регуляторные требования"),
]

HEADER = "Напиши подробный технический разбор темы «{topic}» на русском языке. Требования: не менее 1500 слов, структурируй ответ разделами с подзаголовками, в каждом разделе приведи конкретные примеры, числа, названия инструментов и версий. Не сокращай, не используй общие фразы, раскрывай каждый пункт подробно. В конце сделай сводную таблицу сравнения подходов и обоснованный вывод."

FILLER = "Дополнительный контекст: {spec}. При описании каждого пункта укажи, какие проблемы он решает, какие ограничения имеет, и как его применяют на практике в крупных компаниях. Приведи не менее двух примеров из реальных проектов и опиши типичные ошибки, которые допускают инженеры при внедрении. Сравни альтернативные решения по критериям производительности, сложности эксплуатации и стоимости. Укажи, какие метрики использовать для оценки эффективности и как их собирать в production-среде. Опиши, как масштабировать решение при росте нагрузки в 10 и 100 раз, какие узкие места появятся и как их обходить. Затрони вопросы мониторинга, алертинга и отладки в эксплуатации."

def build(target_tokens: int, topic: str, spec: str) -> str:
    parts = [HEADER.format(topic=topic)]
    i = 0
    while True:
        text = " ".join(parts)
        n = len(tok.encode(text, add_special_tokens=False))
        if n >= target_tokens:
            break
        parts.append(FILLER.format(spec=spec))
        parts.append(f"Дополнение {i+1}: приведи конкретные цифры, бенчмарки и ссылки на документацию.")
        i += 1
    ids = tok.encode(" ".join(parts), add_special_tokens=False)[:target_tokens]
    return tok.decode(ids, skip_special_tokens=True)

out = Path("prompts/prompts_1k_longgen.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for i, (topic, spec) in enumerate(TOPICS, 1):
        text = build(1000, topic, spec)
        n = len(tok.encode(text, add_special_tokens=False))
        rec = {"id": f"k1-longgen-{i:03d}", "prompt": text, "prompt_len_bucket": "k1", "target_prompt_len": n, "max_tokens": 2000, "seed": 42}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{rec['id']}: prompt_tokens={n}")
import hashlib
print("sha256:", hashlib.sha256(out.read_bytes()).hexdigest())
