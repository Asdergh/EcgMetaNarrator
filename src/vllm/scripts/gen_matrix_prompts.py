import json
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct-AWQ")

TOPICS = [
    ("эволюция архитектур нейросетей для обработки текста", "от RNN и LSTM через attention к трансформерам, включая BERT, GPT, T5, и современные MoE"),
    ("инженерия данных в ML-пайплайнах", "сбор, очистка, разметка, версионирование, feature store, качество данных, дрейф"),
    ("оптимизация запросов в аналитических СУБД", "индексы, планы выполнения, партиционирование, materialized views, columnar storage"),
    ("архитектура микросервисов для высоких нагрузок", "API gateway, service mesh, circuit breaker, saga, event sourcing, CQRS"),
    ("финансовый учёт в крупных компаниях", "двойная запись, консолидация, IFRS, аудит, внутренний контроль, автоматизация"),
    ("разработка мобильных приложений под iOS и Android", "нативные и кросс-платформенные подходы, производительность, память, публикация"),
    ("кибербезопасность корпоративных сетей", "сегментация, zero trust, SIEM, SOAR, threat hunting, incident response"),
    ("управление продуктовой разработкой", "roadmap, OKR, discovery, delivery, метрики, работа с гипотезами"),
    ("облачные платформы и инфраструктура как код", "Terraform, Ansible, Kubernetes, GitOps, secrets management, cost optimization"),
    ("машинное обучение в реальном времени", "online learning, streaming features, low-latency inference, model monitoring"),
    ("правовые аспекты обработки персональных данных", "GDPR, 152-ФЗ, согласия, анонимизация, хранение, трансграничная передача"),
    ("архитектура систем рекомендаций для маркетплейсов", "персонализация, ranking, diversity, exploration, cold start, метрики"),
]

HEADER = (
    "Напиши подробный разбор темы «{topic}» на русском языке. "
    "Требования: не менее 1500 слов, структурируй ответ разделами с подзаголовками, "
    "в каждом разделе приведи конкретные примеры, числа, названия инструментов и версий. "
    "Не сокращай, не используй общие фразы, раскрывай каждый пункт подробно. "
    "В конце сделай сводную таблицу сравнения подходов и обоснованный вывод."
)

FILLER = (
    "Дополнительный контекст: {spec}. При описании каждого пункта укажи, какие проблемы он решает, "
    "какие ограничения имеет, и как его применяют на практике в крупных компаниях. "
    "Приведи не менее двух примеров из реальных проектов и опиши типичные ошибки, которые допускают инженеры при внедрении. "
    "Сравни альтернативные решения по критериям производительности, сложности эксплуатации и стоимости. "
    "Укажи, какие метрики использовать для оценки эффективности и как их собирать в production-среде. "
    "Опиши, как масштабировать решение при росте нагрузки в 10 и 100 раз, какие узкие места появятся и как их обходить. "
    "Затрони вопросы мониторинга, алертинга и отладки в эксплуатации."
)

def build(target_tokens, topic, spec):
    parts = [HEADER.format(topic=topic)]
    i = 0
    while True:
        text = " ".join(parts)
        if len(tok.encode(text, add_special_tokens=False)) >= target_tokens:
            break
        parts.append(FILLER.format(spec=spec))
        parts.append(f"Дополнение {i+1}: приведи конкретные цифры, бенчмарки и ссылки на документацию.")
        i += 1
    ids = tok.encode(" ".join(parts), add_special_tokens=False)[:target_tokens]
    return tok.decode(ids, skip_special_tokens=True)

out = Path("prompts/prompts_matrix_1k.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for i, (topic, spec) in enumerate(TOPICS, 1):
        text = build(1000, topic, spec)
        n = len(tok.encode(text, add_special_tokens=False))
        rec = {
            "id": f"mx-{i:03d}",
            "prompt": text,
            "prompt_len_bucket": "mx",
            "target_prompt_len": n,
            "max_tokens": 2000,
            "seed": 123,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{rec['id']}: prompt_tokens={n}")
import hashlib
print("sha256:", hashlib.sha256(out.read_bytes()).hexdigest())
