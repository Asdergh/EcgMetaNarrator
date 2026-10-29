"""Загрузчики, работающие только с метаданными, для поддерживаемых архивов ЭКГ."""

import ast
import csv
import re
from pathlib import Path
from typing import Any, Iterable

from .mappings import map_binary_diagnosis, map_challenge2020_tokens
from .readers import _header_metadata
from .registry import register_labels, register_loader


def _limit(items: Iterable[dict[str, Any]], limit: int | None):
    for index, item in enumerate(items):
        if limit is not None and index >= limit:
            return
        yield item


def _record_meta(record: Path, label_raw: Any, **extra: Any) -> dict[str, Any]:
    metadata = _header_metadata(record.with_suffix(".hea")) if record.with_suffix(".hea").exists() else {
        "n_sig": None,
        "fs": None,
    }
    metadata.update({"record_id": record.name, "label_raw": label_raw, **extra})
    return metadata


def _header_value(path: Path, key: str) -> str | None:
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if line.startswith("#") and line[1:].lstrip().startswith(key):
            return line.split(":", 1)[-1].strip()
    return None


def _normalise_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _record_path(path: Path) -> Path:
    if path.suffix.lower() in {".dat", ".hea", ".mat"}:
        return path.with_suffix("")
    return path


def _has_signal_pair(record: Path, signal_suffix: str) -> bool:
    return record.with_suffix(".hea").is_file() and record.with_suffix(signal_suffix).is_file()


PTBXL_LABELS = {
    "NORM": {"description": "Normal ECG", "label": 0},
    "MI": {"description": "Myocardial infarction", "label": 1},
    "STTC": {"description": "ST/T change", "label": 2},
    "CD": {"description": "Conduction disturbance", "label": 3},
    "HYP": {"description": "Hypertrophy", "label": 4},
}
PTBDB_LABELS = {
    "NORMAL": {"description": "Normal ECG", "label": 0},
    "ABNORMAL": {"description": "Abnormal ECG", "label": 1},
}
CHALLENGE2020_LABELS = {
    "NORM": {"description": "Normal ECG", "label": 0},
    "AF": {"description": "Atrial fibrillation", "label": 1},
    "IAVB": {"description": "First-degree AV block", "label": 2},
    "LBBB": {"description": "Left bundle branch block", "label": 3},
    "RBBB": {"description": "Right bundle branch block", "label": 4},
    "PAC": {"description": "Premature atrial contraction", "label": 5},
    "PVC": {"description": "Premature ventricular contraction", "label": 6},
    "STD": {"description": "ST depression", "label": 7},
    "STE": {"description": "ST elevation", "label": 8},
}
CHALLENGE2018_LABELS = {
    "NO_APNEA": {"description": "No sleep apnea", "label": 0},
    "APNEA": {"description": "Sleep apnea", "label": 1},
}
LUDB_LABELS = {
    "NORMAL": {"description": "Normal ECG", "label": 0},
    "ABNORMAL": {"description": "Abnormal ECG", "label": 1},
}

for _name, _labels in {
    "ptbxl": PTBXL_LABELS,
    "ptbdb": PTBDB_LABELS,
    "challenge2020": CHALLENGE2020_LABELS,
    "challenge2018": CHALLENGE2018_LABELS,
    "ludb": LUDB_LABELS,
}.items():
    register_labels(_name, _labels)


@register_loader("ptbxl")
def load_ptbxl(root: str | Path, split: str | None = None, limit: int | None = None, **_: Any):
    root_path = Path(root)
    csv_path = root_path / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"PTB-XL metadata file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        yielded = 0
        for row in rows:
            filename = row.get("filename_hr") or row.get("filename_lr")
            if not filename:
                continue
            if split and row.get("strat_fold") and int(row["strat_fold"]) != int(split):
                continue
            raw = row.get("scp_codes", "{}")
            try:
                codes = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                codes = {}
            label = next((key for key in ("NORM", "MI", "STTC", "CD", "HYP") if key in codes), None)
            if label is None:
                continue
            record = _record_path(root_path / filename)
            if not _has_signal_pair(record, ".dat"):
                continue
            yield {"file_path": str(record), "label": label, "meta": _record_meta(
                record, codes, ecg_id=row.get("ecg_id"), age=row.get("age"),
                sex=row.get("sex"), strat_fold=row.get("strat_fold"),
            )}
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _load_header_labels(root: Path, labels: set[str], limit: int | None, **_: Any):
    yielded = 0
    for header in sorted(root.rglob("*.hea")):
        raw = _header_value(header, "Dx")
        if raw is None:
            raw = _header_value(header, "Diagnosis")
        token = _normalise_token(raw or "")
        label = map_binary_diagnosis(token)
        if label not in labels:
            continue
        record = header.with_suffix("")
        if not _has_signal_pair(record, ".dat"):
            continue
        yield {"file_path": str(record), "label": label, "meta": _record_meta(record, raw)}
        yielded += 1
        if limit is not None and yielded >= limit:
            return


@register_loader("ptbdb")
def load_ptbdb(root: str | Path, limit: int | None = None, **kwargs: Any):
    yield from _load_header_labels(Path(root), set(PTBDB_LABELS), limit, **kwargs)


@register_loader("ludb")
def load_ludb(root: str | Path, limit: int | None = None, **kwargs: Any):
    yield from _load_header_labels(Path(root), set(LUDB_LABELS), limit, **kwargs)


@register_loader("challenge2020")
def load_challenge2020(root: str | Path, limit: int | None = None, **_: Any):
    yielded = 0
    for header in sorted(Path(root).rglob("*.hea")):
        raw = _header_value(header, "Dx")
        tokens = {_normalise_token(value) for value in (raw or "").split(",")}
        label = map_challenge2020_tokens(tokens)
        if label is None:
            continue
        record = header.with_suffix("")
        if not _has_signal_pair(record, ".mat"):
            continue
        yield {"file_path": str(record), "label": label, "meta": _record_meta(record, raw)}
        yielded += 1
        if limit is not None and yielded >= limit:
            return


@register_loader("challenge2018")
def load_challenge2018(root: str | Path, limit: int | None = None, **_: Any):
    yielded = 0
    signals = sorted(Path(root).rglob("*.mat")) + sorted(Path(root).rglob("*.dat"))
    for signal in signals:
        record = signal.with_suffix("")
        if not record.with_suffix(".hea").is_file():
            continue
        annotation = record.with_suffix(".apn")
        if not annotation.is_file():
            continue
        raw = _read_apnea_annotation(annotation)
        if not raw:
            continue
        label = "APNEA" if raw.count("A") > raw.count("N") else "NO_APNEA"
        yield {"file_path": str(record), "label": label, "meta": _record_meta(record, raw)}
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _read_apnea_annotation(path: Path) -> list[str] | None:
    """Return apnea annotation symbols without reading the ECG signal."""
    try:
        import wfdb
        annotation = wfdb.rdann(str(path.with_suffix("")), "apn")
    except ImportError:
        text = path.read_text(encoding="ascii", errors="ignore")
        return re.findall(r"\b[AN]\b", text.upper())
    except (OSError, ValueError):
        return None
    return [str(symbol).upper() for symbol in annotation.symbol if str(symbol).upper() in {"A", "N"}]
