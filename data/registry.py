"""Датасеты: метки и загрузчики."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

LABELS: dict[str, dict[str, dict[str, Any]]] = {}
LOADERS: dict[str, Callable[..., Iterator[dict[str, Any]]]] = {}


def register_labels(name: str, labels: dict[str, dict[str, Any]]) -> None:
    if not name or not isinstance(labels, dict) or not labels:
        raise ValueError("Dataset name and labels must be non-empty")
    seen: set[int] = set()
    for acronym, spec in labels.items():
        if not acronym or not isinstance(spec, dict):
            raise ValueError(f"Invalid label specification for {acronym!r}")
        label = spec.get("label")
        description = spec.get("description")
        if not isinstance(label, int) or isinstance(label, bool):
            raise ValueError(f"Label index for {acronym!r} must be an integer")
        if label in seen:
            raise ValueError(f"Duplicate label index: {label}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Description for {acronym!r} must be non-empty")
        seen.add(label)
    if seen != set(range(len(labels))):
        raise ValueError("Label indices must be continuous from 0 to N-1")
    LABELS[name] = labels


def register_loader(name: str):
    def decorator(fn: Callable[..., Iterator[dict[str, Any]]]):
        if name not in LABELS:
            raise ValueError(f"No labels registered for dataset: {name}")
        LOADERS[name] = fn
        return fn

    return decorator


def get_loader(name: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset: {name}")
    return LOADERS[name](**kwargs)


def get_labels(name: str) -> dict[str, dict[str, Any]]:
    if name not in LABELS:
        raise KeyError(f"No labels for dataset: {name}")
    return LABELS[name]


def categorical_vec(dataset: str, acronyms: Iterable[str]) -> list[int]:
    labels = get_labels(dataset)
    vector = [0] * len(labels)
    for acronym in acronyms:
        if acronym not in labels:
            raise KeyError(f"Unknown label {acronym!r} for dataset {dataset!r}")
        vector[labels[acronym]["label"]] = 1
    return vector


def labels_from_vector(dataset: str, vector: Sequence[int]) -> list[str]:
    labels = get_labels(dataset)
    if len(vector) != len(labels) or any(value not in (0, 1) for value in vector):
        raise ValueError(f"Expected a binary vector of length {len(labels)}")
    return [acronym for acronym, spec in labels.items() if vector[spec["label"]]]


vector_to_labels = labels_from_vector


def label_to_vector(dataset: str, acronym: str) -> list[int]:
    return categorical_vec(dataset, [acronym])


def get_label_by_idx(dataset: str, idx: int) -> str | None:
    for acronym, spec in get_labels(dataset).items():
        if spec["label"] == idx:
            return acronym
    return None
