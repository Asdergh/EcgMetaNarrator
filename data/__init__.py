"""Единообразные описания наборов данных ЭКГ и доступ к сигналам"""

from .loaders import (
    load_challenge2018,
    load_challenge2020,
    load_ludb,
    load_ptbdb,
    load_ptbxl,
)
from .readers import SignalReader
from .registry import (
    LABELS,
    LOADERS,
    get_label_by_idx,
    get_labels,
    get_loader,
    label_to_vector,
    register_labels,
    register_loader,
)

__all__ = [
    "LABELS",
    "LOADERS",
    "SignalReader",
    "get_label_by_idx",
    "get_labels",
    "get_loader",
    "label_to_vector",
    "register_labels",
    "register_loader",
]
