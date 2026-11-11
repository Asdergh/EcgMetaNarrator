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
    categorical_vec,
    get_label_by_idx,
    get_labels,
    get_loader,
    label_to_vector,
    labels_from_vector,
    vector_to_labels,
    register_labels,
    register_loader,
)



import numpy as np
def _prune(root: str, min_samples: int=0, **kwargs):
    """dropout labels from __LABELS__ with 0 friquencies. """
    loader = get_loader("ptbxl", root=root, **kwargs)
    labels = LABELS.get("ptbxl", None)
    if labels is None:
        raise ValueError("coudn't find ptbxl labels in russion dataset. ")
    
    labels_stack = np.stack([np.asarray(sample["label"]) for sample in loader])
    columns_stack = labels_stack.sum(axis=0)
    indices = np.where(columns_stack > min_samples)[0].tolist()

    keys = list(labels.keys())
    index = 0
    for key in keys:
        if labels[key]["label"] not in indices:
            LABELS["ptbxl"].pop(key)
        else:
            LABELS["ptbxl"][key]["label"] = index
            index += 1

def prune_labels_ptbxl(root: str, min_samples: int, **kwargs):
    loader = get_loader("ptbxl", root=root, **kwargs)
    _ = next(iter(loader))
    _prune(root, min_samples, **kwargs)

def get_ptbxl_with_pruning(root: str, min_samples: int=0, **kwargs):
    prune_labels_ptbxl(root, min_samples, **kwargs)
    loader = get_loader("ptbxl", root=root, **kwargs)
    labels = get_labels("ptbxl")
    return {"loader": loader, "labels": labels}
    
__all__ = [
    "LABELS",
    "LOADERS",
    "SignalReader",
    "categorical_vec",
    "get_label_by_idx",
    "get_labels",
    "get_loader",
    "label_to_vector",
    "labels_from_vector",
    "vector_to_labels",
    "register_labels",
    "register_loader",
    "prune_labels_ptbxl",
    "get_ptbxl_with_pruning"
]
