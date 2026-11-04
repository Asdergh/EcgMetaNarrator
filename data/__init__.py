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

import wfdb 
import torch as th
import numpy as np
from typing import Literal
from torch.utils.data import Dataset
from tqdm import tqdm

class EcgAnnotationDataset(Dataset):
    def __init__(self, root, loader, cache_splits_portions=(0.5, 0.5)):
        assert abs(sum(cache_splits_portions) - 1.0) < 1e-6
        self._meta = list(get_loader(loader, root=root))
        self._labels = get_labels(loader)

        n = len(self._meta)
        sizes = [int(n * p) for p in cache_splits_portions]
        sizes[-1] += n - sum(sizes)

        self._bounds = [0]
        for s in sizes:
            self._bounds.append(self._bounds[-1] + s)
        self._n = n
        self._chunks = [
            self._meta[self._bounds[i]:self._bounds[i + 1]]
            for i in range(len(sizes))
        ]
        self._cache = {}

    def _split_of(self, idx):
        for sid in range(len(self._chunks)):
            if idx < self._bounds[sid + 1]:
                return sid, idx - self._bounds[sid]
        raise IndexError(idx)

    def _load_split(self, sid):
        if sid in self._cache:
            return self._cache[sid]
        out = []
        for meta in tqdm(self._chunks[sid]):
            print(meta["label"])
            # print(list(self._labels[meta["label"]].keys()))
            h = wfdb.rdrecord(meta["file_path"])
            out.append({
                **meta,
                "raw_signal": th.from_numpy(h.p_signal).float().transpose(1, 0),
                "fs": h.fs,
                "label": self._labels[meta["label"]]["label"],
            })
        self._cache[sid] = out
        return out

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._n
        if not (0 <= idx < self._n):
            raise IndexError(idx)
        sid, local = self._split_of(idx)
        sample = self._load_split(sid)[local]
        print(list(sample.keys()))
        return {k: sample[k] for k in ("raw_signal", "fs", "label")}


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from pathlib import Path
    name = "ptbxl"
    root = "/run/media/ramzan/T7/datasets/ecg/ptb-xl-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    root = Path(root)
    dataset = EcgAnnotationDataset(root, name, cache_splits_portions=[0.3, 0.4, 0.3])
    loader = DataLoader(dataset=dataset, batch_size=32)
    print(len(dataset))
    for sample in loader:
        print(sample["raw_signal"].shape, sample["label"].shape, th.unique(sample["label"]))
    for sample in loader:
        print(sample["raw_signal"].shape, sample["label"].shape, sample["label"][:10])