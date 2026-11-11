import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable, Literal
from torch.optim import Optimizer, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR

from .register import get_backbone
from heapq import (heappop, heappush, heapreplace, heapify)
from dataclasses import dataclass, field

import numpy as np
from warnings import warn
import lightning as l
from torch.utils.data import Dataset
import wfdb
from copy import deepcopy
from data import get_loader, get_ptbxl_with_pruning
from .criterion import MultiBCECriterion

class EcgAnnotatedSet(Dataset):
    def __init__(self, meta: List[Dict[str, Any]], get_raw: bool=True):
        self._meta = meta
        self._get_raw = get_raw
        self._raw_cache: Dict[int, th.Tensor] = {}

    def _get_sample(self, idx: int):
        """get samples and cache signals. """
        sample = deepcopy(self._meta[idx])
        if idx not in self._raw_cache:
            meta = wfdb.rdrecord(sample["file_path"])
            fs = meta.fs
            raw = th.from_numpy(meta.p_signal).float().transpose(0, 1)
            self._raw_cache[idx] = raw
        sample["raw_signal"] = self._raw_cache[idx]
        sample["fs"] = fs
        sample["label"] = th.asarray(sample["label"]).to(th.long)
        return sample

    def __len__(self):
        return len(self._meta)

    def __getitem__(self, idx: int):
        if idx == len(self):
            raise IndexError("out of range for EcgAnnotatedSet with len: {}".format(len(self)))
        sample = self._get_sample(idx)
        return {key: value 
                for (key, value) in sample.items() 
                if key in ["raw_signal", "label", "fs"]}
    
class BackboneDataModule(l.LightningDataModule):
    def __init__(self, 
                root: str, 
                loader: str, 
                loader_cfg: Optional[dict]=None,
                splits_vols: List[float]=[0.5, 0.4, 0.1],
                batch_size: int=32,
                shuffle: bool=False):
        super(BackboneDataModule, self).__init__()
        self._loader_name = loader
        self._loader_cfg = loader_cfg
        self._data_root = root
        self._splits_vols = splits_vols
        self._batch_size = batch_size
        self._shuffle = shuffle
        assert len(self._splits_vols) == 3
        assert abs(sum(self._splits_vols) - 1) < 1e-6

    def prepare_data(self):
        """load data meta storage and create data splits. """
        loader_cfg = {} if self._loader_cfg is None else self._loader_cfg
        if self._loader_name == "ptbxl":
            self._meta = list(get_ptbxl_with_pruning(root=self._data_root, **loader_cfg)["loader"])
        else:
            self._meta = list(get_loader(self._loader_name, root=self._data_root, **loader_cfg))

        parts = np.cumsum(
            np.asarray([len(self._meta) * p for p in self._splits_vols])
        ).astype(np.int32).tolist()
        self._splits = {"train": self._meta[0:parts[0]],
                        "validate": self._meta[parts[0]:parts[1]],
                        "test": self._meta[parts[1]:]}

    def setup(self, stage: str):
        """setup train/validate/test dataset splits. """
        if stage in ("vlaidate", "fit"):
            self._train_dataset = EcgAnnotatedSet(meta=self._splits["train"])
            self._validate_dataset = EcgAnnotatedSet(meta=self._splits["validate"])
        elif stage == "test":
            self._test_dataset = EcgAnnotatedSet(meta=self._splits["test"])

    def train_dataloader(self):
        if hasattr(self, "_train_dataset"):
            return DataLoader(dataset=self._train_dataset, 
                            batch_size=self._batch_size, 
                            shuffle=self._shuffle)
        else:
            return 
    def val_dataloader(self):
        if hasattr(self, "_validate_dataset"):
            return DataLoader(dataset=self._validate_dataset, 
                            batch_size=self._batch_size, 
                            shuffle=self._shuffle)
        else:
            return 
    def test_dataloader(self):
        if hasattr(self, "_test_dataset"):
            return DataLoader(dataset=self._test_dataset, 
                            batch_size=self._batch_size, 
                            shuffle=self._shuffle)
        else:
            return 

    
class BackboneTrianingContainer(l.LightningModule):
    def __init__(self, 
                module_name: str,
                k: int=3, 
                mode: Literal["min", "max"]="min",
                monitor: Optional[str]=None,
                save_per_epoch: int=100,
                initial_lr: float=0.01,
                lr_exponential_gamma: float=0.95,
                metrics_list: Optional[List[str]]=None):
        super(BackboneTrianingContainer, self).__init__()
        self._backbone_cls = get_backbone(module_name)
        self._backbone = self._backbone_cls()
        self._backbone.init_model()
        self._loss_states = []
        self._k = k
        self._monitor = monitor
        self._mode = mode
        self._per_epoch = save_per_epoch
        self._lr = initial_lr
        self._gamma = lr_exponential_gamma

        criterion_kwargs = {k: True for k in metrics_list}
        self._criterion = MultiBCECriterion(**criterion_kwargs)
        # self._val_loss_accum = None

    def _save_weights(self, value: float):
        if len(self._loss_states) < self._k:
            idx = len(self._loss_states)
            self._loss_states.append(value)
            self._backbone.save2hf(f"version_{idx}")
        else:
            assert (value is not None)
            _mode = dict(min=max, max=min)
            idx = self._loss_states.index(_mode[self._mode](self._loss_states))
            if (self._mode == "min") and (value < self._loss_states[idx]) \
                or (self._mode == "max") and (value > self._loss_states[idx]):
                self._loss_states[idx] = value
                self._backbone.save2hf(f"version_{idx}")

    def _check_batch(self, batch: Dict[str, Any]):
            for key in ["raw_signal", "fs", "labels"]:
                if key not in batch:
                    raise ValueError(f"attrib {key} required in batch")

    def configure_optimizers(self):
        """simple adam optimizer with exponential scheduler. """
        opt = Adam(self._backbone.parameters(), lr=self._lr)
        scheduler = ExponentialLR(opt, gamma=self._gamma)
        return {"optimizer": opt, "scheduler": scheduler}
        
    def batch_step(self, batch: Dict[str, Any], batch_idx: int, mode: str):
        """batch step function. """
        assert mode in ["train", "val", "test"]
        accum = True if mode == "val" else False
        self._check_batch(batch)
        (raw, fs, labels) = (batch["raw_signal"], batch["fs"], batch["labels"])
        model_out = self._backbone(raw=raw, fs=fs)
        loss_pkg = self._criterion(labels, model_out.prediction, accum=accum)
        for (key, value) in loss_pkg.items():
            self.log(f"{mode}_{key}", value, on_epoch=True)
        return {"loss": loss_pkg["loss"]}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.batch_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        with th.no_grad():
            return self.batch_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        """check wheree to save latest epoch call states. """
        metrics = self._criterion.normalize()
        assert (self._monitor in metrics), \
            (f"coudn't find metric: {self._monitor} in metrics list: {list(metrics.keys())}")
        value = metrics[self._monitor]
        self._criterion.reset()
        if (self.current_epoch % self._per_epoch) == 0:
            self._save_weights(value)
