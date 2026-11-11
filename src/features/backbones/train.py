import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
from torch.utils.data import DataLoader
from  data import EcgAnnotationDataset
from typing import Dict, List, Tuple, Optional, Any, Callable, Literal
from torch.optim import Optimizer, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR

from .register import get_backbone
from heapq import (heappop, heappush, heapreplace, heapify)
from dataclasses import dataclass, field

# @dataclass(order=True)
# class Checkpoint:
#     metric: float=field(compare=True)
#     state_dict: Dict[str, Any]=field(compare=False)
#     def save(self, path: str):
#         "save checkpoin path as .ckpt pickle file. "
#         if os.path.exists(path):
#             th.save(self.state_dict, path)
        
# class CheckpointsCallback:
#     def __init__(self, 
#                 module: Callable[..., nn.Module],
#                 mode: Literal["min", "max"]="min", 
#                 k: int=3,
#                 name: Optional[str]=None):
#         self._k = k
#         self._mode = mode
#         self._moule = module
#         self._name = name if name is not None else "model"
#         self._heap: List[Checkpoint] = []

#     def save(self, path: str):
#         """save collected checkpoints. """
#         root = os.path.join(path, self._naem)
#         os.makedirs(root)
#         for idx, state in enumerate(self._heap):
#             fname = os.path.join(root, f"params_v_{idx}.ckpt")
#             state.save(fname)

#     def push(self, value: float):
#         state = Checkpoint(value, self._module.state_dict())
#         if len(self._heap) < self._k:
#             heappush(self._heap, state)
#         else:
#             if self._mode == "min" \
#                 and (state < self._heap[0]):
#                 heapreplace(self._heap, state)
#             elif self._mode == "max" \
#                 and (state > self._heap[0]):
#                 heapreplace(self._heap, state)
    
#     def pop(self):
#         return heappop(self._heap)
    
#     def __len__(self):
#         return len(self._heap)

# __OPTIMIZERS__ = {"adam": Adam, "sgd": SGD}
# def get_optimizer(name: str, **kwwargs):
#     """choose optimizer. """
#     opt = __OPTIMIZERS__.get(name.islower())
#     assert (opt is not None), (f"coudn't find optimizer: {name}")
#     return opt(**kwwargs)

# def _get_loader(name: str, 
#                 root: str, 
#                 batch_size: int=32, 
#                 shuffle: bool=False):
#     """get data loader. """
#     dataset = EcgAnnotationDataset(root=root, loader=name)
#     return DataLoader(dataest=dataset, shuffle=shuffle, batch_size=batch_size)



# def batch_step_backbone(model: Callable[..., nn.Module],
#                         batch: Dict[str, th.Tensor],
#                         optimizer: Optional[Optimizer]=None,
#                         loss_module: Optional[Callable[..., nn.Module]]=None):
#     """process one batch for """
#     output = dict(zip(["loss", "predictions", "features"], [None, ]*3))
#     assert ("raw_signal" in batch) and ("fs" in batch)
#     (raw, fs, label) = (batch["raw_signal"], batch["fs"], batch["label"])
#     out = model(raw=raw, fs=fs)
#     output["predictions"] = out.prediction
#     output["features"] = out.last_hidden_state
#     if loss_module is not None:
#         loss = loss_module(out.prediction, label)
#         output["loss"] = loss
#         if optimizer is not None:
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
        
#     return output

# def evalute(backbone: str,
#             loader: str,
#             data_root: str,
#             out_path: str,
#             epochs: int=1000,
#             batch_size: int=32,
#             shuffle: bool=False,
#             verbose: bool=False,
#             mode: Literal["train", "val"]="val",
#             optimizer: Literal["adam", "sgb"]="adam",
#             optimizer_config: Dict[str, Any]={},):
#     """backbone evaluation function. """
#     backbone = get_backbone(backbone)
#     checkpoints = CheckpointsCallback(backbone)
#     loader = _get_loader(loader, data_root, batch_size, shuffle)
#     if mode == "train":
#         optimizer = get_optimizer(optimizer, **optimizer_config)
#         backbone.train()
#     else:
#         backbone.eval()

#     loss_module = loss_module
#     for epoch in range(epochs):
#         for (batch_idx, batch) in enumerate(loader):
#             if mode == "val":
#                 with th.no_grad():
#                     batch_pkg = batch_step_backbone(backbone, batch, optimizer, loss_module)
#             else:
#                 batch_pkg = batch_step_backbone(backbone, batch, optimizer, loss_module)
#                 if verbose:
#                     pass
#         if mode == "train":
            

#     if mode == "train":
#         os.makedirs(out_path, exist_ok=True)

from warnings import warn
from .criterion import MultiBCECriterion
import lightning as l

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
        # self._latest_loss_value = loss.item()
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




def train(backbone: str,
        criterion: str,
        train_loader: str,
        train_loader_root: str,
        val_loader: Optional[str]=None,
        val_loader_root: Optional[str]=None,
        epochs: int=1000,
        batch_size: int=32,
        shuffle: bool=False,
        initial_lr: float=0.01,
        lr_exponential_gamma: float=0.95):
    """main training function for BackboneTrainingContainer. """
    criterion = get_criterion(criterion)
    container = BackboneTrianingContainer(backbone, 
                                        criterion=criterion,
                                        mode=)