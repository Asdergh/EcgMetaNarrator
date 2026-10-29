import torch as th
import torch.nn as nn
from inspect import signature
from warnings import warn
from typing import Any, Callable, Dict, Optional, List
from dataclasses import (dataclass, fields)
from .layers import get_activation

__BACKBONES__ = {}
def _check_backbon(backbone: str):
    for attr in ["_name",
                "_embedding_dim",
                "_repo_id",
                "_acronim"]:
        if not hasattr(backbone, attr):
            raise ValueError(f"wrong backbone object. Cound't fidn: {attr} attribute.")
        
def register_backbone(owerride: bool=False):
    """backbone registraction decorator. 
    Arguments:
        owerride: where owerride already registered backbone or raise en error. 
        Else arguments provided accoring BackboneMeta structure. """
    def decodator(cls):
        _check_backbon(cls)
        name = cls._name
        if (name not in __BACKBONES__) or owerride:
            __BACKBONES__[name] = cls
        else:
            RuntimeError(f"{name} backbone is already in __BACKBONES__")
        return cls
    return decodator

def get_backbone(name: str):
    meta = __BACKBONES__.get(name, None)
    if name is None:
        warn(f"unknow backbone: {name}")
    return meta


class FilmFiltration(nn.Module):
    def __init__(self, features: int):
        self._alpha_gate = nn.Sequential(nn.Linear(features, 2), get_activation("sigmoid"))
        self._beta = nn.Sequential(nn.Linear(features, 1), get_activation("relu"))

    def forward(self, x: th.Tensor):
        alpha, gate = self._alpha_gate(x).chunk(2, dim=-1)
        beta = self._beta(x)
        x_transformed = alpha*x + beta
        x_gated = gate*x + (1 - gate)*x_transformed
        return x_gated

@dataclass
class fusionHeadConfig:
    core_embedding_dim: int
    dp: float=0.45
    activation: str="relu"
    filtration: bool=False
    normalization: bool=False

class BackboneWrapper(nn.Module):
    def __init__(self, fusion_config: fusionHeadConfig):
        """Fusion backbone wrapper to project different models 
        latent features into one core_embedding_dim. 
        Arguments:
            fusion_config: fusionHeadConfig class with fields:
                core_embedding_dim: int - features dimentionality to project backbone latents.
                dp: float - dropout rate for input latent. Adds some noise to signal from backbone
                            to make _head more robust to specified input type. 
                activations: str - activation function for _head. 
                filtration: bool - wheere or not to apply film gated filtration layer. 
                normalization: bool - where or not to apply layer normalization after projection in _head. 
        Note:
            Every backbone pre-wrapper that is registered to be incapsulated
            by BackboneWrapper must have _embedding_dim of its outputs specified 
            by pre-wrapper instance i.e backbone itself. """
        super(BackboneWrapper, self).__init__()
        self._backbone.eval()
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        self._fcfg = fusion_config
        self._backbone_embedding_dim = self._backbone._embedding_dim
        _blocks = []
        _blocks.append(nn.Dropout(self._fcfg.dp))
        _blocks.append(nn.Linear(self._backbone_embedding_dim, self._fcfg.core_embedding_dim))
        if self._fcfg.normalization:
            _blocks.append(nn.LayerNorm(self._fcfg.core_embedding_dim))
        _blocks.append(get_activation(self._fcfg.activation))
        if self._fcfg.filtration:
            _blocks.append(FilmFiltration(self._fcfg.core_embedding_dim))
        self._head = nn.Sequential(*_blocks)

    def _check_backbon(self, backbone):
        """Check backbone. """
        for attrib in ["_name", "_embedding_dim", "_params_src", "load"]:
            if not hasattr(self._backbone, attrib):
                raise ValueError(f"missing {attrib} attribute in {type(backbone)}")
        if backbone._name not in __BACKBONES__:
            raise NotImplemented(f"backbon {backbone._name} with type: {type(backbone)} wasn't registere. ")
        
    def state_dict(self, *args, **kwargs):
        """clean state dict without _backbone weightss for BackBoneWrapper. """
        state_dict = super().state_dict(*args, **kwargs)
        clean_state_dict = {k: v for (k, v) in state_dict.items() if not k.startith("_backbone")}
        return clean_state_dict
    
    def forward(self, x: th.Tensor, **kwargs):
        x = self._backbone(x, **kwargs)
        assert x.ndim == 3, \
            ("backbone wrapper expects features in form [B, N, C] \n"
            "where N - tokens lenghts. Thereforre you must check \n"
            "tour backbone forward function to satisfy this requirement.")
        return self._head(x)



class MetaFeaturesMixer(nn.Module):
    def __init__(self, 
                backbones: Dict[str, str | BackboneWrapper],
                fusion_config: fusionHeadConfig):
        super(MetaFeaturesMixer, self).__init__()
        _backbones = {}
        for (name, src) in backbones.items():
            if isinstance(src, str):
                meta = get_backbone(name)
                if (meta is None):
                    warn(f"unknown backbone: {name} at MetaFeaturesMixer initialization")
                    continue 
                try:
                    backbone = meta.cls.load(src)
                    backbone = BackboneWrapper(backbone=backbone, fusion_config=fusion_config)
                except Exception as e:
                    warn(f"coudn't load backbone: {name} due to: {e}")
                    continue
            elif isinstance(src, BackboneWrapper):
                if src._fcfg.core_embedding_dim != fusion_config.core_embedding_dim:
                    warn("only BackboneWrapper module with the same \n" \
                    "core_embedding_dim can be used as backbone in MetaFeaturesFuser. \n" \
                    f"This instance have: {src._fcfg.core_embedding_dim}, therefore is will be skiped. \n")
                    continue 
                backbone = src
            else:
                raise TypeError(f"unknown backbone source type: {type(src)}")
            _backbones[name] = backbone
        if _backbones:
            self._backbones = nn.ModuleDict(_backbones)
        else:
            raise ValueError("empty self._backbones. ")
        
    def forward(self, inputs: th.Tensor, return_dict: bool=False, **kwargs):
        tokens = {} if return_dict else []
        for name, m in self._backbones.items():
            f = m(inputs, **kwargs)
            if return_dict:
                tokens[name] = f
            else:
                tokens.append(f)
        return tokens \
                if return_dict \
                else th.cat(tokens, dim=1)
