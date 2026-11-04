import io
import abc
import os
import tempfile 
from pathlib import Path
import timm
from timm.data import resolve_data_config, create_transform
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn
from huggingface_hub import HfApi, hf_hub_download, file_exists
from .register import register_backbone
from typing import Optional, Tuple, Dict, Any, List, Literal
from ...utils.signals import raw_sig2spec
from torchvision.transforms import (Compose, 
                                    Normalize,
                                    Resize,
                                    GaussianBlur,
                                    CenterCrop)
from dataclasses import dataclass
import torch.nn.init as init



def _check_input(raw: th.Tensor):
    if raw.ndim > 3:
        raise ValueError("A batch of raw signals stacked into 2D Tensors expected")
    if raw.shape[0] > raw.shape[1]:
        raw = raw.transpose(0, 1)
    return raw

def _is_rgb_tf(module) -> bool:
    classes = []
    try:
        from torchvision.transforms import RandomGrayscale as V1
        classes.append(V1)
    except ImportError:
        pass
    try:
        from torchvision.transforms.v2 import RandomGrayscale as V2
        classes.append(V2)
    except ImportError:
        pass
    return bool(classes) and isinstance(module, tuple(classes))

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)


@dataclass
class featuresOutput:
    feature_maps: Optional[Tuple[th.Tensor]]=None
    last_hiddden_state: Optional[th.Tensor]=None
    prediction: Optional[th.Tensor]=None

class BaseFeaturesModel(nn.Module, abc.ABC):
    _repo_id: str="DeltaRim/corinteltech"
    def __init__(self):
        super(BaseFeaturesModel, self).__init__()

    def set_num_classes(self, n: int):
        """set n classes for classification. """
        self._n_classes = n
        
    def init_model(self, state_dict: Optional[Dict[Any, Any]]=None, **kwargs):
        """initialize timm model. """
        n_classes = kwargs.pop("num_classes", None)
        n_classes = n_classes if n_classes is not None else 1
        self._model = timm.create_model(self._acronim, 
                                        pretrained=False, 
                                        in_chans=12, 
                                        num_classes=n_classes, 
                                        **kwargs)
        if state_dict is not None:
            try:
                self._model.load_state_dict(state_dict)
            except Exception as e:
                warn(f"coudn't load state dict, due to: {e}")
        self._tf = self.get_transforms()

    @abc.abstractmethod
    def forward(self, raw: th.Tensor):
        raise NotImplemented("features module must have forward module. ")

    def get_transforms(self):
        tfs = []
        if hasattr(self._model, "pretrained_cfg"):
            data_cfg = resolve_data_config({}, model=self._model)
            tf_compose = create_transform(**data_cfg)
            for tf in tf_compose.transforms:
                if not isinstance(tf, (Resize, Normalize, GaussianBlur, CenterCrop)):
                    continue
                if isinstance(tf, Normalize):
                    mean = th.asarray(tf.mean)
                    mean = F.pad(mean, (0, 9), value=mean.mean())
                    mean[3:] += 1e-1*th.normal(mean.mean(), mean.std(), (9, ))
                    std = th.asarray(tf.std)
                    std = F.pad(std, (0, 9), value=std.mean())
                    std[3:] += 1e-1*th.normal(std.mean(), std.std(), (9, )) 
                    tf = Normalize(mean=mean.tolist(), std=std.tolist())
                tfs.append(tf)
            tfs = Compose(tfs)
        else:
            tfs = nn.Identity()
        return tfs
    
    def save2hf(self, fname: str):
        fname = fname if fname is not None else "backbone"
        api = HfApi()
        state_dict = self._model.state_dict()
        if hasattr(self, "_n_classes"):
            state_dict["n_classes"] = self._n_classes
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmpf = Path(tmp.name)
            try:
                th.save(state_dict, tmpf)
                api.upload_file(repo_id=self._repo_id,
                                repo_type="model",
                                path_in_repo= "{}/{}.bin".format(self._name, fname),
                                path_or_fileobj=str(tmpf))
            finally:
                tmpf.unlink(missing_ok=True)

    @classmethod
    def list_versions(cls):
        api = HfApi()
        return [f.split("/")[-1].replace(".bin", "") 
                for f in api.list_repo_files() 
                if f.startswith(f"{cls._name}")]
    
    @classmethod
    def load(cls, version_fname: str):
        hf_path = "{}/{}.bin".format(cls._name, version_fname)
        cls = cls()
        if file_exists(repo_id=cls._repo_id, 
                    filename=hf_path):
            statesf = hf_hub_download(repo_id=cls._repo_id,
                                        filename=hf_path,
                                        repo_type="model")
            state_dict = th.load(statesf, map_location="cpu")
            n_classes = state_dict.pop("n_classes", None)
            cls.init_model(state_dict, num_classes=n_classes)
        else:
            warn(f"coudn't find model weigths version at: {cls._repo_id}")
        return cls

@register_backbone()
class featuresFasterNet(BaseFeaturesModel):
    _name: str="fasternet"
    _embedding_dim: int=320
    _acronim: str="fasternet_t0.in1k"

    def __init__(self):
        super(featuresFasterNet, self).__init__()
        self.init_model()

    def forward(self, 
                raw: th.Tensor, 
                fs: float, 
                out_indices: Optional[List[int]]=None):

        raw = _check_input(raw)
        dense = raw_sig2spec(raw=raw, sampling_rate=fs)
        dense = self._tf(dense)

        features = self._model.patch_embed(dense)
        intermediates = []
        for idx, layer in enumerate(self._model.stages.children()):
            features = layer(features)
            if idx in out_indices:
                intermediates.append(features)

        x_cls = features
        for layer in ["global_pool", 
                    "conv_head", 
                    "act", "flatten", 
                    "classifier"]:
            layer = getattr(self._model, layer)
            x_cls = layer(x_cls)

        return featuresOutput(prediction=x_cls,
                            feature_maps=intermediates,
                            last_hidden_state=features.flatten(start_dim=-2).transpose(-1, -2))

@register_backbone()
class featuresCrossVit(BaseFeaturesModel):
    _name: str = "cross_vit"
    _embedding_dim: int = 328
    _acronim: str = "crossvit_9_240.in1k"
    _img_size: Tuple[int]=(240, 240)

    def __init__(self):
        super(featuresCrossVit, self).__init__()
        self.init_model(img_size=self._img_size)
        
    def forward(self,
                raw: th.Tensor,
                fs: float,
                out_branch: int=0,
                **kwargs):
        
        raw = _check_input(raw)
        dense = raw_sig2spec(raw=raw, sampling_rate=fs)
        dense = self._tf(dense)

        features = self._model.forward_features(dense)
        embedding = th.cat([f[:, 0, :] for f in features], dim=-1)
        predictions = self._model.forward_head(features)
        return featuresOutput(prediction=predictions[out_branch], last_hidden_state=embedding)



if __name__ == "__main__":

    model = featuresCrossVit()
    fs = 500
    T = 10
    n = int(T * fs)
    data = th.normal(0, 1, (1, 12, n))
    output = model(data, fs=fs)
    print(output.prediction[0].shape)




