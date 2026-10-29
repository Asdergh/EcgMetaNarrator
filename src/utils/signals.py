import torch as th
from torchaudio.transforms import (Spectrogram, )
from torchvision.transforms import (Compose, Resize, Lambda)
from typing import Optional, Tuple
import kornia.enhance as  KE



def normalize(x: th.Tensor, a: float=0, b: float=1):
    return a + ((x - x.min()) * (b - a) 
                / (x.max() - x.min() + 1e-5))

def raw_sig2spec(sampling_rate: float,
                raw: Optional[th.Tensor]=None,
                delta_f: float=5.0,
                delta_t: float=0.05,
                equalize: bool=False,
                standartization: bool=False,
                return_callable: bool=False,
                size: Optional[Tuple[int] | int]=None,
                **kwargs):
    """convert raw 12 channeled ECG signals into Spectrograms of size. 
    Arguments:
        saampling_rate: signal sampling ratee. 
        raw: th.Tensor[channels, N], where N = sampling_rate * T, T in sec. 
        delfa_f: float - frequence window step. 
        delfa_t: float - time bins winow step.
        standartization: bool - where or not to apply features standartization. 
                                Stnadartization will applied before euqalization 
                                to change data digits range before applying normlization.
        equzalization: bool - where or not to apply image equalization 
                            to resulting spectrogram. If equalization=True
                            additional normalization will be added into Compose 
                            callable. 
        **kwargs - additional arguments for Spectrogram module. 
    Note:
        delta_f and delta_t  sets the resulting height and width
        of spectrogram according to formulas. As lower values to choose as bigger 
        the respective axis becomes. """
    tfs = []
    nfft = int(sampling_rate / delta_f)
    hop_lenght = int(sampling_rate * delta_t)
    tfs.append(Spectrogram(n_fft=nfft, hop_length=hop_lenght, **kwargs))
    if size is not None:
        size = size if isinstance(size, tuple) else (size, )*2
        assert len(size) == 2
        tfs.append(Resize(size))
    if standartization:
        func = lambda x: (x / (x.max() + 1e-5))
        tfs.append(Lambda(func))
    if equalize:
        func = lambda x: normalize(x, 0, 1)
        tfs.append(Lambda(func))
        func = lambda x: KE.equalize(x)
        tfs.append(Lambda(func))
    if return_callable:
        return Compose(tfs)
    else:
        return Compose(tfs)(raw)
