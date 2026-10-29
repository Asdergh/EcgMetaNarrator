import torch as th
from torchaudio.transforms import Spectrogram
from typing import Optional


def raw2spec(raw: th.FloatTensor, 
            f_rate: float, 
            delta_t: float, 
            delta_f: float,
            equalization: bool=False,
            standartization: bool=False,
            normalization_range: Optional[tuple]=None,
            size: Optional[tuple | int]=None):
    """raw signal to spectrograms conversion.
    Arguments:
        f_rate: sampling frequence rate from input raw signal. 
        delta_t: time windows step between two spectral stacks. 
        delta_f: spectral steep between spectral bings on plot. 
        equalization: bool - apply wisual featuers equalization. 
        std_normalization: apply standartization to spectrogram. 
        normalization_range: Optional[(a, b)] - values range to nfr restrict output spectrogram. 
        size: Optional[tuple | int]: size for output spectrograms. """

    n_fft = int(f_rate / delta_f)
    hop_lenght = n_fft // 2 
    tfs = []
    tfs.append(Spectrogram(n_fft=n_fft, hop_length=hop_lenght))
    if size is not None:
        size = size if isinstance(size, tuple) else 2*(size, )
        tfs.appnd()

    # spectrograms = tf(raw)
