import torch as th
import numpy as np
import requests
import io
import base64
import os
import json
from functools import wraps
from openai import OpenAI
from typing import List, Optional, Dict, Tuple, Union, Any, Literal
from dataclasses import dataclass, fields
from ..backbones import MetaFeaturesMixer, backbone_pre_forward



from pathlib import Path
ROLES_PATH = Path(__file__).parent / "roles.json"

def list_roles():
    """list all system roles for llm agents. """
    fname = "roles.json"
    assert os.path.exists(fname)

def add_role(name: str, role: str, message: str | List[str]):
    """update roles storage. """
    with open(str(ROLES_PATH), "r+", encoding="utf-8") as file:
        roles = json.load(file)
        roles.update({name: {"role": role, "message": message}})
        file.seek(0)
        json.dump(roles, file)
        file.truncate()

def get_role(name: str="general"):
    """get prompt by name. """
    with open(str(ROLES_PATH), "r", encoding="utf-8") as file:
        roles = json.load(file)
    if name != "general":
        return roles.get(name, None)
    else:
        return tuple(roles.values())

def encode_b64(x: th.Tensor):
    """return base64 encoding for torch tensors. """
    buffer = io.BytesIO()
    th.save(x, buffer)
    buffer.seek(0)
    return base64.encode(buffer.read()).decode()
    
@dataclass
class WaveDescriptionPrompt:
    text: Optional[str]=None
    text_response: Optional[str]=None
    wave: Optional[np.ndarray | th.Tensor]=None
    sampling_rate: Optional[float]=None
    wave_description: Optional[str]=None
    wave_full_description: Optional[str]=None

@dataclass
class GenerationParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0

    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None

    do_sample: bool = False
    num_beams: int = 1
    num_return_sequences: int = 1
    early_stopping: Union[bool, str] = False

    exponential_decay_length_penalty: Optional[Tuple[int, float]] = None
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    forced_decoder_ids: Optional[List[List[int]]] = None
    sequence_bias: Optional[Dict[Tuple[int, ...], float]] = None
    guidance_scale: Optional[float] = None
    watermarking_config: Optional[Any] = None

    output_scores: bool = False
    output_logits: bool = False
    return_dict_in_generate: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False

    def to_hf_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "early_stopping": self.early_stopping,
            "output_scores": self.output_scores,
            "output_logits": self.output_logits,
            "return_dict_in_generate": self.return_dict_in_generate,
            "output_attentions": self.output_attentions,
            "output_hidden_states": self.output_hidden_states,
        }

        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        elif self.max_length is not None:
            kwargs["max_length"] = self.max_length

        if self.min_new_tokens is not None:
            kwargs["min_new_tokens"] = self.min_new_tokens
        elif self.min_length:
            kwargs["min_length"] = self.min_length

        if self.exponential_decay_length_penalty is not None:
            kwargs["exponential_decay_length_penalty"] = self.exponential_decay_length_penalty
        if self.suppress_tokens is not None:
            kwargs["suppress_tokens"] = self.suppress_tokens
        if self.begin_suppress_tokens is not None:
            kwargs["begin_suppress_tokens"] = self.begin_suppress_tokens
        if self.forced_decoder_ids is not None:
            kwargs["forced_decoder_ids"] = self.forced_decoder_ids
        if self.sequence_bias is not None:
            kwargs["sequence_bias"] = self.sequence_bias
        if self.guidance_scale is not None:
            kwargs["guidance_scale"] = self.guidance_scale
        if self.watermarking_config is not None:
            kwargs["watermarking_config"] = self.watermarking_config

        return kwargs

    def to_vllm_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": self.num_return_sequences,
        }

        if self.max_new_tokens is not None:
            kwargs["max_tokens"] = self.max_new_tokens
        elif self.max_length is not None:
            kwargs["max_tokens"] = self.max_length

        if self.min_new_tokens is not None:
            kwargs["min_tokens"] = self.min_new_tokens
        elif self.min_length:
            kwargs["min_tokens"] = self.min_length

        if self.num_beams > 1:
            kwargs["use_beam_search"] = True
            kwargs["best_of"] = self.num_beams
            if self.early_stopping is not False:
                kwargs["early_stopping"] = self.early_stopping

        return kwargs


class WaveDescriptorInferencePiepeline:
    _functions_attributes: Dict[str, Tuple[str]]={
        "process_wave": ("wave", "sampling_rate"),
        "process_description": ("wave_description", ),
        "process_text": ("text", )
    }
    def _pre_inference_check(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            prompt = kwargs["prompt"]
            fname = func.__name__
            llm_mode = "narrator" if fname == "process_wave" else "core"
            if not hasattr(self, f"llm_{llm_mode}"):
                    raise RuntimeError(f"first setup {llm_mode} model with setup_llm(..., mode='narrator'). ")
            fn_prompt_attributes = self._functions_attributes.get(fname)
            for attr in fn_prompt_attributes:
                if getattr(prompt, attr) is None:
                    raise ValueError("prompt is missing: {} attribute for {}".format(attr, fname))
            return func(self, *args, **kwargs)
        return wrapper
                
    def __init__(self, repo_id: str):
        """Wave llm narrator that converts raw waves into input embeddings
        for tiny wave narration llm through vllm. 
        Arguments:
            repo_id: Hugging face api from where to take weights for features mixer. 
            vllm_url: URL to hosting vllm server with train llm for wave description. """
        self._fmixer = MetaFeaturesMixer.load(repo_id)
        self._fmixer.eval()
        for p in self._fmixer.parameters():
            p.requires_grad_(False)
        
    def setup_llm(self, 
                url: str, 
                name: str, 
                mode: Literal["narrator", "core"],
                gen_cfg: Dict[str, Any] | GenerationParams={}):
        """setup model for inference"""
        assert mode in ["narrator", "core"]
        llm = OpenAI(base_url=url)
        if isinstance(gen_cfg, GenerationParams):
            gen_cfg = gen_cfg.to_vllm_kwargs()

        setattr(self, f"llm_{mode}", llm)
        setattr(self, f"llm_{mode}_url", url)
        setattr(self, f"llm_{mode}_name", name)
        setattr(self, f"llm_{mode}_generation_cfg", gen_cfg)

    def set_llm_generation_cfg(self, 
                            mode: Literal["narrator", "core"], 
                            gen_cfg: Dict[str, Any] | GenerationParams):
        """set llm generation config. """
        if isinstance(gen_cfg, GenerationParams):
            gen_cfg = gen_cfg.to_vllm_kwargs()
        assert mode in ["narrator", "core"]
        if not hasattr(self, f"llm_{mode}"):
            raise ValueError(f"first setup {mode} model with setup_llm(..., mode='narrator'). ")
        setattr(self, f"llm_{mode}_generation_cfg", gen_cfg)
    
    def get_model(self, mode: Literal["narrator", "core"]):
        """get llm OpenAi meta attributes for llm model. """
        if hasattr(self, f"llm_{mode}"):
            return {k: getattr(self, f"llm_{mode}{k}") for k in ["", "_url", "_name", "_gen_cfg"]}
        
    def encode_wave(self, raw: th.Tensor, fs: float):
        """convert input waves into fusion embeddings features. """
        return self._fmixer(raw=raw, fs=fs)

    @_pre_inference_check
    def process_wave(self, prompt: WaveDescriptionPrompt):
        """fill prompt body with wave_description. """
        llm, _, name, gen_cfg = tuple(self.get_model("narrator").values())
        embeds = self.encode_wave(prompt.wave, prompt.sampling_rate)
        embeds = encode_b64(embeds)
        response = llm.completions.create(
            model=name,
            prompt="",
            extra_body={"prompt_embedding": embeds},
            **gen_cfg
        )
        prompt.wave_description = response.choices[0].text
        return prompt

    @_pre_inference_check
    def process_description(self, prompt: WaveDescriptionPrompt):
        """fill prompt with full ecg description based on wve_description. """
        llm, _, name, gen_cfg = tuple(self.get_model("core").values())
        system_prompt = get_role("general")
        wave_description = f"""<wave_description> \n{prompt.wave_description}\n<wave_description>"""
        response = llm.chat.completion.create(
            model=name,
            messages=[system_prompt, wave_description],
            **gen_cfg
        )
        response = response.choices[0].message.content
        prompt.wave_full_description = response
        return prompt

    @_pre_inference_check
    def process_text(self, prompt: WaveDescriptionPrompt):
        """process single prompt. """

        llm, _, name, gen_cfg = tuple(self.get_model("core").values())

        
if __name__ == "__main__":
    pass