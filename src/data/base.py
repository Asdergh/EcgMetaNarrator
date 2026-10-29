import abc
import torch as th
from typing import List, Dict, Literal
from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import DataCollatorWithFlattening
from transformers import PreTrainedTokenizerBase
from collections.abc import Collection

def _is_struct(obj):
    return isinstance(obj, Collection) \
        and not isinstance(obj, (str, bytes, th.Tensor))

class BaseFeaturesNarrationDataset(Dataset, abc.ABC):
    """This datasts uses flattening data collate stratagy 
    to collect semantical annotations for features n extracted from 
    different backbone models. (therefore its premier task ``features interpretation``)."""
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer
        self._collater = DataCollatorWithFlattening(return_position_ids=True)

    
    def collate_fn(self, batch: List[Dict[str, th.Tensor]]):
        """constract batch handeling llm fields with falttening collation. 
            All other fields from batch would be collated in simple default collation manner. """
        txt_batch = self._collater(batch)
        non_txt_batch = {}
        for sample in batch:
            for (key, values) in sample.items():
                if key in ["input_ids", 
                        "token_type_ids", 
                        "attention_mask"]:
                    continue 
                if key not in non_txt_batch:
                    non_txt_batch[key] = []
                if not isinstance(values, th.Tensor):
                    if _is_struct(values) and len(values) > 0:
                        if _is_struct(values[0]):
                            raise ValueError("BaseTxTFeaturesInterpretationDataset works only with th.Tensor fields or with raw List[Any] data."
                                            f"Elements from field: {key} in provided batch have tyep: {type(values[0])}")
                non_txt_batch[key].append(values)

        non_txt_batch = {k: th.stack(v) 
                        if isinstance(v[0], th.Tensor) and v[0].ndim == 0 
                        else th.cat(v) if isinstance(v[0], th.Tensor)
                        else th.asarray(v) 
                        for (k, v) in non_txt_batch.items()}
        txt_batch.update(non_txt_batch)
        return txt_batch











