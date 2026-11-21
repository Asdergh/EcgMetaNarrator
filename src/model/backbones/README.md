This module is a minimalist set of tools for training ECG LLM fusion backbone models. It allows you to customize the forward pass of Hugging Face timm models and train them to solve the task of multi-binary classification on labeled ECG datasets, followed by LLM fusion training.

There are two main modes of using the module:
	. Training backbone models
	. Training an LLM model in the autoregressive fusion format, i.e., using the outputs of backbone models as input_embeds during LLM training

### Training Backbone Models.

Each registered model is essentially a timm model with a custom inference variant. All models inherit from the ```BaseFeaturesModel``` class, which is responsible for I/O integration with Hugging Face. This module has three useful functions:

```python
BaseFeaturesModel.set_repo_id(repo_id: str) # allows you to specify the path to the Hugging Face repository the model should work with
BaseFeaturesModel.save2hf(fname: str) # save the model's state_dict to the specified repository with the specified file name
BaseFeaturesModel.load(fname: str) # download a version of the model weights
BaseFeaturesModel.list_versions() # get a list of all available versions of the model weights in the repository
featuresMyBackbone.init_model() # create the model either from a state_dict or via timm with named arguments; convenient to call from the __init__ method of custom architectures
```

When forming a custom version of a backbone model, the following rules must be taken into account:

. Each model must be registered in the common model pool and have the following fields:
```python
@register_backbone()
class featuresCustomModelName(BaseFeaturesModel):
	_name: str # the model name by which BaseFeaturesModel will create a directory in the repository
	_embedding_dim: int # the dimensionality of the model's final latent activation
	_acronim: str # the exact model name in the HF registry that timm.create_model can find
```
. Each model must return an object:
```python
@dataclass
class featuresOutput:
	feature_maps: Optional[Tuple[th.Tensor]]=None # feature_maps similar to AutoBackbone out_features_maps
	last_hidden_state: Optional[th.Tensor]=None # last_hidden_state - the final latent whose dimensionality must strictly correspond to the model's _embedding_dim
	prediction: Optional[th.Tensor]=None # multi-binary classification vector whose dimensionality must correspond to the model's num_classes attribute
```

. For each model, its own ```forward_impl(self, dense: th.Tensor)``` function must be implemented, accepting the ```dense``` argument - the spectrogram of the input signal.

During model initialization, you can immediately specify a state_dict or pass a set of named arguments corresponding to the signature of the [timm.create_model(...)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_factory.py) method.

#### Usage Example

. **Model registration:**
```python

from backbones import register_backbone, featuresOutput, BaseFeaturesModel

@register_backbone() # registration of the module is based on its _name field
class featuresMyBackbone(BaseFeaturesModel):
	_name: str="my_model"
	_embedding_dim: int=328
	_acronim: str="timm-model-name"
	
	def __init__(self, num_classes: int=12):
		super(featuresMyBackbone, self).__init__()
		self.init_model(num_classes=num_classes) 
	
	def forward_inmpl(self, dense: th.Tensor, **kwargs):
		"""forward implementation"""
		dense = self._tf(dense) # self._tf - is determined based on the model arguments by timm inside BaseFeaturesModel
		...
		return featuresOutput(features_maps,
							last_hidden_state,
							prediction)


# TimeBins = TimeInSec * fs
# raw: th.Tensor[B, C, TimeBins]
# fs: float - sampling rate
raw, fs = ...
dense = spectrogram(raw, fs)
model = featuresMyBackbone()

output = model(raw=raw, 
				fs=fs, 
				dense=(dense | None), 
				**kwargs)
# it is important to remember that despite the fact that forward_impl itself works with dense, you can simply pass raw and fs to the forward of BaseFeaturesModel; it will automatically compute dense from raw and fs, or use the dense passed by the user if the latter is not None. 
```

.**Model training:**
```python 

# first register the model with register_backbone()
# then you can simple train backbone on prepared dataset from ``data`` module by 
# using train(...) function. 
from backbones import train

train(hf_repo: str # Hugging Face repository id
	root: str # dataset root path,
	loader_name: str # dataset loader name from ``data`` module registry
	model_name: str # model name from ``backbones`` module registry
	epochs: int, # total epochs count
	batch_size: int # dataset batch size 
	shuffle: bool # shuffle dataset
	accelerator: str # training device
	devices: int # number of devices
	**kwargs) # for kwargs you can check BackboneTrianingContainer and loaders signature from ``backbones`` and ``data``
# all moedl weights + num_classes attribute will be saved in specified hf_repo
```
. You can list all features that you can monitor on validation using: `MultiBCECriterion.list_eval_metrics()`  
. You can list all possible metric_accumulators using: `MultiBCECriterion.list_metrics()`  
. **Note**: be sure that the eval_metrics you selected as monitored are included in metrics_list.  
. **Note**: num_classes must be specified to match the number of labels for the chosen loader (i.e., the dimensionality of the multi-binary classification vector)