import timm
from timm.data import resolve_data_config
# # print(timm.list_models())
# model = timm.create_model("samvit_base_patch16", features_only=True)

# from pprint import pprint
# from tqdm import tqdm

# _model_params = {}
# for model in tqdm(timm.list_models()):
#     model = timm.create_model(model)
#     _model_params[model] = sum(p.numel() for p in model.parameters())
# pprint(_model_params)


from datasets import load_dataset

ds = load_dataset("spaicom-lab/semasia-model-registry")
df = ds["train"].to_pandas()
df = df[["model_name", "num_parameters", "head_type", "latent_dim"]]
df = df[df["num_parameters"] < 10000000]
df.to_csv("models_info.csv")
# Запрос количества параметров по имени модели
names = df["model_name"].to_list()



# model = timm.create_model("vit_betwixt_patch16_reg1_gap_256.sbb_in1k")
# print(model)