import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)  # 参数名和形状一一对应时直接复制


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})  # 描述 Hugging Face 分离权重如何装入本地合并参数
    for file in glob(os.path.join(path, "*.safetensors")):  # 大模型通常被拆成多个 safetensors 分片
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)  # 例如 q_proj -> qkv_proj
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")  # 合并或张量并行参数必须提供自定义加载器
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))  # 普通参数直接复制，张量并行参数则按 rank 切分
