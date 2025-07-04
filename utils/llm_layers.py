import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel, GPTJForCausalLM
from torch import Tensor
import numpy as np


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


class ICVLayer(nn.Module):

    def __init__(self, icv, lam):
        super(ICVLayer, self).__init__()
        self.icv = icv
        self.lam = lam
        self.sentence_len = 0

    def forward(self, x):
        if self.icv is not None:
            x_dtype = x.dtype
            x = x.float()
            original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            directions_all = []
            y = 0
            if x.size(1) > 1:
                self.sentence_len = 0

            for i in range(len(self.icv)): #
                schedule_rate = max(1.0 * cosine_rampdown(self.sentence_len, 50), 0.5) 
                y += self.lam[i] * schedule_rate * F.normalize(self.icv[i], dim=-1).repeat(1,x.shape[1],1)
            y = y/len(self.icv)
            x[:,-1] = (F.normalize(F.normalize(x.float(), p=2, dim=-1) + y, p=2, dim=-1) * original_norm)[:,-1]

            if x.size(1) < 2:
                self.sentence_len += 1
            return x.to(x_dtype)
        else:
            return x

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.embed_tokens
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.word_embeddings

    keywords = ["emb", "wte"]
    return find_module(model, keywords)


def get_lm_head(model: PreTrainedModel):
    keywords = ["lm_head", "embed_out"]
    return find_module(model, keywords)


def get_lm_pipeline(model: PreTrainedModel):
    model_class = model.__class__.__name__

    if model_class == "LlamaForCausalLM":
        return nn.Sequential(model.model.norm, model.lm_head)
    elif model_class == "RWForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoXForCausalLM":
        return nn.Sequential(model.gpt_neox.final_layer_norm, model.embed_out)

    # TODO: make the default case more robust
    return get_lm_head(model)


def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def get_attention_layers(model: PreTrainedModel):
    layers = get_layers(model)
    keywords = ["attention", "attn"]
    attention_layers = [find_module(layer, keywords) for layer in layers]
    return attention_layers

def add_icv_layers(model: PreTrainedModel, icv: Tensor, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    assert len(icv) == len(layers)
    for i, layer in enumerate(layers):
        # if (i < len(layers) -3) and (i > 2):
        original_mlp = find_module(layer, mlp_keywords)
        layer.mlp = nn.Sequential(original_mlp, ICVLayer(icv[i], alpha)) 

def remove_icv_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        # if (i < len(layers) -3) and (i > 2):
        icv_mlp = find_module(layer, mlp_keywords)
        layer.mlp = icv_mlp[0]