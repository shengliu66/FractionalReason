import torch
from transformers import PreTrainedModel
from typing import List, Optional

from .llm_layers import get_layers


class LatentShifter:
    def __init__(
        self,
        model: PreTrainedModel,
        shift_layers: torch.Tensor,  # (batch_size)
        shift_positions: torch.Tensor,  # (batch_size)
        icv_to_shift: List[torch.Tensor],  # (N_Task, L, hidden_size)
        alpha: List[float],
    ):
        """
        Args:
            model: The model to shift latent states into
            injection_layer: the layer to inject latent states into, for each example in the batch (batch_size)
            injection_position: the position to inject latent states into, for each example in the batch (batch_size)
            icv_to_shift: the latent states to shift icv
        """

        self._model = model
        self._shift_layer = shift_layers
        self._shift_position = shift_positions
        self._icv_to_shift = icv_to_shift

        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                hidden_states = out[0] if isinstance(out, tuple) else out
                data_type = hidden_states.dtype
                combined_tasks = 0
                mask = self._injection_layer == layer_idx
                if mask.any():
                    x = hidden_states[idx_to_inject, self._shift_position[mask]].detach().float()
                    norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)     
                    for task_id, task_icv in enumerate(self._icv_to_shift):
                        icv = task_icv.to(hidden_states.device)
                        idx_to_inject = torch.arange(hidden_states.shape[0], device=hidden_states.device)[mask]
                        lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(hidden_states.device), F.cosine_similarity(x, -icv[None,None,:], dim=-1)).unsqueeze(-1)
                        combined_tasks += lambda_sim * alpha[task_id] * F.normalize(icv[task_id][mask], dim=-1).repeat(1,x.shape[1],1)

                combined_tasks = combined_tasks / (task_id + 1)
                hidden_states[idx_to_inject, self._shift_position[mask]] = (F.normalize(F.normalize(hidden_states[idx_to_inject, self._shift_position[mask]].float(),dim=-1) +  0.1 * combined_tasks, dim=-1) * norm).type(data_type)

                return out

            return inject_hidden

        for i, layer in enumerate(get_layers(self._model)):
            hook = layer.register_forward_hook(inject_hidden_hook(i))
            self._hooks.append(hook)