import torch
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftSupervisedDataset,
    make_last_position_supervised_data_module,
    ConsreftIntervention,
    LoreftIntervention,
)

class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)

# keep hidden states close by  h(x+\delta) = h(x)
class LoreftIntervention_Implicit(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        self.noise_std = kwargs.get("noise_std", 0.0)
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.index=None
        
    def forward(
        self, base, source=None, subspaces=None, record_index=False
    ):
        if record_index:
            self.index = base.mean(dim=0)
            return self.index
        
        consistency_loss = None
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(base) * self.noise_std
            base_with_noise = base + noise
        else:
            base_with_noise = base
            
        if self.training :
            
            if self.dropout_rate>0:
                base_noisy = self.dropout(base)
            else:
                epsilon = torch.randn_like(base) * self.noise_std
                base_noisy = base + epsilon
            
            R_h = self.rotate_layer(base)  # R(h)
            R_h_prime = self.rotate_layer(base_noisy)  # R(h')
            consistency_loss = torch.nn.functional.mse_loss(R_h_prime, R_h)
            
        rotated_base = self.rotate_layer(base_with_noise)
        output = base_with_noise + torch.matmul(
            (self.act_fn(self.learned_source(base_with_noise)) - rotated_base), self.rotate_layer.weight.T
        )
        
        output = self.dropout(output.to(base.dtype))
        
        if self.training and consistency_loss is not None:
            self._last_consistency_loss = consistency_loss
        else:
            self._last_consistency_loss = None
            
        return output
    
    def record_index(self, data):
        res = self.forward(data, record_index=True)
        return res

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
      
        self.learned_source.load_state_dict(state_dict, strict=False)

        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return


class LoreftIntervention_Adv_Explicit(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        # Adversarial perturbation controls
        self.adv_epsilon: float = kwargs.get("adv_epsilon", 0.001)
        self.adv_steps: int = int(kwargs.get("adv_steps", 1))
        self.adv_norm: str = str(kwargs.get("adv_norm", "l2")).lower()  # "l2" or "linf"
        # Optional starting noise for stability (set 0.0 to disable)
        self.init_noise_std: float = kwargs.get("init_noise_std", 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        if self.training and self.init_noise_std > 0:
            base_start = base + torch.randn_like(base) * self.init_noise_std
        else:
            base_start = base

        def low_rank_correction(hidden_states: torch.Tensor) -> torch.Tensor:
            rotated = self.rotate_layer(hidden_states)
            source_acts = self.act_fn(self.learned_source(hidden_states))
            return torch.matmul((source_acts - rotated), self.rotate_layer.weight.T)

        if self.training and torch.is_grad_enabled() and self.adv_epsilon > 0 and self.adv_steps > 0:
            adv = torch.zeros_like(base_start)
            step_size = self.adv_epsilon / float(self.adv_steps)

            if self.init_noise_std > 0:
                adv = adv + torch.randn_like(adv) * min(self.init_noise_std, self.adv_epsilon)

            def project_onto_ball(center: torch.Tensor, pert: torch.Tensor) -> torch.Tensor:
                diff = pert
                if self.adv_norm == "linf":
                    diff = torch.clamp(diff, min=-self.adv_epsilon, max=self.adv_epsilon)
                    return diff
                diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True) + 1e-12
                scale = torch.clamp(self.adv_epsilon / diff_norm, max=1.0)
                return diff * scale

            adv = project_onto_ball(base_start, adv)

            base_adv = (base_start + adv).detach()
            base_adv.requires_grad_(True)
            for _ in range(self.adv_steps):
                corr = low_rank_correction(base_adv)
                proxy = (corr.pow(2).sum(dim=-1)).mean()
                grad, = torch.autograd.grad(proxy, base_adv, retain_graph=False, create_graph=False)

                if self.adv_norm == "linf":
                    step = step_size * grad.sign()
                else:
                    grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-12
                    step = step_size * grad / grad_norm

                base_adv = base_adv.detach() + step
                
                adv = base_adv - base_start
                adv = project_onto_ball(base_start, adv)
                base_adv = (base_start + adv).detach()
                base_adv.requires_grad_(True)

            hidden_for_edit = base_adv.detach()
        else:
            hidden_for_edit = base_start

        rotated_hidden = self.rotate_layer(hidden_for_edit)
        output = hidden_for_edit + torch.matmul(
            (self.act_fn(self.learned_source(hidden_for_edit)) - rotated_hidden), self.rotate_layer.weight.T
        )

        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return

# keep Output distribution close by  f(h(x+\delta)) = f(h(x))
class LoreftIntervention_Explicit(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)

    Explicit variant: applies LoReFT with optional noise for stability.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        # Optional starting noise for stability (set 0.0 to disable)
        self.init_noise_std: float = kwargs.get("init_noise_std", 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        if self.training and self.init_noise_std > 0:
            hidden_for_edit = base + torch.randn_like(base) * self.init_noise_std
        else:
            hidden_for_edit = base

        rotated_hidden = self.rotate_layer(hidden_for_edit)
        output = hidden_for_edit + torch.matmul(
            (self.act_fn(self.learned_source(hidden_for_edit)) - rotated_hidden), self.rotate_layer.weight.T
        )

        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
    
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.learned_source.load_state_dict(state_dict, strict=False)

        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return