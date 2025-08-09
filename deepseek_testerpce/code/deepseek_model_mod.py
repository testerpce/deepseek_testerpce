"""
Modified DeepSeek model with improved Mixture‑of‑Experts (MoE) layer.

This file shows how to incorporate weighted gating, improved load balancing, optional
distributed load balancing, shared experts and quantization support.  It is not a
complete, runnable module but illustrates the changes you would need to make to
your existing `deepseek_model.py` file.  See inline comments for details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple


class DeepSeekConfig:
    """Configuration for Deepseek model with extended MoE options."""
    # original fields omitted for brevity …
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embed: int = 512
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True

    use_mla: bool = True
    mla_kv_heads: int = 4
    mla_q_lora_rank: int = 32
    mla_kv_lora_rank: int = 16

    use_moe: bool = True
    moe_num_experts: int = 6
    top_k_experts: int = 2
    moe_variable_experts: int = 4
    moe_expert_capacity: float = 1.25
    moe_aux_loss_coeff: float = 0.1
    # new configuration options
    moe_shared_experts_number: int = 0  # number of shared experts always used
    moe_capacity_factor: float = 1.0    # capacity factor for limiting per‑expert tokens

    number_of_tokens: int = 2
    use_quantization: bool = False
    quantization_bits: int = 8


class MOEExpert(nn.Module):
    """Single expert feedforward network."""
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.linear2 = nn.Linear(config.n_embed * 4, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MixtureOfExperts(nn.Module):
    """
    Improved MoE layer with weighted routing, load balancing and optional shared experts.

    Differences from the original implementation:

      * Uses softmax probabilities and top‑k selection to weight expert outputs rather than
        simply averaging across experts.
      * Computes auxiliary loss based on density/usage and importance, similar to
        Switch/DeepSeek MoE implementations.
      * Optionally aggregates usage and importance across distributed processes.
      * Supports shared experts that are always invoked regardless of routing.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.moe_num_experts
        self.top_k_experts = config.top_k_experts
        self.variable_experts = config.moe_variable_experts
        if self.top_k_experts > self.num_experts:
            raise ValueError("Top K experts cannot be greater than total experts.")
        self.fixed_experts = self.num_experts - self.variable_experts
        if self.fixed_experts < 0:
            raise ValueError("Number of variable experts cannot be greater than total experts.")
        # create experts
        self.experts = nn.ModuleList([MOEExpert(config) for _ in range(self.num_experts)])
        # router for variable experts only
        self.router = nn.Linear(config.n_embed, self.variable_experts, bias=False)
        # layer norm
        self.layer_norm = nn.LayerNorm(config.n_embed, bias=config.bias)
        # shared experts if configured
        self.shared_experts_number = config.moe_shared_experts_number
        if self.shared_experts_number > 0:
            self.shared_experts = nn.ModuleList([
                MOEExpert(config) for _ in range(self.shared_experts_number)
            ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_embed = x.shape
        # router probabilities over variable experts
        router_logits = self.router(x)  # (b, s, variable_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        # select top‑k variable experts per token
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k_experts, dim=-1)
        # normalise top‑k weights so they sum to one across selected experts
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        # map variable expert indices to global expert indices (after fixed experts)
        adjusted_topk_indices = topk_indices + self.fixed_experts  # (b, s, top_k)
        # compute expert usage mask for auxiliary loss
        expert_mask = F.one_hot(adjusted_topk_indices, num_classes=self.num_experts).sum(dim=2)
        expert_count = expert_mask.sum(dim=(0, 1)).float()  # (num_experts,)
        # aggregate expert_count across distributed processes if needed
        if dist.is_available() and dist.is_initialized():
            expert_count_all = expert_count.clone()
            dist.all_reduce(expert_count_all)
            expert_count = expert_count_all
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        usage = expert_count / float(batch_size * seq_len * world_size)
        density = router_probs.mean(dim=(0, 1))  # (variable_experts,)
        # pad density for fixed experts (assume uniform usage for fixed experts)
        if self.fixed_experts > 0:
            fixed_density = torch.zeros(self.fixed_experts, device=density.device, dtype=density.dtype)
            density_full = torch.cat((fixed_density, density), dim=0)
        else:
            density_full = density
        balance_loss = (density_full * usage).sum() * self.num_experts
        importance = router_probs.sum(dim=(0, 1))  # (variable_experts,)
        # pad importance for fixed experts
        if self.fixed_experts > 0:
            fixed_importance = torch.zeros(self.fixed_experts, device=importance.device, dtype=importance.dtype)
            importance_full = torch.cat((fixed_importance, importance), dim=0)
        else:
            importance_full = importance
        if dist.is_available() and dist.is_initialized():
            importance_all = importance_full.clone()
            dist.all_reduce(importance_all)
            importance_full = importance_all
        important_loss = (importance_full ** 2).mean()
        aux_loss = balance_loss + important_loss
        # fixed expert outputs
        if self.fixed_experts > 0:
            fixed_outputs = torch.stack([
                self.experts[i](x) for i in range(self.fixed_experts)
            ], dim=-1)  # (b, s, n_embed, fixed_experts)
        else:
            fixed_outputs = None
        # variable expert outputs weighted by router weights
        variable_outputs = []
        for k in range(self.top_k_experts):
            expert_ids = adjusted_topk_indices[..., k]  # (b, s)
            weights = topk_probs[..., k]  # (b, s)
            # gather outputs for each expert selected by each batch/seq position
            expert_out = torch.stack([
                self.experts[idx](x[b]) for b, idx in enumerate(expert_ids)
            ], dim=0)  # (b, s, n_embed)
            variable_outputs.append(expert_out)
        variable_outputs = torch.stack(variable_outputs, dim=-1)  # (b, s, n_embed, top_k)
        weights_expanded = topk_probs.unsqueeze(-2)  # (b, s, 1, top_k)
        weighted_variable = variable_outputs * weights_expanded
        # concatenate and sum across experts
        if fixed_outputs is not None:
            combined_outputs = torch.cat((fixed_outputs, weighted_variable), dim=-1)
        else:
            combined_outputs = weighted_variable
        combined_outputs = combined_outputs.sum(dim=-1)  # (b, s, n_embed)
        # add shared expert outputs
        if self.shared_experts_number > 0:
            shared_out = sum(expert(x) for expert in self.shared_experts)  # (b, s, n_embed)
            combined_outputs = combined_outputs + shared_out
        # quantization
        if self.config.use_quantization:
            scale = 2 ** self.config.quantization_bits - 1
            combined_outputs = torch.round(combined_outputs * scale) / scale
        combined_outputs = self.layer_norm(combined_outputs)
        return combined_outputs, aux_loss

    def _compute_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Deprecated auxiliary loss function.  The new auxiliary loss is computed inside
        the forward method.  This method is retained for backward compatibility.
        """
        return router_logits.new_tensor(0.0)
