"""
Basic Deepseek architecture to test for model training and testing.
Implementing:
    Multihead Latent Attention (MLA) - not done yet
    Mixture of Experts (MOE) - not done yet
    Multi-token prediction
    Quantization support
    Rotary Position embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    """Configuration for Deepseek model initially optimized for children's stories."""
    vocab_size: int = 50257 #GPT2 vocab size
    n_layer: int = 6        #number of layers in the Decoder style transformer
    n_head: int = 8         #number of attention heads
    n_embed: int = 512      #Embedding dimension
    block_size: int = 1024  #Context window (I think it is for training)
    dropout: float = 0.1    #Fraction of dropout during training
    bias: bool = True       #Use bias in the linear layers

    # MLA Multi head Latent Attention config
    use_mla: bool = True # Enable Multihead Latent Attention
    mla_kv_heads: int = 4 # Number of heads for MLA key and value
    mla_q_lora_rank: int = 32 # LoRA rank for MLA query
    mla_kv_lora_rank: int = 16 # LoRA rank for MLA key value

    # MOE Mixture of Experts config
    use_moe: bool = True
    moe_num_experts: int = 6 # Number of experts in MOE
    top_k_experts: int = 2 # Number of top experts to use per token among variable experts
    moe_variable_experts: int = 4 # Number of Variable experts to use per token. Fixed experts is moe_num_experts - moe_variable_experts 
    moe_expert_capacity: float = 1.25 # Capacity factor for experts
    moe_aux_loss_coeff: float = 0.1 # Coefficient for auxiliary loss

    #multi-token prediction
    number_of_tokens: int = 2 # Number of tokens to predict in one forward pass

    #Quantization support
    use_quantization: bool = False
    quantization_bits: int = 8



class RoPEPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) for Transformer models."""
    
    def __init__(self, d_model: int, max_seq_len: int = 1024, base: float = 10000.0):
        super(RoPEPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        #Precompute frequency matrix
        inv_frequency = 1.0 / (self.base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_frequency", inv_frequency)

        #Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0

    def compute_cos_sin(self, seq_len: int, device: torch.device):
        """Compute cosine and sine values for given sequence length."""
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        if seq_len > self._cached_seq_len or self._cached_cos is None:
            #Create position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_frequency.dtype)

            #Compute frequencies
            freqs = torch.outer(t, self.inv_frequency)

            #Create rotation matrix components
            cos_vals = torch.cos(freqs)
            sin_vals = torch.sin(freqs)

            #Cache the results
            self._cached_cos = cos_vals
            self._cached_sin = sin_vals
            self._cached_seq_len = seq_len
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
    
    def apply_rope(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply RoPE to input tensor."""
        batch_size, seq_len, n_heads, d_head = x.shape

        #Get cos sine values
        cos, sin = self.compute_cos_sin(seq_len, x.device)

        #Handle position ids if provided
        if position_ids is not None:
            if position_ids.shape[0] != batch_size or position_ids.shape[1] != seq_len:
                raise ValueError("Position ids shape does not match input tensor shape.")
            cos = cos[position_ids]
            sin = sin[position_ids]

        #Reshape cos and sin to match input tensor
        cos = cos.unsqueeze(0).unsqueeze(2)  # Shape: (1, 1, seq_len, d_head/2)
        sin = sin.unsqueeze(0).unsqueeze(2) # Shape: (1, 1, seq_len, d_head/2)

        #Split x into even and odd indices
        x1 = x[..., ::2] # Even indices
        x2 = x[..., 1::2] # Odd indices

        #Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        #Recombine the rotated parts
        rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)
        
        return rotated_x
    


class MultiHeadLatentAttention(nn.Module):
    """
    Multihead Latent Attention (MLA) module. Deepseek's efficient attention mechanism.
    Uses shared key value heads with Lora style projections for efficiency.
    """
    def __init__(self, config: DeepSeekConfig):
        super(MultiHeadLatentAttention, self).__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = self.n_embed // self.n_head
        self.kv_heads = config.mla_kv_heads
        self.kv_head_dim = self.head_dim
        assert self.n_head % self.kv_heads == 0, "Number of heads must be divisible by kv_heads"
        assert self.n_embed % self.n_head == 0, "Embedding dimension must be divisible by number of heads"
        

        # Query projection with LoRA
        self.q_a_proj = nn.Linear(self.n_embed, config.mla_q_lora_rank, bias=False)
        self.q_b_proj = nn.Linear(config.mla_q_lora_rank, self.n_embed, bias=False)
        
        # Key and Value projections with shared heads
        self.kv_a_proj = nn.Linear(self.n_embed, config.mla_kv_lora_rank, bias=False)
        self.kv_b_proj = nn.Linear(config.mla_kv_lora_rank, self.kv_heads * self.head_dim * 2, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.n_embed, self.n_embed, bias=config.bias)

        # Rope for positional encoding
        self.rope = RoPEPositionalEncoding(self.head_dim)

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Multihead Latent Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            attention_mask: Optional tensor of shape (batch_size, seq_len)
              for masking out padded tokens and autoregressive attention
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to query, key, value
        q_latent = self.q_a_proj(x) # dim (batch_size, seq_len, mla_q_lora_rank)
        q = self.q_b_proj(q_latent) # dim (batch_size, seq_len, n_embed)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim)

        #Project input to key and value shared heads
        # Note: In MLA, we use shared key and value heads with LoRA projections
        kv_latent = self.kv_a_proj(x) # dim (batch_size, seq_len, mla_kv_lora_rank)
        kv_shared = self.kv_b_proj(kv_latent) # dim (batch_size, seq_len, kv_heads * head_dim * 2)
        kv_shared = kv_shared.view(batch_size, seq_len, self.kv_heads, 2 * self.head_dim)
        k, v = kv_shared.split(self.head_dim, dim=-1) # dim (batch_size, seq_len, kv_heads, head_dim)

        # Apply RoPE to query, key before expansion
        # Understand: Why not value ?
        q = self.rope.apply_rope(q)
        k = self.rope.apply_rope(k)

        # Expand key value to match query heads
        k = k.repeat_interleave(self.n_head // self.kv_heads, dim=2)
        v = v.repeat_interleave(self.n_head // self.kv_heads, dim=2)

        #Transpose for attention computation
        q = q.transpose(1, 2) # dim (batch_size, n_head, seq_len, head_dim)
        k = k.transpose(1, 2) # dim (batch_size, n_head, seq_len, head_dim)
        v = v.transpose(1, 2) # dim (batch_size, n_head, seq_len, head_dim)

        #Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # dim (batch_size, n_head, seq_len, seq_len)
        if attention_mask is None:
            # Apply attention mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill_(causal_mask, float('-inf'))
        else:
            # Apply provided attention mask
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores + attention_mask  # (batch_size, 1, seq_len, seq_len)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v) # dim (batch_size, n_head, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embed) # dim (batch_size, seq_len, n_embed)
        
        # Project output back to embedding space
        out = self.out_proj(out) # dim (batch_size, seq_len, n_embed) 

        return out
    

class MOEExpert(nn.Module):
    """
    Single Mixture of Experts (MOE) expert layer.
    Each expert has its own feedforward network.
    """
    def __init__(self, config: DeepSeekConfig):
        super(MOE.Expert, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.linear2 = nn.Linear(config.n_embed * 4, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x) # dim (batch_size, seq_len, n_embed * 4)
        x = F.gelu(x) # Activation function dim (batch_size, seq_len, n_embed * 4)
        x = self.linear2(x) # dim (batch_size, seq_len, n_embed)
        x = self.dropout(x) # Apply dropout dim (batch_size, seq_len, n_embed)
        return x




class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MOE) layer.
    Implements a mixture of experts with fixed and variable experts.
    """
    def __init__(self, config: DeepSeekConfig):
        super(MixtureOfExperts, self).__init__()
        self.config = config
        self.num_experts = config.moe_num_experts
        self.top_k_experts = config.top_k_experts
        if self.top_k_experts > self.num_experts:
            raise ValueError("Top K experts cannot be greater than total experts.")
        if self.top_k_experts <= 0:
            raise ValueError("Top K experts must be greater than 0.")
        # Variable experts are the ones that can change based on input
        # Fixed experts are the ones that are always used
        self.variable_experts = config.moe_variable_experts
        self.fixed_experts = self.num_experts - self.variable_experts
        if self.fixed_experts < 0:
            raise ValueError("Number of variable experts cannot be greater than total experts.")
        if self.variable_experts <= 0:
            raise ValueError("Number of variable experts must be greater than 0.")
        self.expert_capacity = config.moe_expert_capacity

        # Create experts
        self.experts = nn.ModuleList([MOEExpert(config) for _ in range(self.num_experts)])

        # Router for selecting experts
        self.router = nn.Linear(config.n_embed, self.variable_experts, bias=False)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(config.n_embed, bias = config.bias)

        # Auxiliary loss coefficient
        self.aux_loss_coeff = config.moe_aux_loss_coeff

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Mixture of Experts.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
            Auxiliary loss for expert routing
        """
        batch_size, seq_len, n_embed = x.shape

        # Route inputs to experts
        router_logits = self.router(x)  # Shape: (batch_size, seq_len, num_variable_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # Shape: (batch_size, seq_len, num_variable_experts)

        # Select top-k variable experts
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k_experts, dim=-1)  # Shape: (batch_size, seq_len, top_k_experts)

        # Adjust indices for variable experts (offset by fixed_experts)
        adjusted_topk_indices = topk_indices + self.fixed_experts  # Shape: (batch_size, seq_len, top_k_experts)

        # Gather outputs from fixed experts
        fixed_expert_outputs = torch.stack(
            [self.experts[i](x) for i in range(self.fixed_experts)], dim=-1
        )  # Shape: (batch_size, seq_len, n_embed, fixed_experts)

        # Gather outputs from top-k variable experts
        variable_expert_outputs = []
        for i in range(self.top_k_experts):
            expert_idx = adjusted_topk_indices[..., i]  # Shape: (batch_size, seq_len)
            expert_output = torch.stack(
                [self.experts[idx](x[b]) for b, idx in enumerate(expert_idx)], dim=0
            )  # Shape: (batch_size, seq_len, n_embed)
            variable_expert_outputs.append(expert_output)

        variable_expert_outputs = torch.stack(variable_expert_outputs, dim=-1)  # Shape: (batch_size, seq_len, n_embed, top_k_experts)

        # Combine fixed and variable expert outputs
        combined_outputs = torch.cat((fixed_expert_outputs, variable_expert_outputs), dim=-1)  # Shape: (batch_size, seq_len, n_embed, fixed_experts + top_k_experts)

        # Compute weighted sum across all experts
        # all_expert_weights = torch.cat(
        #     (torch.ones(batch_size, seq_len, self.fixed_experts, device=x.device), topk_probs), dim=-1
        # )  # Shape: (batch_size, seq_len, fixed_experts + top_k_experts)
        # all_expert_weights = all_expert_weights.unsqueeze(-2)  # Shape: (batch_size, seq_len, 1, fixed_experts + top_k_experts)

        # combined_outputs = torch.sum(all_expert_outputs * all_expert_weights, dim=-1)  # Shape: (batch_size, seq_len, n_embed)

        # Average across experts
        combined_outputs = combined_outputs.mean(dim=-1)  # Average across experts
        
        # Quantization support
        if self.config.use_quantization:
            # Apply quantization if enabled
            scale = 2 ** self.config.quantization_bits - 1
            combined_outputs = torch.round(combined_outputs * scale) / scale
        # Apply layer normalization
        combined_outputs = self.layer_norm(combined_outputs)

        

        return combined_outputs, router_logits

    def _compute_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss for expert routing.
        
        Args:
            router_logits: Logits from the router of shape (batch_size, seq_len, num_variable_experts)
        
        Returns:
            Auxiliary loss tensor
        """
        # Compute auxiliary loss as mean of router logits
        router_probs = F.softmax(router_logits, dim=-1)
        mean_expert_usage = router_probs.mean(dim=[0,1])  # Mean across batch and sequence length

        target_usage = 1.0 / self.variable_experts  # Target usage per expert

        aux_loss = F.mse_loss(mean_expert_usage, target_usage)  # Mean squared error loss
        return aux_loss * self.aux_loss_coeff  # Scale by auxiliary loss coefficient
    

class DeepSeekBlock(nn.Module):
    """
    A Deepseek block consisting of Multihead Latent Attention and MOE layers
    """
    def __init__(self, config: DeepSeekConfig):
        super(DeepSeekBlock, self).__init__()
        self.config = config

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embed, bias=config.bias)

        if self.config.use_mla:
            # Multihead Latent Attention
            self.attn = MultiHeadLatentAttention(config)
        else:
            # Use standard MultiHead Attention if MLA is not enabled
            self.attn = nn.MultiheadAttention(embed_dim=config.n_embed, 
                                             num_heads=config.n_head, 
                                             dropout=config.dropout, 
                                             bias=config.bias,
                                             batch_first=True)
        
        # Mixture of Experts
        if self.config.use_moe:
            self.moe = MixtureOfExperts(config)
        else:
            # Use a simple feedforward layer if MOE is not enabled
            self.moe = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Deepseek block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            attention_mask: Optional tensor of shape (batch_size, seq_len)
              for masking out padded tokens and autoregressive attention
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
            Auxiliary loss from MOE if applicable
        """

        # Apply Multihead Latent Attention
        if self.config.use_mla:
            x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        else:
            attn_output, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=attention_mask)
            x = x + attn_output
        
        # Apply Mixture of Experts
        if self.config.use_moe:
            moe_output, router_logits = self.moe(self.ln2(x))
            x = x + moe_output
            return x, router_logits
        else:
            x = x + self.moe(self.ln2(x))
            return x, None


class MultiTokenPrediction(nn.Module):
    """
    Multi-token prediction layer for Deepseek model.
    Predicts multiple tokens in one forward pass.
    """
    def __init__(self, config: DeepSeekConfig):
        super(MultiTokenPrediction, self).__init__()
        self.config = config
        self.num_tokens = config.number_of_tokens
        
        # Combine all predictors into a single linear layer with expanded output size
        self.predictor = nn.Linear(config.n_embed, config.vocab_size * self.num_tokens, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-token prediction.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, n_embed)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size * num_tokens)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Predict all tokens in one pass
        logits = self.predictor(hidden_states)  # Shape: (batch_size, seq_len, vocab_size * num_tokens)

        # Reshape to separate predictions for each token
        logits = logits.view(batch_size, seq_len, self.num_tokens, self.config.vocab_size)  # Shape: (batch_size, seq_len, num_tokens, vocab_size)
        
        return logits
    

class DeepSeekModel(nn.Module):
    """
    Deepseek model architecture.
    Implements a stack of Deepseek blocks with multi-token prediction.
    """
    def __init__(self, config: DeepSeekConfig):
        super(DeepSeekModel, self).__init__()
        self.config = config
        self.n_layer = config.n_layer

        # Input embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.n_embed)

        # Positional encoding
        self.positional_encoding = RoPEPositionalEncoding(config.n_embed, max_seq_len=config.block_size)

        # Stack of Deepseek blocks
        self.blocks = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.n_layer)])

        # Multi-token prediction layer
        self.multi_token_predictor = MultiTokenPrediction(config)

    # def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     """
    #     Forward pass for Deepseek model.
        
    #     Args:
    #         input_ids: Input tensor of shape (batch_size, seq_len)
    #         attention_mask: Optional tensor of shape (batch_size, seq_len)
    #           for masking out padded tokens and autoregressive attention
        
    #     Returns:
    #         Output tensor of shape (batch_size, seq_len, vocab_size * num_tokens)
    #         Auxiliary loss from MOE if applicable
    #     """
    def build_mtp_targets(self, input_ids: torch.Tensor,
                      K: int,
                      ignore_index: int = -100) -> torch.Tensor:
        """
        Build targets for multi-token prediction.
        input_ids: (B, S)
        returns:   (B, S, K) where [b,t,k] = token at t+k+1 or ignore_index
        """
        B, S = input_ids.shape
        device = input_ids.device
        dtype  = input_ids.dtype

        # start with all ignore
        targets = torch.full((B, S, K), ignore_index, dtype=dtype, device=device)

        # offsets 1..K
        offsets = torch.arange(1, K+1, device=device)  # (K,)

        # compute future absolute positions for each t
        pos = torch.arange(S, device=device).unsqueeze(-1) + offsets  # (S,K)

        # mask of where a valid future token exists
        valid = pos < S  # (S,K) bool

        # For each k, copy the shifted slice where valid
        # We loop over K (small) but stay in PyTorch ops; negligible overhead.
        for k in range(K):
            if valid[:, k].any():
                upto = S - (k + 1)            # number of valid positions for this k
                targets[:, :upto, k] = input_ids[:, k+1:]  # shift copy

        return targets  # (B,S,K)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor]=None,
                attention_mask: Optional[torch.Tensor]=None):
        """
            Forward pass for Deepseek model.
            
            Args:
                input_ids: Input tensor of shape (batch_size, seq_len)
                attention_mask: Optional tensor of shape (batch_size, seq_len)
                for masking out padded tokens and autoregressive attention
                targets: Optional tensor of shape (batch_size, seq_len)
                for computing loss during training. If None, returns logits only.
            
            Returns:
                Output tensor of shape (batch_size, seq_len, vocab_size * num_tokens)
                Auxiliary loss from MOE if applicable
        """ 
        B, S = input_ids.shape
        x = self.embedding(input_ids) #dim (Batch, Seq_len, n_embed)
        aux_loss = 0.0

        for block in self.blocks:
            x, router_logits = block(x, attention_mask) # dim (Batch, Seq_len, n_embed)
            # If using MOE, accumulate auxiliary loss
            if router_logits is not None:
                aux_loss += block.moe._compute_aux_loss(router_logits)

        x = self.ln_f(x)                       # add final LN

        logits = self.multi_token_predictor(x) # dim (Batch, Seq_len, num_tokens, vocab_size)

        if targets is None:
            return logits, None               # inference path

        # ------ build masked targets (as shown earlier) ------
        masked_tgt = self.build_mtp_targets(targets, self.config.number_of_tokens) # dim masked_tgt (Batch, Seq_len, num_tokens)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            masked_tgt.view(-1),
            ignore_index=-100
        )
        loss = loss + self.config.moe_aux_loss_coeff * aux_loss
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,          # == 1.0  → disabled
    ):
        """
        Autoregressive generation for a model whose head returns
        logits shape (Batch_size, Seq_len, num_tokens, vocab_size).  We use only horizon k = 0
        (the “next-token” distribution) each step.

        Args
        ----
        input_ids : (B, Seq_len)  initial context
        max_new_tokens : how many tokens to append
        temperature : softmax temperature (>0)
        top_k  : keep only the k highest-logit tokens (0 = disabled)
        top_p  : nucleus sampling threshold in cumulative prob (1.0 = disabled)

        Returns
        -------
        (B, S₀ + max_new_tokens) tensor of generated token ids
        """
        device = input_ids.device
        generated = input_ids

        for _ in range(max_new_tokens):
            # Keep only the last block_size tokens to honour the context window
            if generated.size(1) > self.config.block_size:
                context = generated[:, -self.config.block_size :]
            else:
                context = generated

            # Forward pass – logits  (B, S_ctx, K, V)
            logits, _ = self(context)                 # your forward returns (logits, aux)
            next_logits = logits[:, -1, 0, :]         # use horizon k = 0 at last position  → (B, V)

            # Temperature
            next_logits = next_logits / temperature

            # --------------  top-k  ------------------
            if top_k > 0:
                values, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                kth_best = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < kth_best, torch.full_like(next_logits, -float("Inf")), next_logits
                )

            # --------------  nucleus (top-p / min_p) --------------
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # mask tokens with cumulative prob above p
                sorted_mask = cumprobs > top_p
                # shift mask right to include at least one token
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                # scatter back to original index order
                idx_mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                next_logits = next_logits.masked_fill(idx_mask, -float("Inf"))

            # Softmax → sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B,1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        return generated



        





        
        






    