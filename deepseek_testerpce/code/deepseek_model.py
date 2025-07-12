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
    