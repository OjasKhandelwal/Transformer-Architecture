import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    """
    Encoder Layer module.
    
    This implements a single encoder layer :
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        """
        Apply the encoder layer to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor of shape [batch_size, 1, seq_len, seq_len]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len, d_model]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len, seq_len]
        """
        # Apply multi-head self-attention
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        
        # Apply feed-forward network
        output = self.feed_forward(attn_output)
        
        return output, attention_weights


class Encoder(nn.Module):
    """
    Encoder module.
    
    This implements the encoder as described in the paper:
    - Stack of N identical layers
    - Each layer has multi-head self-attention and position-wise feed-forward network
    """
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Apply the encoder to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor of shape [batch_size, 1, seq_len, seq_len]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len, d_model]
            attention_weights: List of attention weights for each layer
        """
        attention_weights = []
        
        # Apply each encoder layer
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights 
