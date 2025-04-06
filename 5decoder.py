import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    """
    Decoder Layer module.
    
    This implements a single decoder layer as described in the paper:
    - Masked multi-head self-attention
    - Multi-head attention over encoder output
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Multi-head attention over encoder output
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        """
        Apply the decoder layer to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            self_mask: Optional mask for self-attention of shape [batch_size, 1, seq_len, seq_len]
            cross_mask: Optional mask for cross-attention of shape [batch_size, 1, seq_len, seq_len]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len, d_model]
            self_attention_weights: Self-attention weights
            cross_attention_weights: Cross-attention weights
        """
        # Apply masked multi-head self-attention
        self_attn_output, self_attention_weights = self.self_attention(x, x, x, self_mask)
        
        # Apply multi-head attention over encoder output
        cross_attn_output, cross_attention_weights = self.cross_attention(
            self_attn_output, enc_output, enc_output, cross_mask
        )
        
        # Apply feed-forward network
        output = self.feed_forward(cross_attn_output)
        
        return output, self_attention_weights, cross_attention_weights


class Decoder(nn.Module):
    """
    Decoder module.
    
    This implements the decoder as described in the paper:
    - Stack of N identical layers
    - Each layer has masked multi-head self-attention, multi-head attention over encoder output,
      and position-wise feed-forward network
    """
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        """
        Apply the decoder to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output tensor of shape [batch_size, seq_len, d_model]
            self_mask: Optional mask for self-attention of shape [batch_size, 1, seq_len, seq_len]
            cross_mask: Optional mask for cross-attention of shape [batch_size, 1, seq_len, seq_len]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len, d_model]
            self_attention_weights: List of self-attention weights for each layer
            cross_attention_weights: List of cross-attention weights for each layer
        """
        self_attention_weights = []
        cross_attention_weights = []
        
        # Apply each decoder layer
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, enc_output, self_mask, cross_mask
            )
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)
        
        return x, self_attention_weights, cross_attention_weights 