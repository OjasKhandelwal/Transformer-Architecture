import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    This implements the attention mechanism described in the paper:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Compute attention weights and apply them to values.
        
        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len_q, d_k]
            key: Key tensor of shape [batch_size, num_heads, seq_len_k, d_k]
            value: Value tensor of shape [batch_size, num_heads, seq_len_v, d_v]
            mask: Optional mask tensor of shape [batch_size, 1, seq_len_q, seq_len_k]
                
        Returns:
            output: Weighted sum of values of shape [batch_size, num_heads, seq_len_q, d_v]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    This implements the multi-head attention mechanism described in the paper:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor of shape [batch_size, 1, seq_len_q, seq_len_k]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len_q, d_model]
            attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(query + output)
        
        return output, attention_weights 