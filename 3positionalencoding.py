import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module.
    
    This implements the positional encoding described in the paper:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1)]
        
        # Apply dropout
        output = self.dropout(x)
        
        return output 