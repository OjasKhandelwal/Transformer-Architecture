import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This implements the feed-forward network described in the paper:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Apply the feed-forward network to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Apply first linear layer and ReLU activation
        ff_output = self.linear1(x)
        ff_output = torch.relu(ff_output)
        
        # Apply dropout
        ff_output = self.dropout(ff_output)
        
        # Apply second linear layer
        ff_output = self.linear2(ff_output)
        
        # Apply dropout and residual connection
        ff_output = self.dropout(ff_output)
        output = self.layer_norm(x + ff_output)
        
        return output 