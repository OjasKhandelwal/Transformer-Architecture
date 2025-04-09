import torch
import torch.nn as nn

class Embeddings(nn.Module):
    """
    Embeddings module.
    
    This implements the embeddings:
    - Word embeddings
    - Positional encoding
    """
    
    def __init__(self, vocab_size, d_model, max_seq_length=5000, dropout=0.1):
        super(Embeddings, self).__init__()
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.position_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Apply embeddings to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len]
                
        Returns:
            output: Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Apply word embeddings
        embeddings = self.word_embeddings(x)
        
        # Add positional encoding
        embeddings = embeddings + self.position_encoding[:, :seq_len, :]
        
        # Apply layer normalization and dropout
        output = self.layer_norm(embeddings)
        output = self.dropout(output)
        
        return output 
