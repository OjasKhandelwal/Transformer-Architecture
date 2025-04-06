import torch
import torch.nn as nn
import math

def create_padding_mask(seq):
    """
    Create padding mask for the input sequence.
    
    Args:
        seq: Input tensor of shape [batch_size, seq_len]
            
    Returns:
        mask: Mask tensor of shape [batch_size, 1, 1, seq_len]
    """
    # Create mask for padding tokens (0)
    mask = (seq == 0).unsqueeze(1).unsqueeze(2)
    
    return mask

def create_look_ahead_mask(seq_len):
    """
    Create look-ahead mask for the target sequence.
    
    Args:
        seq_len: Length of the target sequence
            
    Returns:
        mask: Mask tensor of shape [seq_len, seq_len]
    """
    # Create upper triangular matrix
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    
    # Convert to binary mask (0 for masked positions, 1 for unmasked positions)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    
    return mask

def create_masks(src, tgt):
    """
    Create all necessary masks for the Transformer model.
    
    Args:
        src: Source tensor of shape [batch_size, src_seq_len]
        tgt: Target tensor of shape [batch_size, tgt_seq_len]
            
    Returns:
        src_mask: Mask for source of shape [batch_size, 1, 1, src_seq_len]
        tgt_mask: Mask for target of shape [batch_size, 1, tgt_seq_len, tgt_seq_len]
        memory_mask: Mask for memory of shape [batch_size, 1, tgt_seq_len, src_seq_len]
    """
    # Create padding mask for source
    src_mask = create_padding_mask(src)
    
    # Create padding mask for target
    tgt_mask = create_padding_mask(tgt)
    
    # Create look-ahead mask for target
    look_ahead_mask = create_look_ahead_mask(tgt.size(1))
    
    # Combine padding mask and look-ahead mask for target
    tgt_mask = tgt_mask + look_ahead_mask.unsqueeze(0).unsqueeze(0)
    
    # Create memory mask (padding mask for source)
    memory_mask = src_mask
    
    return src_mask, tgt_mask, memory_mask

def get_positional_encoding(d_model, max_seq_length=5000):
    """
    Create positional encoding matrix.
    
    Args:
        d_model: Dimension of the model
        max_seq_length: Maximum sequence length
            
    Returns:
        pe: Positional encoding matrix of shape [max_seq_length, d_model]
    """
    # Create positional encoding matrix
    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # Apply sine to even indices and cosine to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe 