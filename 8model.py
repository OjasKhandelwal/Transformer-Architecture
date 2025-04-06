import torch
import torch.nn as nn

from transformer.embeddings import Embeddings
from transformer.encoder import Encoder
from transformer.decoder import Decoder

class Transformer(nn.Module):
    """
    This implements the complete Transformer model as described in the paper:
    - Encoder
    - Decoder
    - Final linear layer
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Source and target embeddings
        self.src_embeddings = Embeddings(src_vocab_size, d_model, dropout=dropout)
        self.tgt_embeddings = Embeddings(tgt_vocab_size, d_model, dropout=dropout)
        
        # Encoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)
        
        # Decoder
        self.decoder = Decoder(d_model, num_heads, d_ff, num_decoder_layers, dropout)
        
        # Final linear layer
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Apply the Transformer model to the input.
        
        Args:
            src: Source tensor of shape [batch_size, src_seq_len]
            tgt: Target tensor of shape [batch_size, tgt_seq_len]
            src_mask: Optional mask for source of shape [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: Optional mask for target of shape [batch_size, 1, tgt_seq_len, tgt_seq_len]
            memory_mask: Optional mask for memory of shape [batch_size, 1, tgt_seq_len, src_seq_len]
                
        Returns:
            output: Output tensor of shape [batch_size, tgt_seq_len, tgt_vocab_size]
            enc_attention_weights: List of encoder attention weights
            dec_self_attention_weights: List of decoder self-attention weights
            dec_cross_attention_weights: List of decoder cross-attention weights
        """
        # Apply embeddings
        src_embeddings = self.src_embeddings(src)
        tgt_embeddings = self.tgt_embeddings(tgt)
        
        # Apply encoder
        enc_output, enc_attention_weights = self.encoder(src_embeddings, src_mask)
        
        # Apply decoder
        dec_output, dec_self_attention_weights, dec_cross_attention_weights = self.decoder(
            tgt_embeddings, enc_output, tgt_mask, memory_mask
        )
        
        # Apply final linear layer
        output = self.final_layer(dec_output)
        
        return output, enc_attention_weights, dec_self_attention_weights, dec_cross_attention_weights
