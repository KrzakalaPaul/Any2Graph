import torch.nn as nn
import torch
from math import log

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return self.pe[:x.size(1)][None,:,:]
    
class TokenEncoder(nn.Module):
    
    def __init__(self, model_dim: int, vocab_len: int, pos_embed: bool):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_len,embedding_dim=model_dim)
        if pos_embed:
            self.pos_encoding = PositionalEncoding(d_model=model_dim,dropout=0.,max_len=100)
        else:
            self.pos_encoding = None
        
    def forward(self,inputs):
        tokens = inputs['tokens']
        masks = inputs['masks']
        x = self.token_embedding(tokens)
        if self.pos_encoding != None:
            pos_embed = self.pos_encoding(x)
        else:
            pos_embed = torch.zeros_like(x)
        return x,masks,pos_embed

