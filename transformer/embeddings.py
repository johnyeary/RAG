import torch
import torch.nn as nn
import math
class TokenPositionalEmbedding(nn.Module):
    #positional encoding
    def __init__(self,vocab_size,d_model,max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size,d_model)
        #sine-cosine positional encodings
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
      
       # interleave sin/cos encodings
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        
        # want dim (1,max_len,d_model)
        pe = pe.unsqueeze(0)

        # register as buffer so not updated by gradients
        self.register_buffer("pos_enc",pe)
    
    def forward(self,token_ids: torch.LongTensor):
        """
        token_ids: (batch_size, seq_len) of token IDs
        returns: (batch_size,seq_len, d_model) embeddings
        """
        batch_size,seq_len = token_ids.size()

        # Look up token embeddings
        tok_emb = self.token_emb(token_ids)

        # Slice positional encodings for seq_len
        pos_emb = self.pos_enc[:,:seq_len,:]
        
        # add together and return
        return tok_emb + pos_emb
    
    def positional_encoding(self,pos:int):
        """
        Return positional encoding vector for 'pos'
        """
        return self.pos_enc[0,pos,:]