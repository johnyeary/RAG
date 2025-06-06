import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q,k,v, mask=None):
    """
    q,k,v: tensors shaped (batch_size,num_heads,seq_len,d_k)
    mask: (batch_size,1,1,seq_len) or None

    Returns:
        output: (batch_size, num_heads,seq_len,d_k)
        attn_weights: (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = q.size(-1)
    # Compute raw scores
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)
    
    #apply mask (if given)
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    # softmax to get attention weights
    attn_weights = F.softmax(scores,dim=-1) # shape: (batch, heads,seq_len,seq_len)

    # weighted sum over V
    output = torch.matmul(attn_weights, v) #(batch,heads,seq_len, d_k)

    return output,attn_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, D = x.size()

        #linear projections
        Q = self.q_linear(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        K = self.k_linear(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        V = self.v_linear(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        # Get attention
        attn_out,attn_weights = scaled_dot_product_attention(Q,K,V,mask)
       
        # Merge heads
        out = attn_out.transpose(1,2).contiguous().view(B,T,D)
        return self.out_proj(out), attn_weights 