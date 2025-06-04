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