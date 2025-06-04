import unittest
import torch
from transformer.attention import scaled_dot_product_attention

class testAttention(unittest.TestCase):
    def test_attention_shape_no_mask(self):
        batch_size = 2
        num_heads = 4
        seq_len = 5
        d_k = 8

        # random q,k,v
        q = torch.rand(batch_size, num_heads, seq_len, d_k)
        k = torch.rand(batch_size, num_heads, seq_len, d_k)
        v = torch.rand(batch_size, num_heads, seq_len, d_k)

        out, attn = scaled_dot_product_attention(q, k, v, mask=None)

        self.assertEqual(out.shape, (batch_size, num_heads, seq_len, d_k), "Out shape matches")
        self.assertEqual(attn.shape, (batch_size, num_heads, seq_len, seq_len), "attn shape matches")

    def test_attention_with_masking(self):
        batch_size = 1
        num_heads  = 2
        seq_len    = 3
        d_k        = 8

        q = torch.ones(batch_size,num_heads,seq_len,d_k)
        k = torch.ones(batch_size, num_heads,seq_len,d_k)
        v = torch.arange(seq_len, dtype=torch.float).view(1,1,seq_len,1).repeat(1,num_heads,1,d_k)
        #mask out last token
        mask = torch.tensor([[[[1,1,0]]]], dtype = torch.uint8)

        out, attn = scaled_dot_product_attention(q,k,v,mask=mask)

        #last key's probability should be zero
        last_col = attn[:,:,:,2]
        self.assertTrue(torch.all(last_col == 0), "Masked positions have zero weight")

    def test_softmax_sum(self):
        batch_size = 1
        num_heads  = 2
        seq_len    = 3
        d_k        = 8

        q = torch.ones(batch_size,num_heads,seq_len,d_k)
        k = torch.ones(batch_size, num_heads,seq_len,d_k)
        v = torch.arange(seq_len, dtype=torch.float).view(1,1,seq_len,1).repeat(1,num_heads,1,d_k)

        _, weights = scaled_dot_product_attention(q,k,v,mask=None)
        summed = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(summed,torch.ones_like(summed),atol=1e-5))

    def test_input_correctness(self):
        q = torch.tensor([[[1.0,0.0]]])
        k = torch.tensor([[[1.0,0.0]]])
        v = torch.tensor([[[0.0,1.0]]])

        out,attn= scaled_dot_product_attention(q,k,v,mask=None)
        # dot(q,k) = 1 softmax = 1 = output = v
        self.assertTrue(torch.allclose(out,v),"out should match")
        self.assertTrue(torch.allclose(attn,torch.tensor([1.0])))

