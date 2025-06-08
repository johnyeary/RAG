import unittest
import torch
from transformer.attention import scaled_dot_product_attention
from transformer.attention import MultiHeadSelfAttention

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

class TestMultiHeadAttention(unittest.TestCase):
    def setup_method(self,method):
        self.batch = 2
        self.seq_len = 5
        self.d_model = 16
        self.num_heads = 4
        self.attn = MultiHeadSelfAttention(self.d_model,self.num_heads)

    def test_output_and_weight_shapes(self):
        x = torch.randn(self.batch,self.seq_len,self.d_model)
        out, weights = self.attn(x,mask=None)

        self.assertEqual(out.shape, (self.batch, self.seq_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch, self.num_heads, self.seq_len, self.seq_len))

    def test_padding_mask(self):
        x = torch.randn(self.batch, self.seq_len,self.d_model)
        # mask first token
        attn_mask = torch.tensor([[[[0,1,1,1,1]]]],dtype=torch.uint8)
        out,weights = self.attn(x,mask=attn_mask)
        # for each head, and query position, confirm weight at index 0 is 0 
        self.assertTrue(torch.all(weights[...,0] == 0))
    
    def test_gradient_flow(self):
        #ensure backprop works
        x = torch.randn(self.batch,self.seq_len,self.d_model,requires_grad=True)
        out,_ = self.attn(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # at least one parameter should have gradient
        grads = [p.grad for p in self.attn.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))
        
    def test_single_head_equivalence(self):
        # test to see if I have indexing or reshaping issues
        x = torch.randn(self.batch,self.seq_len,self.d_model)
        attn1 = MultiHeadSelfAttention(self.d_model,num_heads = 1)
        out_multi,weights_multi = attn1(x)

        #manually compute linear projections
        Q_chk = attn1.q_linear(x)
        K_chk = attn1.k_linear(x)
        V_chk = attn1.v_linear(x)

        Qh = Q_chk.view(self.batch, self.seq_len, 1, self.d_model).transpose(1, 2)  # (batch, 1, seq_len, d_model)
        Kh = K_chk.view(self.batch, self.seq_len, 1, self.d_model).transpose(1, 2)
        Vh = V_chk.view(self.batch, self.seq_len, 1, self.d_model).transpose(1, 2)
        attn_out_chk, weights_single = scaled_dot_product_attention(Qh,Kh,Vh,mask=None)

        #concat heads (really one head)
        out_single_head = attn_out_chk.transpose(1,2).contiguous().view(self.batch,self.seq_len,self.d_model)
        out_comp = attn1.out_proj(out_single_head)

        assert torch.allclose(out_multi,out_comp,atol=1e-6)
        assert torch.allclose(weights_multi, weights_single,atol=1e-6)
        