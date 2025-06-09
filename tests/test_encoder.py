import unittest
import torch
from transformer.encoder_layer import EncoderLayer

class testEncoderLayer(unittest.TestCase):
    def setup_method(self,method):
        self.d_model = 16
        self.num_heads = 4
        self.d_ff = 64
        self.dropout = 0.1
        self.encoder = EncoderLayer(d_model = self.d_model,num_heads = self.num_heads,d_ff = self.d_ff,dropout = self.dropout)
        self.B,self.T,self.D = 3,10,16
        self.x = torch.randn(self.B,self.T,self.D,requires_grad=True)
    
    def test_output_shape_no_mask(self):
        out = self.encoder(self.x)
        assert out.shape == self.x.shape

    def test_out_shape_with_padding_mask(self):
        #mask last two tokens
        attn_mask = torch.ones(self.B,self.T,dtype=torch.uint8)
        attn_mask[:,-2:] = 0
        padding_mask = attn_mask.unsqueeze(1).unsqueeze(2) # (B,1,1,T)
        out = self.encoder(self.x,mask = padding_mask)
        self.assertEqual(out.shape,self.x.shape)

    def test_residual_when_zeroed(self):
        # if weights are zeroed in self_attn and FFN, output == input normed
        for param in self.encoder.self_attn.parameters():
            param.data.zero_()
        for param in self.encoder.ff.parameters():
            param.data.zero_()

        out = self.encoder(self.x)

        self.assertEqual(out.shape,self.x.shape)
        self.assertTrue(torch.is_floating_point(out))
        # out = ff_norm(attn_norm(x+0) +0)
        assert torch.allclose(out,self.encoder.ff_norm(self.encoder.attn_norm(self.x)),atol=1e-6)
        
    def test_gradient_flow(self):
        out = self.encoder(self.x)
        loss = out.sum()
        loss.backward()
        #input should have gradient
        assert self.x.grad is not None

        grads = [p.grad for p in self.encoder.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum()>0 for g in grads)