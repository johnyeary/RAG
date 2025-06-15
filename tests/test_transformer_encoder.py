import unittest
import torch
import copy
from transformer.transformer_encoder import TransformerEncoder

class TestTransformerEncoder(unittest.TestCase):
    def setup_method(self,method):
        self.vocab_size = 100
        self.d_model = 16
        self.seq_len = 10
        self.batch_size = 2
        self.num_layers = 2
        self.num_heads = 4
        self.d_ff = 64
        self.max_len = 50
        self.encoder = TransformerEncoder(
            vocab_size = self.vocab_size,
            d_model = self.d_model,
            num_layers = self.num_layers,
            num_heads = self.num_heads,
            d_ff  = self.d_ff,
            max_len  = self.max_len,
            dropout = 0.0
        )
        self.token_ids = torch.randint(0, self.vocab_size, (self.batch_size,self.seq_len))

    def generate_causal_mask(self,seq_len,batch_size=1):
        mask = torch.tril(torch.ones(seq_len,seq_len)).bool()
        return mask.unsqueeze(0).unsqueeze(1).expand(batch_size,1,seq_len,seq_len)

    def test_output_shape(self):
        test_encoder = copy.deepcopy(self.encoder)
        out = test_encoder(self.token_ids)
        self.assertEqual(out.shape,(self.batch_size, self.seq_len, self.d_model))

    def test_gradients_flow(self):
        test_encoder = copy.deepcopy(self.encoder)
        out = test_encoder(self.token_ids)
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in self.encoder.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads), "Some parameters did not receive gradients")

    def test_zero_layer_identity(self):
        test_encoder = copy.deepcopy(self.encoder)
        #test_encoder.eval()
        for layer in test_encoder.layers:
            layer.ff[0].weight.data.zero_()
            layer.ff[0].bias.data.zero_()
            layer.ff[2].weight.data.zero_()
            layer.ff[2].bias.data.zero_()
            layer.self_attn.q_linear.weight.data.zero_()
            layer.self_attn.q_linear.bias.data.zero_()
            layer.self_attn.k_linear.weight.data.zero_()
            layer.self_attn.k_linear.bias.data.zero_()
            layer.self_attn.v_linear.weight.data.zero_()
            layer.self_attn.v_linear.bias.data.zero_()
            layer.self_attn.out_proj.weight.data.zero_()
            layer.self_attn.out_proj.bias.data.zero_()
            layer.attn_norm.weight.data.fill_(1.0)
            layer.attn_norm.bias.data.zero_()
            layer.ff_norm.weight.data.fill_(1.0)
            layer.ff_norm.bias.data.zero_()
        embeddings = test_encoder.embedding(self.token_ids)
        output = test_encoder(self.token_ids)
        print(f"Output shape {output.shape} embeddings.shape {embeddings.shape}")
        print(f"Max abs diff: {(output - embeddings).abs().max()}")
        self.assertTrue(torch.allclose(output,embeddings,atol=1e-5))

    def test_causal_mask_encoder(self):
        testEncoder = copy.deepcopy(self.encoder)
        for layer in testEncoder.layers:
            layer.ff[0].weight.data.zero_()
            layer.ff[0].bias.data.zero_()
            layer.ff[2].weight.data.zero_()
            layer.ff[2].bias.data.zero_()
        x = torch.randint(low = 0, high = self.vocab_size,size = (self.batch_size,self.seq_len))

        mask = self.generate_causal_mask(self.seq_len,self.batch_size)
        out_masked = testEncoder(x.clone(),mask = mask)
        out_unmasked = testEncoder(x.clone(),mask = None)
        self.assertFalse(torch.allclose(out_masked,out_unmasked,atol=1e-5))

    def test_i_o(self):
        input_2 = torch.randint(0,self.vocab_size,(self.batch_size,self.seq_len))
        out1 = self.encoder(self.token_ids)
        out2 = self.encoder(input_2)
        self.assertFalse(torch.allclose(out1,out2), "Different inputs cause different outputs")