import unittest
import torch
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

    def test_output_shape(self):
        out = self.encoder(self.token_ids)
        self.assertEqual(out.shape,(self.batch_size, self.seq_len, self.d_model))

    def test_gradients_flow(self):
        out = self.encoder(self.token_ids)
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in self.encoder.parameters() if p.requires_grad]
        assertTrue(all(g is not None for g in grads), "Some parameters did not receive gradients")

    def test_zero_layer_identity(self):
        test_encoder = self.encoder
        for layer in test_encoder.layers:
            for m in layer.modules():
                if isinstance(m,torch.nn.Linear):
                    torch.nn.init.constant_(m.weight,0)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias,0)
        embeddings = test_encoder.embedding(self.token_ids)
        output = test_encoder(self.token_ids)

        self.assertTrue(torch.allclose(output,embeddings,atol=1e-5))

    def test_mask(self):
        padding_mask = self.token_ids == 0
        out = self.encoder(self.token_ids,mask = padding_mask)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len,self.d_model))

    def test_i_o(self):
        input_2 = torch.randint(0,self.vocab_size,(self.batch_size,self.seq_len))
        out1 = self.encoder(self.token_ids)
        out2 = self.encoder(input_2)
        self.assertFalse(torch.allclose(out1,out2), "Different inputs cause different outputs")