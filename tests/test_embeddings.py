import unittest
from transformer.embeddings import TokenPositionalEmbedding
import torch

class TestTokenPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.max_len = 50
        self.vocab_size = 1000
        self.embedding = TokenPositionalEmbedding(d_model=self.d_model,max_len = self.max_len,vocab_size=self.vocab_size)

    def test_output_shape(self):
        batch_size = 2
        seq_len = self.max_len
        dummy_tokens = torch.randint(0,1000,(batch_size,seq_len))
        out = self.embedding(dummy_tokens)
        self.assertEqual(out.shape,(batch_size,self.max_len,self.d_model),
                         f"Expected shape ({self.max_len},{self.d_model}), got {out.shape}")
        
    def test_positional_encoding_consistency(self):
        pos = 10
        enc1 = self.embedding.positional_encoding(pos)
        enc2 = self.embedding.positional_encoding(pos)
        self.assertTrue(torch.allclose(enc1,enc2),
                        "Positional encoding should be consistent for the same position")
        
    def test_positional_encoding_value_range(self):
        pos_enc = self.embedding.pos_enc
        self.assertTrue(torch.all(pos_enc<=1.0) and torch.all(pos_enc >=-1.0),"Positional encoding should be between -1 and 1")

    def test_embedding_and_positional_sum(self):
        # Confirm correct shape and output is differentiable
        batch_size = 4
        seq_len = 10
        token_ids = torch.randint(0,self.vocab_size,(batch_size,seq_len))
        # Enable gradient tracking
        token_ids = token_ids.clone().detach().requires_grad_(False)
        output = self.embedding(token_ids)
        self.assertEqual(output.shape,(batch_size,seq_len,self.d_model),
                         f"Output shape mismatch: got {output.shape}, expected ({batch_size},{seq_len},{self.d_model})")
        self.assertTrue(torch.is_floating_point(output),
                        "Output should be a floating point tensor")
        
    def test_gradients_through_embeddings(self):
        batch_size = 2
        seq_len = 5
        token_ids = torch.randint(0,self.vocab_size,(batch_size,seq_len))

        output = self.embedding(token_ids)
        loss = output.sum()

        loss.backward()

        grad = self.embedding.token_emb.weight.grad
        self.assertIsNotNone(grad, "Gradients should not be None for token embeddings")
        self.assertTrue(torch.any(grad != 0),"Token embedding gradients should not be all zero")
        
    if __name__ == "__main__":
        unittest.main()