import torch
from dit_ml.rope import init_rope_frequencies, compute_rope_embeddings


def test_rope_1d():
    batch_size = 2
    nb_heads = 3
    seq_len = 10
    embedding_dim = 32

    # Initialize frequencies
    freqs = init_rope_frequencies(embedding_dim, 1, max_seq_len=seq_len)
    assert freqs.shape == (seq_len, embedding_dim // 2)

    # Create a dummy tensor
    dummy_tensor = torch.randn(batch_size, nb_heads, seq_len, embedding_dim)

    # Compute RoPE embeddings
    rope_tensor = compute_rope_embeddings(freqs, 1, dummy_tensor)

    # Check output shape
    assert rope_tensor.shape == dummy_tensor.shape

    # Check that the embeddings are not the same as the input
    assert not torch.allclose(rope_tensor, dummy_tensor)

