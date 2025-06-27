import torch
from dit_ml.rope import init_rope_frequencies, compute_rope_embeddings


def test_rope_1d():
    batch_size = 2
    seq_len = 10
    embedding_dim = 32

    # Initialize frequencies
    freqs = init_rope_frequencies(embedding_dim, 1, max_seq_len=seq_len)
    assert freqs.shape == (seq_len, embedding_dim // 2)

    # Create a dummy tensor
    dummy_tensor = torch.randn(batch_size, seq_len, embedding_dim)

    # Compute RoPE embeddings
    rope_tensor = compute_rope_embeddings(freqs, 1, dummy_tensor)

    # Check output shape
    assert rope_tensor.shape == dummy_tensor.shape

    # Check that the embeddings are not the same as the input
    assert not torch.allclose(rope_tensor, dummy_tensor)


def test_rope_2d():
    batch_size = 2
    h = 8
    w = 8
    embedding_dim = 64

    # Initialize frequencies
    freqs_h, freqs_w = init_rope_frequencies(
        embedding_dim, 2, max_height=h, max_width=w
    )
    assert freqs_h.shape == (h, embedding_dim // 4)
    assert freqs_w.shape == (w, embedding_dim // 4)

    # Create a dummy tensor
    dummy_tensor = torch.randn(batch_size, h * w, embedding_dim)

    # Compute RoPE embeddings
    rope_tensor = compute_rope_embeddings((freqs_h, freqs_w), 2, dummy_tensor, h=h, w=w)

    # Check output shape
    assert rope_tensor.shape == dummy_tensor.shape

    # Check that the embeddings are not the same as the input
    assert not torch.allclose(rope_tensor, dummy_tensor)


def test_rope_3d():
    batch_size = 2
    h = 4
    w = 4
    d = 4
    embedding_dim = 96

    # Initialize frequencies
    freqs_h, freqs_w, freqs_d = init_rope_frequencies(
        embedding_dim, 3, max_height=h, max_width=w, max_depth=d
    )
    assert freqs_h.shape == (h, embedding_dim // 6)
    assert freqs_w.shape == (w, embedding_dim // 6)
    assert freqs_d.shape == (d, embedding_dim // 6)

    # Create a dummy tensor
    dummy_tensor = torch.randn(batch_size, h * w * d, embedding_dim)

    # Compute RoPE embeddings
    rope_tensor = compute_rope_embeddings(
        (freqs_h, freqs_w, freqs_d), 3, dummy_tensor, h=h, w=w, d=d
    )

    # Check output shape
    assert rope_tensor.shape == dummy_tensor.shape

    # Check that the embeddings are not the same as the input
    assert not torch.allclose(rope_tensor, dummy_tensor)
