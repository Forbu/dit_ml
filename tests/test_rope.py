import torch
from dit_ml.rope import (
    init_rope_frequencies,
    compute_axial_rope_embeddings,
    compute_mixed_rope_embeddings,
)


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
    rope_tensor = compute_axial_rope_embeddings(freqs, 1, dummy_tensor)

    # Check output shape
    assert rope_tensor.shape == dummy_tensor.shape

    # Check that the embeddings are not the same as the input
    assert not torch.allclose(rope_tensor, dummy_tensor)


def test_rope_2d_mixed():
    batch_size = 2
    nb_heads = 3
    h = 6
    w = 6
    seq_len = h * w
    embedding_dim = 32

    # Initialize frequencies
    freqs_h, freqs_w = init_rope_frequencies(
        embedding_dim, 2, max_height=h, max_width=w
    )
    assert freqs_h.shape == (h, embedding_dim // 2)
    assert freqs_w.shape == (w, embedding_dim // 2)

    # Create a dummy tensor
    dummy_tensor = torch.randn(batch_size, nb_heads, seq_len, embedding_dim)

    # Compute RoPE embeddings
    rope_tensor = compute_mixed_rope_embeddings(
        (freqs_h, freqs_w),
        2,
        dummy_tensor,
        h=h,
        w=w,
    )

    # Check output shape
    assert rope_tensor.shape == dummy_tensor.shape

    # Check that the embeddings are not the same as the input
    assert not torch.allclose(rope_tensor, dummy_tensor)


def test_rope_2d_mixed():
    batch_size = 2
    nb_heads = 3
    h = 6
    w = 6
    d = 2
    seq_len = h * w * d
    embedding_dim = 32

    # Initialize frequencies
    freqs_h, freqs_w, freqs_d = init_rope_frequencies(
        embedding_dim, 3, max_height=h, max_width=w, max_depth=d
    )
    assert freqs_h.shape == (h, embedding_dim // 2)
    assert freqs_w.shape == (w, embedding_dim // 2)
    assert freqs_d.shape == (d, embedding_dim // 2)

    # Create a dummy tensor
    dummy_tensor = torch.randn(batch_size, nb_heads, seq_len, embedding_dim)

    # Compute RoPE embeddings
    rope_tensor = compute_mixed_rope_embeddings(
        (freqs_h, freqs_w, freqs_d),
        3,
        dummy_tensor,
        h=h,
        w=w,
        d=d
    )

    # Check output shape
    assert rope_tensor.shape == dummy_tensor.shape

    # Check that the embeddings are not the same as the input
    assert not torch.allclose(rope_tensor, dummy_tensor)
