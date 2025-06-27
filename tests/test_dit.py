import torch
import pytest

from dit_ml.dit import DiT, Attention

def test_dit_init():
    """Test the initialization of the DiT model."""
    input_size = 32
    in_channels = 4
    hidden_size = 8 # Use a smaller size for faster testing
    depth = 2 # Use a smaller depth
    num_heads = 4 # Use a smaller number of heads
    learn_sigma = True

    model = DiT(
        num_patches=input_size * input_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        learn_sigma=learn_sigma
    )

    assert isinstance(model, DiT)
    assert model.learn_sigma == learn_sigma
    assert model.num_heads == num_heads
    assert model.num_patches == input_size * input_size
    assert len(model.blocks) == depth

def test_dit_forward():
    """Test the forward pass of the DiT model."""
    input_size = 32
    hidden_size = 64 # Use a smaller size for faster testing
    depth = 2 # Use a smaller depth
    num_heads = 4 # Use a smaller number of heads
    learn_sigma = True
    batch_size = 2

    model = DiT(
        num_patches=input_size * input_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        learn_sigma=learn_sigma
    )

    dummy_x = torch.randn(batch_size, input_size * input_size, hidden_size)
    dummy_c = torch.randn(batch_size, hidden_size) # Dummy conditioning vector

    output = model(dummy_x, dummy_c)

    expected_output_shape = (batch_size, input_size * input_size, hidden_size)
    assert output.shape == expected_output_shape


def test_dit_causal_block_init():
    """Test the initialization of the DiT model with causal_block enabled."""
    input_size = 8
    hidden_size = 16
    depth = 2
    num_heads = 4
    learn_sigma = True
    causal_block = True
    causal_block_size = input_size * input_size

    model = DiT(
        num_patches=input_size * input_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        learn_sigma=learn_sigma,
        causal_block=causal_block,
        causal_block_size=causal_block_size,
    )

    assert isinstance(model, DiT)
    assert model.causal_block is causal_block
    assert model.causal_block_size == causal_block_size
    assert len(model.blocks) == depth


def test_dit_causal_block_invalid_size():
    """Test initializing DiT with an invalid causal_block_size raises an error."""
    with pytest.raises(Exception):
        DiT(
            num_patches=10,
            hidden_size=16,
            depth=1,
            num_heads=2,
            learn_sigma=False,
            causal_block=True,
            causal_block_size=4,
        )

def test_rope_attention():
    """Test the Attention module with RoPE enabled."""
    batch_size = 2
    h = 8
    w = 8
    dim = 64
    num_heads = 4

    # With RoPE
    attn_rope = Attention(dim, num_heads, use_rope=True, max_h=h, max_w=w)
    dummy_x = torch.randn(batch_size, h * w, dim)
    output_rope = attn_rope(dummy_x, h=h, w=w)
    assert output_rope.shape == dummy_x.shape

    # Without RoPE
    attn_no_rope = Attention(dim, num_heads, use_rope=False)
    output_no_rope = attn_no_rope(dummy_x, h=h, w=w)
    assert output_no_rope.shape == dummy_x.shape

    # Check that the outputs are different
    assert not torch.allclose(output_rope, output_no_rope)