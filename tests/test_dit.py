import torch
import pytest

from dit_ml.dit import DiT

def test_dit_init():
    """Test the initialization of the DiT model."""
    input_size = 32
    in_channels = 4
    hidden_size = 8 # Use a smaller size for faster testing
    depth = 2 # Use a smaller depth
    num_heads = 4 # Use a smaller number of heads
    learn_sigma = True

    model = DiT(
        input_size=input_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        learn_sigma=learn_sigma
    )

    assert isinstance(model, DiT)
    assert model.learn_sigma == learn_sigma
    assert model.in_channels == in_channels
    assert model.out_channels == in_channels * 2 if learn_sigma else in_channels
    assert model.num_heads == num_heads
    assert model.num_patches == input_size * input_size
    assert len(model.blocks) == depth
    assert model.pos_embed.shape == (1, input_size * input_size, hidden_size)

def test_dit_forward():
    """Test the forward pass of the DiT model."""
    input_size = 32
    in_channels = 4
    hidden_size = 64 # Use a smaller size for faster testing
    depth = 2 # Use a smaller depth
    num_heads = 4 # Use a smaller number of heads
    learn_sigma = True
    batch_size = 2

    model = DiT(
        num_patches=input_size * input_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        learn_sigma=learn_sigma
    )

    # Create dummy input data
    # The forward method expects input x with shape (N, T, D) where T is num_patches and D is hidden_size
    # The original code seems to expect (N, C, H, W) and then converts it. Let's follow the forward method's expected input shape after the initial conversion.
    # Based on the forward method: `x = (x + self.pos_embed)` suggests x is already (N, T, D) or will be broadcasted.
    # Let's assume the input x to the forward method is already in the shape (N, num_patches, hidden_size) after some initial processing (like patch embedding, which is not shown in the provided code).
    # The timestep `t` is expected to be (N,)
    dummy_x = torch.randn(batch_size, input_size * input_size, hidden_size)

    # The forward method signature is forward(self, x, t). It seems the original code snippet might be incomplete or assumes 'y' is handled elsewhere.
    # Let's test with the provided signature: forward(self, x, t)
    # The DiTBlock forward method takes (x, c). In DiT forward, 'c' is 't'.
    # The adaLN_modulation layer in DiTBlock expects input of size (hidden_size).
    # The timestep `t` is (N,). It needs to be processed to get the conditioning vector `c` of size (N, hidden_size).
    # The provided DiT forward method does not show how `t` is converted to `c`.
    # Let's assume for the test that the conditioning vector `c` is generated outside and passed to the forward method, or that `t` is expected to be (N, hidden_size).
    # Looking at the DiTBlock's adaLN_modulation layer: `nn.Linear(hidden_size, 6 * hidden_size, bias=True)`
    # This implies the conditioning vector `c` should have `hidden_size` features.
    # Let's assume the `t` input to the DiT forward method is actually the processed conditioning vector `c` of shape (N, hidden_size).

    dummy_c = torch.randn(batch_size, hidden_size) # Dummy conditioning vector

    output = model(dummy_x, dummy_c)

    # Expected output shape is (N, T, D)
    expected_output_shape = (batch_size, input_size * input_size, hidden_size)
    assert output.shape == expected_output_shape
