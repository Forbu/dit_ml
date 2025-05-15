<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/dit_ml)](https://pypi.org/project/dit_ml/)
[![CI](https://github.com/Forbu/dit_ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Forbu/dit_ml/actions/workflows/ci.yml)

</div>

### dit_ml

The goal of dit_ml is to create a python repository to create the base model for diffusion transformer (https://arxiv.org/abs/2212.09748)

## Installation

You can install `dit_ml` using pip:

```bash
pip install dit_ml
```

## Usage

Here's a basic example of how to use `dit_ml`:

```python
from dit_ml.dit import DiT

model = DiT(
    num_patches=input_size*input_size, # if 2d with flatten size
    in_channels=in_channels,
    hidden_size=hidden_size,
    depth=depth,
    num_heads=num_heads,
    learn_sigma=learn_sigma
)

dummy_x = torch.randn(batch_size, input_size * input_size, hidden_size)

dummy_c = torch.randn(batch_size, hidden_size) # Dummy conditioning vector

output = model(dummy_x, dummy_c) # of shape (batch_size, input_size * input_size, hidden_size)
```

## Development

To set up the development environment:

1. Clone the repository:

```bash
git clone https://github.com/Forbu/dit_ml.git
cd dit_ml
```

2. Install dependencies using uv:

```bash
uv sync
```

3. Run tests:

```bash
uv run pytest
```

## Contributing

Contributions are welcome! Please see the [LICENSE](LICENSE) for details.

## License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.
