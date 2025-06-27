"""
Module to manage RoPe embeddings

For 1D data :
https://arxiv.org/pdf/2104.09864 (classic rope)

For 2D data :
https://arxiv.org/pdf/2403.13298 (2D mixed rope)

"""

import torch
from typing import Union


def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the frequency tensor for RoPE embeddings.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the frequency tensor to be broadcastable with the input tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1] // 2), (
        f"freqs_cis.shape={freqs_cis.shape}, x.shape={x.shape}"
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    shape[-1] = shape[-1] // 2
    return freqs_cis.view(*shape)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary embeddings to the input tensor.
    """
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_ = torch.view_as_complex(x_)
    freqs_cis = _reshape_for_broadcast(freqs_cis, x)
    x_out = x_ * freqs_cis.to(x_.device)
    x_out = torch.view_as_real(x_out)
    x_out = x_out.flatten(2)
    return x_out.type_as(x)


def init_rope_frequencies(
    embedding_dim: int,
    dimensions: int,
    max_seq_len: int = None,
    max_height: int = None,
    max_width: int = None,
    max_depth: int = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    """
    Initialize the frequency tensor for RoPe embeddings.

    Args:
        embedding_dim (int): The embedding dimension of the model.
        dimensions (int): The number of dimensions for the RoPE (1, 2 or 3).
        max_seq_len (int, optional): The maximum sequence length for 1D RoPE. Defaults to None.
        max_height (int, optional): The maximum height for 2D/3D RoPE. Defaults to None.
        max_width (int, optional): The maximum width for 2D/3D RoPE. Defaults to None.
        max_depth (int, optional): The maximum depth for 3D RoPE. Defaults to None.

    Returns:
        Union[torch.Tensor, tuple[torch.Tensor, ...]]: The frequency tensor(s).
    """
    if dimensions == 1:
        assert max_seq_len is not None, "max_seq_len must be provided for 1D RoPE"
        return _precompute_freqs_cis(embedding_dim, max_seq_len)
    elif dimensions == 2:
        assert max_height is not None and max_width is not None, (
            "max_height and max_width must be provided for 2D RoPE"
        )
        freqs_h = _precompute_freqs_cis(embedding_dim, max_height)
        freqs_w = _precompute_freqs_cis(embedding_dim, max_width)
        return freqs_h, freqs_w
    elif dimensions == 3:
        assert (
            max_height is not None and max_width is not None and max_depth is not None
        ), "max_height, max_width and max_depth must be provided for 3D RoPE"

        freqs_h = _precompute_freqs_cis(embedding_dim, max_height)
        freqs_w = _precompute_freqs_cis(embedding_dim, max_width)
        freqs_d = _precompute_freqs_cis(embedding_dim, max_depth)
        return freqs_h, freqs_w, freqs_d
    else:
        raise NotImplementedError(
            f"RoPE for {dimensions} dimensions not implemented yet"
        )


def compute_rope_embeddings(
    frequencies: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    dimensions: int,
    query_or_key: torch.Tensor,
    h: int = None,
    w: int = None,
    d: int = None,
) -> torch.Tensor:
    """
    Compute the RoPe embeddings for a given query of key

    Args:
        frequencies (Union[torch.Tensor, tuple[torch.Tensor, ...]]): The frequency tensor(s) for the RoPe embeddings.
        dimensions (int): The number of dimensions for the query or key (1, 2 or 3).
        query_or_key (torch.Tensor): The input tensor (query or key)
        h (int, optional): The height of the input for 2D/3D RoPE. Defaults to None.
        w (int, optional): The width of the input for 2D/3D RoPE. Defaults to None.
        d (int, optional): The depth of the input for 3D RoPE. Defaults to None.

    Returns:
        torch.Tensor: The tensor with RoPE embeddings applied.
    """
    if dimensions == 1:
        freqs_cis = frequencies[: query_or_key.shape[1]]
        return _apply_rotary_emb(query_or_key, freqs_cis)
    elif dimensions == 2:
        assert h is not None and w is not None, "h and w must be provided for 2D RoPE"
        freqs_cis_h, freqs_cis_w = frequencies
        dim = query_or_key.shape[-1]
        return None
    elif dimensions == 3:
        assert h is not None and w is not None and d is not None, (
            "h, w and d must be provided for 3D RoPE"
        )
        freqs_cis_h, freqs_cis_w, freqs_cis_d = frequencies
        dim = query_or_key.shape[-1]
        return None
    else:
        raise ValueError(f"Unsupported dimensions: {dimensions}")
