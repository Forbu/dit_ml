"""
Module to manage RoPe embeddings

For 1D data :
https://arxiv.org/pdf/2104.09864 (classic rope)

For 2D data and other :
https://arxiv.org/pdf/2403.13298 (RoPE-Mixed)

"""

import torch
import einops
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

    freqs_cis is of dim (seq_len, dim_features // 2)

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
        freqs_h = _precompute_freqs_cis(embedding_dim, max_height, theta=100.0)
        freqs_w = _precompute_freqs_cis(embedding_dim, max_width, theta=100.0)
        return freqs_h, freqs_w
    elif dimensions == 3:
        assert (
            max_height is not None and max_width is not None and max_depth is not None
        ), "max_height, max_width and max_depth must be provided for 3D RoPE"

        freqs_h = _precompute_freqs_cis(embedding_dim, max_height, theta=100.0)
        freqs_w = _precompute_freqs_cis(embedding_dim, max_width, theta=100.0)
        freqs_d = _precompute_freqs_cis(embedding_dim, max_depth, theta=100.0)
        return freqs_h, freqs_w, freqs_d
    else:
        raise NotImplementedError(
            f"RoPE for {dimensions} dimensions not implemented yet"
        )


def compute_axial_rope_embeddings(
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
        query_or_key (torch.Tensor): The input tensor (query or key). Expected shape is (batch_size, num_heads, seq_len, head_dim).
        h (int, optional): The height of the input for 2D/3D RoPE. Defaults to None.
        w (int, optional): The width of the input for 2D/3D RoPE. Defaults to None.
        d (int, optional): The depth of the input for 3D RoPE. Defaults to None.

    Returns:
        torch.Tensor: The tensor with RoPE embeddings applied.
    """
    # The original snippet had a confusing reshape. We work directly on the standard 4D tensor.
    orig_shape = query_or_key.shape
    B, N, S, D = orig_shape

    if dimensions == 1:
        # Flatten batch and head dims for easier processing, then un-flatten
        qok_flat = query_or_key.view(B * N, S, D)
        freqs_cis = frequencies[:S]
        result_flat = _apply_rotary_emb(qok_flat, freqs_cis)
        return result_flat.view(orig_shape)

    elif dimensions == 2:
        assert h is not None and w is not None, "h and w must be provided for 2D RoPE"
        assert S == h * w, "Sequence length S must equal h * w"
        assert D % 2 == 0, "Head dimension D must be even for 2D RoPE"

        # Split the head dimension for axial application
        d_part = D // 2
        qok_h, qok_w = query_or_key.split(d_part, dim=-1)

        # Get precomputed frequencies
        freqs_h, freqs_w = frequencies

        # Apply RoPE to height dimension
        qok_h = einops.rearrange(qok_h, "b n (h w) d -> (b n w) h d", h=h, w=w)
        rotated_h = _apply_rotary_emb(qok_h, freqs_h)
        rotated_h = einops.rearrange(
            rotated_h, "(b n w) h d -> b n (h w) d", b=B, n=N, h=h, w=w
        )

        # Apply RoPE to width dimension
        qok_w = einops.rearrange(qok_w, "b n (h w) d -> (b n h) w d", h=h, w=w)
        rotated_w = _apply_rotary_emb(qok_w, freqs_w)
        rotated_w = einops.rearrange(
            rotated_w, "(b n h) w d -> b n (h w) d", b=B, n=N, h=h, w=w
        )

        return torch.cat([rotated_h, rotated_w], dim=-1)

    elif dimensions == 3:
        assert h is not None and w is not None and d is not None, (
            "h, w, and d must be provided for 3D RoPE"
        )
        assert S == h * w * d, "Sequence length S must equal h * w * d"
        assert D % 6 == 0, "Head dimension D must be divisible by 6 for 3D RoPE"

        # Split the head dimension for axial application
        d_part = D // 3
        qok_h, qok_w, qok_d = query_or_key.split(d_part, dim=-1)

        # Get precomputed frequencies
        freqs_h, freqs_w, freqs_d = frequencies

        # Apply RoPE to height dimension
        qok_h = einops.rearrange(
            qok_h, "b n (h w d) dim -> (b n w d) h dim", h=h, w=w, d=d
        )
        rotated_h = _apply_rotary_emb(qok_h, freqs_h)
        rotated_h = einops.rearrange(
            rotated_h, "(b n w d) h dim -> b n (h w d) dim", b=B, n=N, h=h, w=w, d=d
        )

        # Apply RoPE to width dimension
        qok_w = einops.rearrange(
            qok_w, "b n (h w d) dim -> (b n h d) w dim", h=h, w=w, d=d
        )
        rotated_w = _apply_rotary_emb(qok_w, freqs_w)
        rotated_w = einops.rearrange(
            rotated_w, "(b n h d) w dim -> b n (h w d) dim", b=B, n=N, h=h, w=w, d=d
        )

        # Apply RoPE to depth dimension
        qok_d = einops.rearrange(
            qok_d, "b n (h w d) dim -> (b n h w) d dim", h=h, w=w, d=d
        )
        rotated_d = _apply_rotary_emb(qok_d, freqs_d)
        rotated_d = einops.rearrange(
            rotated_d, "(b n h w) d dim -> b n (h w d) dim", b=B, n=N, h=h, w=w, d=d
        )

        return torch.cat([rotated_h, rotated_w, rotated_d], dim=-1)

    else:
        raise ValueError(f"Unsupported dimensions: {dimensions}")


def _apply_rotary_emb_mixed_2d(
    x: torch.Tensor,
    freqs_cis_h: torch.Tensor,
    freqs_cis_w: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Applies rotary embeddings to the input tensor.
    x is of shape (b, nb_seq, d_feat)
    freqs_cis_h is of shape (h, d_feat)
    freqs_cis_w is of shape (w, d_feat)

    """
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_ = torch.view_as_complex(x_)  # (b, nb_seq, d_feat // 2)

    x_ = einops.rearrange(x_, "b (h w) d -> b h w d", h=h, w=w)

    freqs_cis_h_ = freqs_cis_h.unsqueeze(0).unsqueeze(2)  # shape (1, h, 1, d)
    freqs_cis_w_ = freqs_cis_w.unsqueeze(0).unsqueeze(1)  # shape (1, 1, w, d)

    x_out = x_ * freqs_cis_h_ * freqs_cis_w_

    x_out = torch.view_as_real(x_out)
    x_out = x_out.flatten(3)

    return x_out.type_as(x)


def _apply_rotary_emb_mixed_3d(
    x: torch.Tensor,
    freqs_cis_h: torch.Tensor,
    freqs_cis_w: torch.Tensor,
    freqs_cis_d: torch.Tensor,
    h: int,
    w: int,
    d: int,
) -> torch.Tensor:
    """
    Applies rotary embeddings to the input tensor.
    x is of shape (b, nb_seq, d_feat)
    freqs_cis_h is of shape (h, d_feat)
    freqs_cis_w is of shape (w, d_feat)
    freqs_cis_d is of shape (d, d_feat)

    """
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_ = torch.view_as_complex(x_)  # (b, nb_seq, d_feat // 2)

    x_ = einops.rearrange(x_, "b (t h w) d -> b h w t d", h=h, w=w, t=d)

    freqs_cis_h_ = (
        freqs_cis_h.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    )  # shape (1, h, 1, 1, d_feat // 2)
    freqs_cis_w_ = (
        freqs_cis_w.unsqueeze(0).unsqueeze(1).unsqueeze(3)
    )  # shape (1, 1, w, 1, d_feat // 2)
    freqs_cis_d_ = (
        freqs_cis_d.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    )  # shape (1, 1, 1, d, d_feat // 2)

    x_out = x_ * freqs_cis_h_.to(x_.device) * freqs_cis_w_.to(x_.device) * freqs_cis_d_.to(x_.device)

    x_out = torch.view_as_real(x_out)
    x_out = x_out.flatten(4)

    return x_out.type_as(x)


def compute_mixed_rope_embeddings(
    frequencies: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    dimensions: int,
    query_or_key: torch.Tensor,
    h: int = None,
    w: int = None,
    d: int = None,
) -> torch.Tensor:
    """
    Compute the RoPe embeddings for a given query of key
    but for mixed version :)

    Args:
        frequencies (Union[torch.Tensor, tuple[torch.Tensor, ...]]): The frequency tensor(s) for the RoPe embeddings.
        dimensions (int): The number of dimensions for the query or key (1, 2 or 3).
        query_or_key (torch.Tensor): The input tensor (query or key). Expected shape is (batch_size, num_heads, seq_len, head_dim).
        h (int, optional): The height of the input for 2D/3D RoPE. Defaults to None.
        w (int, optional): The width of the input for 2D/3D RoPE. Defaults to None.
        d (int, optional): The depth of the input for 3D RoPE. Defaults to None.

    Returns:
        torch.Tensor: The tensor with RoPE embeddings applied.
    """
    # The original snippet had a confusing reshape. We work directly on the standard 4D tensor.
    orig_shape = query_or_key.shape
    B, N, S, D = orig_shape

    qok_flat = einops.rearrange(query_or_key, "b h s d -> (b h) s d")

    if dimensions == 1:
        # Flatten batch and head dims for easier processing, then un-flatten
        freqs_cis = frequencies[:S]
        result_flat = _apply_rotary_emb(qok_flat, freqs_cis)

    elif dimensions == 2:
        assert h is not None and w is not None, "h and w must be provided for 2D RoPE"
        assert S == h * w, "Sequence length S must equal h * w"

        # Get precomputed frequencies
        freqs_h, freqs_w = frequencies

        result_flat = _apply_rotary_emb_mixed_2d(qok_flat, freqs_h, freqs_w, h, w)
        result_flat = einops.rearrange(result_flat, "b h w d -> b (h w) d")

    elif dimensions == 3:
        assert h is not None and w is not None and d is not None, (
            "h, w, and d must be provided for 3D RoPE"
        )
        assert S == h * w * d, "Sequence length S must equal h * w * d"

        # Get precomputed frequencies
        freqs_h, freqs_w, freqs_d = frequencies
        result_flat = _apply_rotary_emb_mixed_3d(
            qok_flat, freqs_h, freqs_w, freqs_d, h, w, d
        )

        result_flat = einops.rearrange(result_flat, "b h w t d -> b (t h w) d")

    else:
        raise ValueError(f"Unsupported dimensions: {dimensions}")

    return einops.rearrange(result_flat, "(b h) s d -> b h s d", b=B, h=N)