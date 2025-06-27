from typing import Final, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models.vision_transformer import Mlp
from dit_ml.rope import init_rope_frequencies, compute_rope_embeddings


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        causal_block=False,
        causal_block_size=32*32,
        use_rope: bool = False,
        rope_dimension: int = 2,
        max_h: int = 32,
        max_w: int = 32,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
            use_rope: Whether to use rotary positional embeddings
            rope_dimension: The dimension of the RoPE (1, 2 or 3)
            max_h: The maximum height of the input for 2D RoPE
            max_w: The maximum width of the input for 2D RoPE
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        if qk_norm or scale_norm:
            assert norm_layer is not None, (
                "norm_layer must be provided if qk_norm or scale_norm is True"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = use_rope
        self.rope_dimension = rope_dimension

        self.causal_block = causal_block
        self.causal_block_size = causal_block_size

        if self.use_rope:
            self.rope_frequencies = init_rope_frequencies(
                self.head_dim * self.num_heads,
                self.rope_dimension,
                max_height=max_h,
                max_width=max_w,
            )

        if causal_block:
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx + self.causal_block_size >= kv_idx

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            q_b, q_h, q_n, q_d = q.shape
            k_b, k_h, k_n, k_d = k.shape

            q = q.permute(0, 2, 1, 3).reshape(q_b, q_n, q_h * q_d)
            k = k.permute(0, 2, 1, 3).reshape(k_b, k_n, k_h * k_d)

            q = compute_rope_embeddings(self.rope_frequencies, self.rope_dimension, q, h=h, w=w)
            k = compute_rope_embeddings(self.rope_frequencies, self.rope_dimension, k, h=h, w=w)

            q = q.reshape(q_b, q_n, q_h, q_d).permute(0, 2, 1, 3)
            k = k.reshape(k_b, k_n, k_h, k_d).permute(0, 2, 1, 3)

        if not self.causal_block:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            
            block_mask = create_block_mask(causal_mask, B, self.num_heads, N, N, device=self.device)
            x = flex_attention(q, k, v, block_mask=block_mask)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat(
            [torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    embed_dim: output dimension for each position
    grid: a list of positions to be encoded: size (2, 1, H, W)
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, causal_block=False, causal_block_size=32*32, use_rope=False, rope_dimension=2, max_h=32, max_w=32, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, causal_block=False, causal_block_size=32*32, use_rope=use_rope, rope_dimension=rope_dimension, max_h=max_h, max_w=max_w, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, h=None, w=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            h=h,
            w=w
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        num_patches=32 * 32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        causal_block=False,
        causal_block_size=32 * 32,
        use_rope=False,
        rope_dimension=2,
        max_h=32,
        max_w=32,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.causal_block = causal_block
        self.causal_block_size = causal_block_size
        self.use_rope = use_rope

        # check if the causal block divide the num_patches
        if causal_block and num_patches % causal_block_size != 0:
            raise ("causal_block_size doesn't divide num_patches")

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    causal_block=False,
                    causal_block_size=32 * 32,
                    use_rope=use_rope,
                    rope_dimension=rope_dimension,
                    max_h=max_h,
                    max_w=max_w,
                )
                for _ in range(depth)
            ]
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, h=None, w=None):
        """
        Forward pass of DiT.
        x: (N, nb_seq, hidden_dim) tensor of spatial inputs (images or latent representations of images)
        t: (N, hidden_dim) tensor of diffusion timesteps
        h: (int) height of the input
        w: (int) width of the input
        """

        for block in self.blocks:
            x = block(x, t, h=h, w=w)  # (N, T, D)
        return x
