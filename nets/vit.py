import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_classes: int,
                 dim: int, depth: int, heads: int, mlp_dim: int, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1), dim)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embed, nhead, dropout=0., kdim=None, vdim=None, device=None, dtype=None):
        super(MultiHeadAttention, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.dim_embed = dim_embed
        self.kdim = kdim if kdim is not None else dim_embed
        self.vdim = vdim if vdim is not None else dim_embed
        self._qkv_same_dim_embed = ((self.kdim == dim_embed) and (self.vdim == dim_embed))

        self.nhead = nhead
        self.dropout = dropout
        self.head_dim = dim_embed // nhead
        assert self.head_dim * nhead == self.dim_embed, "embedding dim must be divisible by num_heads"

        if self._qkv_same_dim_embed is False:
            self.q_proj_weight = nn.Parameter(torch.empty((dim_embed, dim_embed), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((dim_embed, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((dim_embed, self.vdim), **factory_kwargs))
            self.regiter_parameter('in_proj_weight', None)
        else:
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
            self.in_proj_weight = nn.Parameter(torch.empty((3 * dim_embed, dim_embed), **factory_kwargs))

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):

