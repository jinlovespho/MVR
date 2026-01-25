from math import sqrt
from re import L
from regex import B
import torch
import torch.nn as nn

from transformers import SwinModel
import torch
from torch import nn
from .lightningDiT import PatchEmbed, Mlp, NormAttention

from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *


# PHO
from einops import rearrange
from .model_utils import NormAttentionMVRM
from .rope import RotaryPositionEmbedding2D, PositionGetter
from torch.utils.checkpoint import checkpoint


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift, 
        with shift and scale repeated to match L_x if necessary.
    """
    # breakpoint()
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # if shift has shape (1,d) -> repeat_interleave(n, dim=0) -> (n,d)
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate, 
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        rope=None,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttentionMVRM(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            rope=rope,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, pos=None):
        
        if len(c.shape) < len(x.shape): # t
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:   # f
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:   # t
            '''
                c: concated (time and label) used for condition 
                for adaptive layer normalization, the shift, scale, and gate are computed from the condition c 
                    attention - shift,scale,gate
                    mlp - shift,scale, gate 
                ANOTHER UNDERSTANDING
                    layernorm -> for (n,d) tokens, it normalizes each tokens independently, 
                    first, for one vector token (1,d) normalize it as (v - v.mean() / v.std()), then apply scale SC and shift SH
                    this is done for all tokens, the same scale SC and shift SH is applied. 
                    However, for adaptiveLN, while (v-v.mean())/v.std() is the same, every tokens gets a different SC and SH value.
                    this value is learnable. Also the scale and shift equation is computed as x*(1+scale) + shift, instead of x*scale + shift
            '''
            # c: (b 1 1152)
            # all shift_xxx, scale_xxx, and gate_xxx have the same shape as (b1 1 1152)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), pos=pos), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x


class DiTwDDTHeadMVRM(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
    ):
        super().__init__()
        
        
        # PHO 
        self.num_cls_tkn = 1
        self.vit_patch_size=14  # for giant
        
        
        
        
        self.in_channels = in_channels      # 768
        self.out_channels = in_channels     # 768

        self.encoder_hidden_size = hidden_size[0]   # 1152
        self.decoder_hidden_size = hidden_size[1]   # 2048
        
        self.num_heads = [num_heads, num_heads] if isinstance(num_heads, int) else list(num_heads)  # (16,16)
        
        self.num_encoder_blocks = depth[0]  # 28
        self.num_decoder_blocks = depth[1]  # 2
        self.num_blocks = depth[0] + depth[1]   # 30
        
        self.use_rope = use_rope    # t
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # [1, 1] patch size for s , x embed 
        assert len(patch_size) == 2, f"patch size should be a list of two numbers, but got {patch_size}"
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        
        self.s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size   # 768
        s_input_size = input_size           # 32
        s_patch_size = self.s_patch_size    # 1
        
        x_input_size = input_size           # 32
        x_patch_size = self.x_patch_size    # 1
        self.x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size   # 768

        self.s_projector = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.s_embedder = PatchEmbed(
            img_size=s_input_size, 
            patch_size=s_patch_size, 
            in_chans=self.s_channel_per_token, 
            embed_dim=self.encoder_hidden_size, 
            bias=True
            )
        
        self.x_embedder = PatchEmbed(
            img_size=x_input_size, 
            patch_size=x_patch_size, 
            in_chans=self.x_channel_per_token, 
            embed_dim=self.decoder_hidden_size, 
            bias=True
            )
        
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, self.encoder_hidden_size, class_dropout_prob)
        # print(f"x_channel_per_token: {x_channel_per_token}, s_channel_per_token: {s_channel_per_token}")
        
        self.final_layer = DDTFinalLayer(self.decoder_hidden_size, 1, self.x_channel_per_token, use_rmsnorm=use_rmsnorm)
        # Will use fixed sin-cos embedding:
        if use_pos_embed:   # t
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_cls_tkn, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None
            
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        
        
        # # use rotary position encoding, borrow from EVA
        # if self.use_rope:
        #     enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
        #     self.enc_feat_rope = VisionRotaryEmbeddingFast(dim=enc_half_head_dim, pt_hw=self.s_embedder.grid_size, num_cls_tkn=1)
            
        #     dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
        #     self.dec_feat_rope = VisionRotaryEmbeddingFast(dim=dec_half_head_dim, pt_hw=self.x_embedder.grid_size)
        # else:
        #     self.feat_rope = None
            

        # self.blocks = nn.ModuleList([
        #     LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
        #                       enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
        #                       mlp_ratio=mlp_ratio,
        #                       use_qknorm=use_qknorm,
        #                       use_rmsnorm=use_rmsnorm,
        #                       use_swiglu=use_swiglu,
        #                       wo_shift=wo_shift,
        #                       ) for i in range(self.num_blocks)
        # ])



        # PHO rope 
        rope_freq=100
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()
        self.blocks = nn.ModuleList([
            LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                              enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                              mlp_ratio=mlp_ratio,
                              use_qknorm=use_qknorm,
                              use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu,
                              wo_shift=wo_shift,
                              rope=self.rope,
                              ) for i in range(self.num_blocks)
        ])
        


        
        # PHO handle camera cls tkn as well
        self.enc_cls_tkn = nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_size))
        self.dec_clk_tkn = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
        
        
        
        self.initialize_weights()


    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if self.use_pos_embed:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # PHO
            # self.pos_embed: b 972+1 1536
            # self.s_embedder.grid_size: (27, 36)
            # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.s_embedder.grid_size)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                                self.s_embedder.grid_size, 
                                                cls_token=True if self.num_cls_tkn > 0 else False, 
                                                extra_tokens=self.num_cls_tkn)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            # breakpoint()
            

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.x_channel_per_token        
        p = self.x_embedder.patch_size[0]
        h, w = self.s_embedder.grid_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


    def _prepare_rope(self, B, S, num_pH, num_pW, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(B * S, num_pH, num_pW, device=device)   # b*v n 2 
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)   # b v n 2 
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            pos = pos + 1   # to not account cls_tkn
            pos_special = torch.zeros(B * S, 1, 2).to(device).to(pos.dtype)
            pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
            pos = torch.cat([pos_special, pos], dim=2)
            pos_nodiff = pos_nodiff + 1     # to not account cls_tkn
            pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff


    def process_attention(self, x, c, block, attn_type="global", pos=None):
        
        # PHO DEBUG
        # if pos is not None:
        #     breakpoint()
            
        b, s, n, _ = x.shape
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")


        # x = block(x, c, pos=pos)
        # ✅ gradient checkpointing here
        def block_forward(x, c, pos):
            return block(x, c, pos=pos)

        if self.training:
            x = checkpoint(block_forward, x, c, pos)
        else:
            x = block(x, c, pos=pos)
            
        # x = checkpoint(block_forward, x, c, pos)


        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x


    def forward(self, x, t, y=None, s=None, mask=None):
        
        
        # hypersim 
        orig_H, orig_W = 768, 1024 
        model_H, model_W = 378, 504
        pH = pW = 14 
        num_pH = model_H//pH    # 27
        num_pW = model_W//pW    # 36
        


        
        cls_tkn = x[:,:,0:1]        # b v 1 d
        patch_tkns = x[:,:,1:]      # b v n d
        
        
        b, v, n, d = patch_tkns.shape
        x = rearrange(patch_tkns, 'b v (num_pH num_pW) d -> (b v) d num_pH num_pW', b=b, v=v, num_pH=num_pH, num_pW=num_pW)   # b v 3072 27 36
        
        
        # time condition
        t = self.t_embedder(t)          # b*v d        
        c = nn.functional.silu(t)       # b*v d
        

        # patch embedding 
        s = self.s_embedder(x)  # b*v 3072 27 36 -> b*v 972 1536
        
        
        # prepare encoder cls tkn
        enc_cls_tkn = self.enc_cls_tkn.expand(b, v, -1)                         # b v d
        enc_cls_tkn = enc_cls_tkn.reshape(b*v, -1, self.encoder_hidden_size)    # b*v 1 d 
        
        
        # concat encoder cls tkn w/ patch_tkns 
        s = torch.cat([enc_cls_tkn, s], dim=1)      # b*v n+1 d
        
        
        # add pos emb
        if self.use_pos_embed:  # t
            s = s + self.pos_embed  # b*v n+1 d
        
        
        # reshape to (b v n d)
        s = rearrange(s, '(b v) n d -> b v n d', b=b, v=v)
            

        # PHO prepare rope
        pos, pos_nodiff = self._prepare_rope(b, v, num_pH, num_pW, s.device)  # b v n+1 2
        g_pos = pos_nodiff
        l_pos = pos
        
        
        # ddt encoder forward apss     
        for i in range(self.num_encoder_blocks):    # len: 28
            # s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
            s = self.process_attention(s, c, self.blocks[i], 'local', pos=l_pos)    # b v n+1 d


        # reshape to (b*v n d)
        s = rearrange(s, 'b v n d -> (b v) n d')


        # broadcast t to s
        t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
        s = nn.functional.silu(t + s)
            
        
        # expand encoder output dimension to decoder dimension
        s = self.s_projector(s) # b*v n d1 -> b*v n d2
        
        
        # decoder input patch embedding
        x = self.x_embedder(x)  # b*v d h w -> b*v n d
        
        
        # prepare decoder cls tkn
        dec_cls_tkn = self.dec_clk_tkn.expand(b, v, -1)
        dec_cls_tkn = dec_cls_tkn.reshape(b*v, -1, self.decoder_hidden_size)    # b*v 1 d 
        
        
        # concat decoder cls_tkn w/ patch_tkns
        x = torch.cat([dec_cls_tkn, x], dim=1)  # b*v n+1 d
        
        
        # add decoder pos emb
        if self.use_pos_embed and self.x_pos_embed is not None: # f
            x = x + self.x_pos_embed
            
        
        # reshape to (b v n d)
        x = rearrange(x, '(b v) n d -> b v n d', b=b, v=v)
        
        
        # ddt decoder forward pass
        for i in range(self.num_encoder_blocks, self.num_blocks):   # len: 2
            x = self.process_attention(x, s, self.blocks[i], 'local', pos=l_pos)
            # x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)


        # reshape to (b*v n d)
        x = rearrange(x, 'b v n d -> (b v) n d')
        
        
        x = self.final_layer(x, s)  # last adaLN and restore to original dim: 2048 -> 768


        x = rearrange(x, '(b v) n d -> b v n d', b=b, v=v)
        
        
        return x

        cls_tkn = x[:,0]
        patch_tkns = x[:,1:]
        
        patch_tkns = self.unpatchify(patch_tkns)  # b n d -> b num_pH*num_pW pH*pW*c -> b c num_pH*pH num_pW*pW -> b c h w 
        
        
        return patch_tkns, cls_tkn   



    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:,
                              :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )

        return eps
