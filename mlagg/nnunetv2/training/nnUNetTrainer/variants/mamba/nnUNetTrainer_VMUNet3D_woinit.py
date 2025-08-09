from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.checkpoint as checkpoint
import timm
import timm.optim
import timm.scheduler


class nnUNetTrainer_VMUNet3D_woinit(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-4         # 1e-4
        self.weight_decay = 1e-2        # 1e-3
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 500  # 1000
        self.current_epoch = 0

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)

        # network class name!!
        model = VSSM_UNETR(
            strides=[[2, 4, 4], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            in_chans=num_input_channels,
            classes=label_manager.num_segmentation_heads,
            depths=[2, 2, 2, 2],
            dims=[96, 192, 384, 768],
            use_checkpoint=False,
        )
        # model.apply(InitWeights_He(1e-2))
        return model

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
    #     lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
    #     return optimizer, lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer, t_initial=self.num_epochs, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5)
        return optimizer, lr_scheduler

    def plot_network_architecture(self):
        pass



import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class PatchEmbed3D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int or list): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, list or tuple):
            pass
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 4, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class PatchEmbed3Dv2(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int or list): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, list or tuple):
            pass
        stride1 = [p // 2 for p in patch_size]
        stride2 = [patch_size[i] // stride1[i] for i in range(len(patch_size))]
        self.proj = nn.Sequential(
                        nn.Conv3d(in_chans, embed_dim // 2, kernel_size=3, stride=stride1, padding=1),
                        Permute(0, 2, 3, 4, 1),
                        norm_layer(embed_dim // 2),
                        Permute(0, 4, 1, 2, 3),
                        nn.GELU(),
                        nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=stride2, padding=1),
                        Permute(0, 2, 3, 4, 1),
                        norm_layer(embed_dim),
                    )

    def forward(self, x):
        return self.proj(x)


class PatchMerging3D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        stride (int or list): stride size
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, stride, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        out_dim = out_dim or 2 * dim
        self.reduction = nn.Sequential(
                            Permute(0, 4, 1, 2, 3),
                            nn.Conv3d(dim, out_dim, kernel_size=3, stride=stride, padding=1),
                            Permute(0, 2, 3, 4, 1),
                            norm_layer(out_dim),
                        )

    def forward(self, x):
        return self.reduction(x)


class PatchExpanding3D(nn.Module):

    def __init__(self, dim, out_dim=None, upsample_kernel_size=2, norm_layer=nn.LayerNorm):
        super().__init__()
        out_dim = out_dim or dim // 2
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims=3,
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        x = Permute(0, 4, 1, 2, 3)(x)
        x = self.transp_conv(x)
        x = Permute(0, 2, 3, 4, 1)(x)
        x = self.norm(x)
        return x


class SS3D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in range(12)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=12, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs) for i in range(12)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=12, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=12, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=12, merge=True)  # (K=12, D, N)
        self.Ds = self.D_init(self.d_inner, copies=12, merge=True)  # (K=12, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, D, H, W = x.shape
        L = D * H * W
        K = 12

        x_dhwdwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)],
                               dim=1).view(B, 2, -1, L)
        x_hdwhwd = torch.stack([x.permute(0,1,3,2,4).contiguous().view(B, -1, L), x.permute(0,1,3,4,2).contiguous().view(B, -1, L)],
                               dim=1).view(B, 2, -1, L)
        x_wdhwhd = torch.stack([x.permute(0,1,4,2,3).contiguous().view(B, -1, L), x.permute(0,1,4,3,2).contiguous().view(B, -1, L)],
                               dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_dhwdwh, x_hdwhwd, x_wdhwhd, torch.flip(x_dhwdwh, dims=[-1]),
                        torch.flip(x_hdwhwd, dims=[-1]),
                        torch.flip(x_wdhwhd, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        out_y[:, 6:12] = torch.flip(out_y[:, 6:12], dims=[-1]).view(B, 6, -1, L)
        out_y[:, 1] = out_y[:, 1].view(B, -1, D, W, H).permute(0,1,2,4,3).contiguous().view(B, -1, L)         # dwh_y
        out_y[:, 2] = out_y[:, 2].view(B, -1, H, D, W).permute(0,1,3,2,4).contiguous().view(B, -1, L)         # hdw_y
        out_y[:, 3] = out_y[:, 3].view(B, -1, H, W, D).permute(0,1,4,2,3).contiguous().view(B, -1, L)         # hwd_y
        out_y[:, 4] = out_y[:, 4].view(B, -1, W, D, H).permute(0,1,3,4,2).contiguous().view(B, -1, L)         # wdh_y
        out_y[:, 5] = out_y[:, 5].view(B, -1, W, H, D).permute(0,1,4,3,2).contiguous().view(B, -1, L)         # whd_y
        out_y[:, 7] = out_y[:, 7].view(B, -1, D, W, H).permute(0,1,2,4,3).contiguous().view(B, -1, L)      # invdwh_y
        out_y[:, 8] = out_y[:, 8].view(B, -1, H, D, W).permute(0,1,3,2,4).contiguous().view(B, -1, L)      # invhdw_y
        out_y[:, 9] = out_y[:, 9].view(B, -1, H, W, D).permute(0,1,4,2,3).contiguous().view(B, -1, L)      # invhwd_y
        out_y[:, 10] = out_y[:, 10].view(B, -1, W, D, H).permute(0,1,3,4,2).contiguous().view(B, -1, L)      # invwdh_y
        out_y[:, 11] = out_y[:, 11].view(B, -1, W, H, D).permute(0,1,4,3,2).contiguous().view(B, -1, L)      # invwhd_y

        return out_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, D, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, D, h, w, d)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))  # (b, d, D, h, w)
        # y1, y2, y3, y4 = self.forward_core(x)
        y = self.forward_core(x)
        assert y[0].dtype == torch.float32
        y = torch.sum(y, dim=1)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, D, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class DWConv3D(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3D, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        assert len(x.shape) == 5
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv3D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        #### x is shape of [B, D, H, W, C]
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            mlp_ratio=4.,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.norm = norm_layer(hidden_dim)
        self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=expand, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.norm(input)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            d_state=16,
            expand=2.,
            mlp_ratio=4.,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                expand=expand,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                **kwargs,
            )
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        return x


class VSSM(nn.Module):
    def __init__(
        self,
        strides=[[2,4,4],[2,2,2],[2,2,2],[1,2,2]],
        in_chans=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,
        # =========================
        posembed=False,
        imgsize=[40, 192, 224],
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.pos_embed = self._pos_embed(dims[0], strides[0], imgsize) if posembed else None

        self.downsample = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                if patchembed_version == 'v1':
                    self.downsample.append(PatchEmbed3D(patch_size=strides[0], in_chans=in_chans, embed_dim=dims[0]))
                elif patchembed_version == 'v2':
                    self.downsample.append(PatchEmbed3Dv2(patch_size=strides[0], in_chans=in_chans, embed_dim=dims[0]))
            else:
                self.downsample.append(PatchMerging3D(stride=strides[i_layer], dim=dims[i_layer-1], out_dim=dims[i_layer]))

            self.layers.append(VSSLayer(dims[i_layer],
                                        depths[i_layer],
                                        drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        # =========================
                                        d_state=ssm_d_state,
                                        expand=ssm_ratio,
                                        # =========================
                                        mlp_ratio=mlp_ratio,
                                        mlp_act_layer=mlp_act_layer,
                                        mlp_drop_rate=mlp_drop_rate,
                                        # =========================
                                        use_checkpoint=use_checkpoint,))

        # self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_depth, patch_height, patch_width = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_depth, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    # def _init_weights(self, m: nn.Module):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        outs = []
        for i in range(self.num_layers):
            x = self.downsample[i](x)
            if i == 0:
                if self.pos_embed is not None:
                    pos_embed = self.pos_embed.permute(0, 2, 3, 4, 1)
                    x = x + pos_embed

            x = self.layers[i](x)
            outs.append(x.permute(0, 4, 1, 2, 3).contiguous())
        return outs


class VSSM_UNETR(nn.Module):
    def __init__(
            self,
            strides=[[2, 4, 4], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            in_chans=3,
            classes=4,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            # =========================
            posembed=False,
            imgsize=[40, 192, 224],
            **kwargs,
    ):
        super().__init__()
        self.vssm = VSSM(strides=strides,
                         in_chans=in_chans,
                         depths=depths,
                         dims=dims,
                         ssm_d_state=ssm_d_state,
                         ssm_ratio=ssm_ratio,
                         mlp_ratio=mlp_ratio,
                         mlp_act_layer=mlp_act_layer,
                         mlp_drop_rate=mlp_drop_rate,
                         drop_path_rate=drop_path_rate,
                         patchembed_version=patchembed_version,
                         use_checkpoint=use_checkpoint,
                         posembed=posembed,
                         imgsize=imgsize,)

        spatial_dims = 3
        norm_name = "instance"

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            out_channels=dims[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            out_channels=dims[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[3],
            out_channels=dims[2],
            kernel_size=3,
            upsample_kernel_size=strides[3],
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[2],
            out_channels=dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[0],
            kernel_size=3,
            upsample_kernel_size=strides[0],
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=dims[0], out_channels=classes)

    def forward(self, x_in):
        outs = self.vssm(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder4(enc_hidden, enc4)
        dec2 = self.decoder3(dec3, enc3)
        dec1 = self.decoder2(dec2, enc2)
        dec0 = self.decoder1(dec1, enc1)

        return self.out(dec0)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VSSM_UNETR(
        strides=[[2, 4, 4], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        in_chans=3,
        classes=4,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        use_checkpoint=False,
    ).to(device)
    inputs = torch.randn((1, 3, 40, 192, 224))
    outputs = model(inputs.to(device))
    print(outputs.shape)

