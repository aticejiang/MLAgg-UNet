# ===========================================================================
# MLLA使用SwinUNETR中的编码器策略，每个block后都跟随一个下采样
# 但没有使用UnetrBasicBlock作为bottleneck，而是直接通过跳跃连接+UnetrUpBlock
# ===========================================================================
# 使用block作为bottleneck
# 将AggregateAttention (TransNeXt)引入MLLABlock的Attention部分
# 实现不分组版本 (原版)
# MLLABlock中去掉两个cpe
# ===========================================================================
# 修改Stem的流程，下采样倍率设置为*2
# stage增加为5层，类似MedNext的设计样式
# ===========================================================================
# 分组
# SoftMax Attention
# ===========================================================================
# 简化上采样
# ===========================================================================
# 重新设计网络结构，把patch size设置为4，即主干开始四倍下采样。这样可以增加通道数，增加batch size等
# ===========================================================================
# 下采样patch size设置为2，通道改为48，head改为[2,4,8,16]
# ===========================================================================
# 引入Differential Transformer
# ===========================================================================

from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import timm
import timm.optim
import timm.scheduler


class nnUNetTrainer_MLAgg_2D_dt_MS(nnUNetTrainer):
    ''' With no DeepSupervision
    Outputs of decoder network is a torch.Tensor.
    If with DeepSupervision, this is a list of torch.Tensor. But haven't been deployed in this file.
    '''
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 5e-4
        self.weight_decay = 3e-5
        # self.initial_lr = 5e-5  # 1e-4
        # self.weight_decay = 1e-2  # 1e-3
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
        model = MLLA_Uper(
            img_size=configuration_manager.patch_size,
            patch_size=2,
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            embed_dim=96,               # 96,
            depths=[2, 2, 2, 2],
            num_heads=[2, 4, 8, 16],     # [3, 6, 12, 24]
            mlp_ratio=2,               # 4.
            qkv_bias=True,
            drop_rate=0.,
            dropout_path_rate=0.1,
            sr_ratio=[16, 8, 4, 2],
            norm_layer=nn.LayerNorm,
            ape=False,
            use_checkpoint=False,
            # ====================
            deep_supervision=enable_deep_supervision,
        )
        print("MLLAggAtt_Upernet_neck_new: {}".format(model))
        # model.apply(InitWeights_He(1e-2))
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        self.network.deep_supervision = enabled

    def _get_deep_supervision_scales(self):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            [[1,1],[2,2],[2,2],[2,2],[2,2]]), axis=0))
        return deep_supervision_scales

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        # weights = np.delete(weights, 1)

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
    #     # lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
    #     lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer, t_initial=self.num_epochs, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5)
    #     return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-4  # 1e-8 might cause nans in fp16
        )
        # lr_scheduler = None
        lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer, t_initial=self.num_epochs, lr_min=1e-6,
                                                        warmup_t=10, warmup_lr_init=1e-4)
        return optimizer, lr_scheduler

    def plot_network_architecture(self):
        pass


# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------
# MLAgg-UNet: Advancing Medical Image Segmentation with Efficient Transformer and Mamba-Inspired Multi-Scale Sequence
# Modified by Jiaxu Jiang
# -----------------------------------------------------------------------

import sys
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections.abc import Sequence
# from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from nnunetv2.training.nnUNetTrainer.variants.mamba.MambaSkip import VSS_Conv_Layer
from flash_attn import flash_attn_func              # If flash attention is not required, comment it out. And use diff_attn in line 762-777 instead.


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MedNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 dim='3d',
                 grn=False
                 ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='3d', grn=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim,
                         grn=grn)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='3d', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim,
                         grn=grn)

        self.resample_do_res = do_res

        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class PatchMerging(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 norm_type: str = 'group',
                 dim='3d',
                 do_res=False):
        super().__init__()
        self.resample_do_res = do_res

        self.dim = dim
        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.reduction = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            # groups=out_channels,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

    def forward(self, x):
        x1 = x
        x1 = self.norm(x1)
        x1 = self.reduction(x1)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class PatchExpand(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 7,
                 norm_type: str = 'group',
                 dim='3d',
                 do_res=False,
                 ):

        super().__init__()
        self.resample_do_res = do_res

        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            # groups=out_channels,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.norm(x1)
        x1 = self.conv1(x1)

        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, local=True, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, fixed_pool_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        # self.head_dim = dim // num_heads
        self.head_dim = dim // num_heads // 2
        self.scale = self.head_dim ** -0.5
        self.local = local

        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        if local:
            assert window_size % 2 == 1, "window size must be odd"
            self.window_size = window_size
            self.local_len = window_size ** 2

            self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)

            # Generate padding_mask && sequnce length scale
            local_seq_length, padding_mask = get_seqlen_and_mask(input_resolution, window_size)
            self.register_buffer("padding_mask", padding_mask, persistent=False)
        else:
            self.sr_ratio = sr_ratio

            if fixed_pool_size is None:
                self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
            else:
                assert fixed_pool_size < min(input_resolution), \
                    f"The fixed_pool_size {fixed_pool_size} should be less than the shorter side of input resolution {input_resolution} to ensure pooling works correctly."
                self.pool_H, self.pool_W = fixed_pool_size, fixed_pool_size
            self.pool_len = self.pool_H * self.pool_W

            # Components to generate pooled features.
            self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # LePE
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W

        # Generate queries
        q = self.q(x).reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q * self.scale

        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        v_pe = v_local

        if self.local:
            # Generate unfolded keys and values
            # kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
            # k_local, v_local = self.unfold(kv_local).reshape(
            #     B, 4 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
            k_local = k_local.permute(0, 2, 1).reshape(B, -1, H, W)
            k_local = self.unfold(k_local).reshape(
                B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3)
            v_local = v_local.permute(0, 2, 1).reshape(B, -1, H, W)
            v_local = self.unfold(v_local).reshape(
                B, self.num_heads, 2 * self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3)

            # Compute local similarity
            attn_local = ((q.unsqueeze(-2) @ k_local).squeeze(-2)).masked_fill(self.padding_mask, float('-inf'))
            attn_local = attn_local.softmax(dim=-1)

            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
            attn_local = attn_local.view(B, self.num_heads, 2, N, self.local_len)
            attn_local = attn_local[:, :, 0] - lambda_full * attn_local[:, :, 1]

            x = (attn_local.unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)      # B, num_head, N, 2*head_dim
            x = self.subln(x)
            x = x * (1 - self.lambda_init)
        else:
            del k_local, v_local
            # Generate pooled features
            x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
            x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
            x_ = self.norm(x_)

            # Generate pooled keys and values
            # kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # k_pool, v_pool = kv_pool.chunk(2, dim=1)
            k_pool, v_pool = self.kv(x_).chunk(2, dim=-1)
            k_pool = k_pool.reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim)
            v_pool = v_pool.reshape(B, self.pool_len, self.num_heads, 2 * self.head_dim)

            # Compute pooled similarity
            ## flash_attn
            q = q.permute(0, 2, 1, 3).reshape(B, N, self.num_heads, 2, self.head_dim)
            k_pool = k_pool.reshape(B, self.pool_len, self.num_heads, 2, self.head_dim)
            q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
            k1, k2 = k_pool[:, :, :, 0], k_pool[:, :, :, 1]
            #### for packages that support different qk/v dimensions
            # attn1 = flash_attn_func(q1, k1, v_pool, causal=True)
            # attn2 = flash_attn_func(q2, k2, v_pool, causal=True)

            #### for that do not support different qk/v dimensions
            v_pool = v_pool.reshape(B, self.pool_len, self.num_heads, 2, self.head_dim)
            v1, v2 = v_pool[:, :, :, 0], v_pool[:, :, :, 1]
            attn11 = flash_attn_func(q1, k1, v1, causal=False)
            attn12 = flash_attn_func(q1, k1, v2, causal=False)
            attn1 = torch.cat([attn11, attn12], dim=-1)

            attn21 = flash_attn_func(q2, k2, v1, causal=False)
            attn22 = flash_attn_func(q2, k2, v2, causal=False)
            attn2 = torch.cat([attn21, attn22], dim=-1)

            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
            x = attn1 - lambda_full * attn2

            x = x.transpose(1, 2)
            x = self.subln(x)
            x = x * (1 - self.lambda_init)

            ## diff_attn (no flash)
            # k_pool = k_pool.permute(0, 2, 1, 3)
            # v_pool = v_pool.permute(0, 2, 1, 3)
            #
            # attn_pool = q @ k_pool.transpose(-2, -1)
            # attn_pool = attn_pool.softmax(dim=-1)
            #
            # lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
            # lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            # lambda_full = lambda_1 - lambda_2 + self.lambda_init
            # attn_pool = attn_pool.view(B, self.num_heads, 2, N, self.pool_len)
            # attn_pool = attn_pool[:, :, 0] - lambda_full * attn_pool[:, :, 1]
            #
            # x = attn_pool @ v_pool
            # x = self.subln(x)
            # x = x * (1 - self.lambda_init)

        x = x.transpose(1, 2).reshape(B, N, C)

        v_pe = v_pe.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.lepe(v_pe).permute(0, 2, 3, 1).reshape(B, N, C)

        return x


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # LePE
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)
        q = q * self.scale

        # Calculate attention map using sequence length scaled cosine attention and query embedding
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(B, N, C)
        return x


class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0., sr_ratio=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        # self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio == 1:
            self.attn = Attention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        else:
            self.attn = nn.ModuleList()
            self.attn.append(
                AggregatedAttention(dim=dim//2, input_resolution=input_resolution, num_heads=num_heads//2, local=True,
                                    qkv_bias=qkv_bias, sr_ratio=sr_ratio)
            )
            self.attn.append(
                AggregatedAttention(dim=dim//2, input_resolution=input_resolution, num_heads=num_heads//2, local=False,
                                    qkv_bias=qkv_bias, sr_ratio=sr_ratio)
            )
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.mlp = Mlp(hidden_size=dim, mlp_dim=int(dim * mlp_ratio), act="GELU", dropout_rate=drop,
        #                dropout_mode="swin")

    def forward(self, x):
        H, W = self.input_resolution
        B, C, h_, w_ = x.shape
        assert (H == h_) and (W == w_), "input feature has wrong size"
        L = H * W
        x = x.reshape(B, C, L).transpose(1, 2)

        # x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # AggregatedAttention or Attention
        if self.sr_ratio == 1:
            x = self.attn(x, H, W)
        else:
            x, z = torch.chunk(x, chunks=2, dim=-1)
            x = self.attn[0](x, H, W)
            z = self.attn[1](z, H, W)
            x = torch.cat([x, z], dim=-1)
            del z

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        # x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., sr_ratio=1, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                      sr_ratio=sr_ratio, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample([input_resolution[0] * 2, input_resolution[1] * 2], dim=dim // 2)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wh, Ww)
        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 patch_size=(2, 2),
                 in_chans=4,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # self.proj = project(in_chans, embed_dim, 2, 1, nn.GELU, nn.LayerNorm, True)
        stride1 = [2, 2]
        stride2 = [patch_size[0] // 2, patch_size[1] // 2]
        self.proj1 = project(in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        x = self.proj1(x)  # B C Wh Ww
        x = self.proj2(x)  # B C Wh Ww
        # x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Wh, Ww)

        return x


class MLLA_Enc(nn.Module):
    r""" MLLA
        A PyTorch impl of : `Demystify Mamba in Vision: A Linear Attention Perspective`

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each MLLA layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 sr_ratio=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.patch_size = [patch_size, patch_size]
        self.patch_norm = False
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
        )
        if isinstance(img_size, int):
            patches_resolution = [img_size//patch_size, img_size//patch_size]
        else:
            patches_resolution = list(i//patch_size for i in img_size)
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               sr_ratio=sr_ratio[i_layer],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.downs = nn.ModuleList()
        for i_layer in range(self.num_layers-1):
            self.downs.append(
                MedNeXtDownBlock(
                    in_channels=int(embed_dim * 2 ** i_layer),
                    out_channels=int(embed_dim * 2 ** (i_layer+1)),
                    exp_r=self.mlp_ratio,
                    kernel_size=3,
                    do_res=True,
                    norm_type='group',
                    dim='2d'
                )
                # PatchMerging(
                #     in_channels=int(embed_dim * 2 ** i_layer),
                #     out_channels=int(embed_dim * 2 ** (i_layer + 1)),
                #     kernel_size=3,
                #     norm_type='group',
                #     dim='2d',
                #     do_res=True
                # )
            )

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def proj_out(self, x, input_resolution, normalize=False):
        b, n, c = x.shape
        assert n == input_resolution[0] * input_resolution[1]
        x = x.view(b, input_resolution[0], input_resolution[1], c)
        if normalize:
            x = F.layer_norm(x, [c])

        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x, normalize=True):
        outs = [x]
        patches_resolution = self.patches_resolution
        x = self.patch_embed(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)
            if i < self.num_layers-1:
                x = self.downs[i](x)
            patches_resolution = [patches_resolution[0] // 2, patches_resolution[1] // 2]

        return outs

    def forward(self, x, normalize=True):
        outs = self.forward_features(x, normalize=normalize)
        return outs



class MLLA_Uper(nn.Module):
    def __init__(
            self,
            img_size: Sequence[int] | int,
            patch_size: Sequence[int] | int,
            in_channels: int,
            out_channels: int,
            embed_dim: int = 96,
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            sr_ratio: list[int] = [8, 4, 2, 1],
            normalize: bool = True,
            norm_layer=nn.LayerNorm,
            ape=False,
            use_checkpoint: bool = False,
            # ==========================
            spatial_dims: str = '2d',
            norm_type: str = 'group',
            do_res: bool = True,
            deep_supervision: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.deep_supervision = deep_supervision

        self.mlla = MLLA_Enc(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=dropout_path_rate,
            sr_ratio=sr_ratio,
            norm_layer=norm_layer,
            ape=ape,
            use_checkpoint=use_checkpoint,
        )

        self.mambaskip = VSS_Conv_Layer(
            [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8],
            embed_dim // 2,
            depth=1,
            drop_path=0.1,
            use_checkpoint=False,
        )

        grn = False

        # self.up_2 = MedNeXtUpBlock(
        #     in_channels=8 * embed_dim,
        #     out_channels=4 * embed_dim,
        #     exp_r=mlp_ratio,
        #     kernel_size=3,
        #     do_res=do_res,
        #     norm_type=norm_type,
        #     dim=spatial_dims,
        #     grn=grn
        # )
        self.up_2 = PatchExpand(
            in_channels=8 * embed_dim,
            out_channels=4 * embed_dim,
            kernel_size=3,
            do_res=do_res,
            norm_type=norm_type,
            dim=spatial_dims,
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=embed_dim * 4,
                out_channels=embed_dim * 4,
                exp_r=mlp_ratio,
                kernel_size=3,
                do_res=do_res,
                norm_type=norm_type,
                dim=spatial_dims,
                grn=grn
            )
            for i in range(depths[-2])]
                                         )

        # self.up_1 = MedNeXtUpBlock(
        #     in_channels=4 * embed_dim,
        #     out_channels=2 * embed_dim,
        #     exp_r=mlp_ratio,
        #     kernel_size=3,
        #     do_res=do_res,
        #     norm_type=norm_type,
        #     dim=spatial_dims,
        #     grn=grn
        # )
        self.up_1 = PatchExpand(
            in_channels=4 * embed_dim,
            out_channels=2 * embed_dim,
            kernel_size=3,
            do_res=do_res,
            norm_type=norm_type,
            dim=spatial_dims,
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=embed_dim * 2,
                out_channels=embed_dim * 2,
                exp_r=mlp_ratio,
                kernel_size=3,
                do_res=do_res,
                norm_type=norm_type,
                dim=spatial_dims,
                grn=grn
            )
            for i in range(depths[-3])]
                                         )

        # self.up_0 = MedNeXtUpBlock(
        #     in_channels=2 * embed_dim,
        #     out_channels=embed_dim,
        #     exp_r=mlp_ratio,
        #     kernel_size=3,
        #     do_res=do_res,
        #     norm_type=norm_type,
        #     dim=spatial_dims,
        #     grn=grn
        # )
        self.up_0 = PatchExpand(
            in_channels=2 * embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            do_res=do_res,
            norm_type=norm_type,
            dim=spatial_dims,
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                exp_r=mlp_ratio,
                kernel_size=3,
                do_res=do_res,
                norm_type=norm_type,
                dim=spatial_dims,
                grn=grn
            )
            for i in range(depths[-4])]
                                         )
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=embed_dim//2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder0 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=embed_dim,
            out_channels=embed_dim//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.out_0 = OutBlock(in_channels=embed_dim//2, n_classes=out_channels, dim=spatial_dims)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=embed_dim, n_classes=out_channels, dim=spatial_dims)
            self.out_2 = OutBlock(in_channels=embed_dim * 2, n_classes=out_channels, dim=spatial_dims)
            self.out_3 = OutBlock(in_channels=embed_dim * 4, n_classes=out_channels, dim=spatial_dims)
            self.out_4 = OutBlock(in_channels=embed_dim * 8, n_classes=out_channels, dim=spatial_dims)

    def forward(self, x_in):
        hidden_states_out = self.mlla(x_in, self.normalize)
        hidden_states_out[1:] = self.mambaskip(hidden_states_out[1:])
        if self.deep_supervision:
            x_ds_4 = self.out_4(hidden_states_out[4])

        x_up_2 = self.up_2(hidden_states_out[4])
        dec_x = hidden_states_out[3] + x_up_2
        x = self.dec_block_2(dec_x)

        if self.deep_supervision:
            x_ds_3 = self.out_3(x)
        del hidden_states_out[4], hidden_states_out[3], x_up_2

        x_up_1 = self.up_1(x)
        dec_x = hidden_states_out[2] + x_up_1
        x = self.dec_block_1(dec_x)
        if self.deep_supervision:
            x_ds_2 = self.out_2(x)
        del hidden_states_out[2], x_up_1

        x_up_0 = self.up_0(x)
        dec_x = hidden_states_out[1] + x_up_0
        x = self.dec_block_0(dec_x)
        if self.deep_supervision:
            x_ds_1 = self.out_1(x)
        del hidden_states_out[1], x_up_0

        enc0 = self.encoder0(hidden_states_out[0])
        x = self.decoder0(x, enc0)
        del enc0

        x = self.out_0(x)

        if self.deep_supervision:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


if __name__ == '__main__':
    # model = MLLA_Enc(
    #     img_size=[320, 384],
    #     patch_size=4,
    #     in_chans=1,
    #     num_classes=14,
    #     embed_dim=96,
    #     depths=[2, 2, 2, 2],
    #     num_heads=[3, 6, 12, 24],
    #     mlp_ratio=4.,
    #     qkv_bias=True,
    #     drop_rate=0.,
    #     drop_path_rate=0.1,
    #     norm_layer=nn.LayerNorm,
    #     ape=False,
    #     use_checkpoint=False,
    # ).cuda()
    model = MLLA_Uper(
        img_size=[320, 384],
        patch_size=4,
        in_channels=1,
        out_channels=14,
        embed_dim=32,  # 96,
        depths=[2, 2, 2, 2, 2],
        num_heads=[2, 4, 8, 16, 32],  # [3, 6, 12, 24]
        mlp_ratio=2,  # 4.
        qkv_bias=True,
        drop_rate=0.,
        dropout_path_rate=0.1,
        sr_ratio=[32, 16, 8, 4, 2],
        norm_layer=nn.LayerNorm,
        ape=False,
        use_checkpoint=False,
        # ====================
        deep_supervision=True,
    ).cuda()
    input_t = torch.rand(2, 1, 320, 384).cuda()
    outputs = model(input_t)
    print(outputs[0].shape)
