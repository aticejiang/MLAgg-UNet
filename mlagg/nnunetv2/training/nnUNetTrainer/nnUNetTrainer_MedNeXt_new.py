# =============================================================
# 对原版blocks做出改动，使其上下采样过程可以在不同方向使用不同步长的strides
# =============================================================

from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.network_architecture.mednextv1.MedNextV1_new import MedNeXt

import torch
from torch import nn
import numpy as np
import timm
import timm.optim
import timm.scheduler


class nnUNetTrainer_MedNeXt_new(nnUNetTrainer):
    ''' With no DeepSupervision
    Outputs of decoder network is a torch.Tensor.
    If with DeepSupervision, this is a list of torch.Tensor. But
    haven't been deployed in this file.
    '''
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-3
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
        strides = configuration_manager.pool_op_kernel_sizes[1:5]
        strides = strides + strides[::-1]       # [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        # network class name!!
        if len(configuration_manager.patch_size) == 3:
            model = MedNeXt(
                in_channels=num_input_channels,
                n_channels=32,
                n_classes=label_manager.num_segmentation_heads,
                exp_r=2,  # Expansion ratio as in Swin Transformers
                kernel_size=3,  # Can test kernel_size
                deep_supervision=enable_deep_supervision,  # Can be used to test deep supervision
                do_res=True,  # Can be used to individually test residual connection
                do_res_up_down=True,
                checkpoint_style=None,       # checkpoint_style in [None, 'outside_block']
                block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                strides=strides,
            )
        elif len(configuration_manager.patch_size) == 2:
            model = MedNeXt(
                in_channels=num_input_channels,
                n_channels=32,
                n_classes=label_manager.num_segmentation_heads,
                exp_r=2,  # Expansion ratio as in Swin Transformers
                kernel_size=3,  # Can test kernel_size
                deep_supervision=enable_deep_supervision,  # Can be used to test deep supervision
                do_res=True,  # Can be used to individually test residual connection
                do_res_up_down=True,
                checkpoint_style=None,      # checkpoint_style in [None, 'outside_block']
                block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                strides=strides,
                dim='2d'
            )
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        print("UMambaEnc: {}".format(model))
        # model.apply(InitWeights_He(1e-2))
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        self.network.do_ds = enabled

    def _get_deep_supervision_scales(self):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            self.configuration_manager.pool_op_kernel_sizes[:6]), axis=0))[:-1]
        return deep_supervision_scales

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
                                                        warmup_t=10, warmup_lr_init=1e-5)
        return optimizer, lr_scheduler


if __name__ == '__main__':
    # =================3d
    # input_t = torch.randn((1, 4, 128, 128, 128)).cuda()
    # model = MedNeXt(
    #     in_channels=4,
    #     n_channels=32,
    #     n_classes=3,
    #     exp_r=2,  # Expansion ratio as in Swin Transformers
    #     kernel_size=3,  # Can test kernel_size
    #     deep_supervision=True,  # Can be used to test deep supervision
    #     do_res=True,  # Can be used to individually test residual connection
    #     do_res_up_down=True,
    #     block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    # ).cuda()
    # outputs = model(input_t)

    # =================2d
    # input_t = torch.randn((1, 4, 128, 128, 128)).cuda()
    # input_t = torch.randn((1, 4, 40, 192, 224)).cuda()
    # model = MedNeXt(
    #     in_channels=4,
    #     n_channels=32,
    #     n_classes=3,
    #     exp_r=2,  # Expansion ratio as in Swin Transformers
    #     kernel_size=3,  # Can test kernel_size
    #     deep_supervision=True,  # Can be used to test deep supervision
    #     do_res=True,  # Can be used to individually test residual connection
    #     do_res_up_down=True,
    #     block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
    #     strides=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    #     dim='3d'
    # ).cuda()
    # outputs = model(input_t)
    # print(outputs.shape)

    from calflops import calculate_flops
    from thop import profile
    from thop import clever_format

    strides = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    model = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=14,
        exp_r=2,  # Expansion ratio as in Swin Transformers
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=True,
        checkpoint_style=None,  # checkpoint_style in [None, 'outside_block']
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        strides=strides,
        dim='2d'
    )

    dummy_input = torch.rand((1, 1, 320, 384))  # .to(device)

    macs, params = profile(model, (dummy_input,))
    flops = 2 * macs
    macs, flops, params = clever_format([macs, flops, params], "%.3f")
    print('FLOPs: ', flops, 'MACs:', macs, 'params: ', params)

    # flops, macs, params = calculate_flops(model=model,
    #                                       input_shape=(1, 1, 320, 384),
    #                                       output_as_string=True,
    #                                       output_precision=4)
    # print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


# =======================================================================
# Configurations from MedNeXt, which is using nnunetv1
# =======================================================================

# class MedNeXt(MedNeXt_Orig, SegmentationNetwork):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
#         self.conv_op = nn.Conv3d
#         self.inference_apply_nonlin = softmax_helper
#         self.input_shape_must_be_divisible_by = 2 ** 5
#         self.num_classes = kwargs['n_classes']
#         # self.do_ds = False        Already added this in the main class
#
#
# class nnUNetTrainerV2_Optim_and_LR(nnUNetTrainerV2):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 1e-3
#
#     def process_plans(self, plans):
#         super().process_plans(plans)
#         # Please don't do this for nnunet. This is only for MedNeXt for all the DS to be used
#         num_of_outputs_in_mednext = 5
#         self.net_num_pool_op_kernel_sizes = [[2, 2, 2] for i in range(num_of_outputs_in_mednext + 1)]
#
#     def initialize_optimizer_and_scheduler(self):
#         assert self.network is not None, "self.initialize_network must be called first"
#         self.optimizer = torch.optim.AdamW(self.network.parameters(),
#                                            self.initial_lr,
#                                            weight_decay=self.weight_decay,
#                                            eps=1e-4  # 1e-8 might cause nans in fp16
#                                            )
#         self.lr_scheduler = None

# class nnUNetTrainerV2_MedNeXt_S_kernel3(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=2,  # Expansion ratio as in Swin Transformers
#             kernel_size=3,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# class nnUNetTrainerV2_MedNeXt_B_kernel3(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
#             kernel_size=3,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# class nnUNetTrainerV2_MedNeXt_M_kernel3(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
#             kernel_size=3,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
#             checkpoint_style='outside_block'
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# class nnUNetTrainerV2_MedNeXt_L_kernel3(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
#             # exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
#             kernel_size=3,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             # block_counts = [6,6,6,6,4,2,2,2,2],
#             block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
#             checkpoint_style='outside_block'
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# # Kernels of size 5
# class nnUNetTrainerV2_MedNeXt_S_kernel5(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=2,  # Expansion ratio as in Swin Transformers
#             kernel_size=5,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_S_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 1e-4
#
#
# class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_S_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 25e-5
#
#
# class nnUNetTrainerV2_MedNeXt_B_kernel5(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
#             kernel_size=5,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 5e-4
#
#
# class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_B_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 25e-5
#
#
# class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 1e-4
#
#
# class nnUNetTrainerV2_MedNeXt_M_kernel5(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
#             kernel_size=5,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
#             checkpoint_style='outside_block'
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 5e-4
#
#
# class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_M_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 25e-5
#
#
# class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 1e-4
#
#
# class nnUNetTrainerV2_MedNeXt_L_kernel5(nnUNetTrainerV2_Optim_and_LR):
#
#     def initialize_network(self):
#         self.network = MedNeXt(
#             in_channels=self.num_input_channels,
#             n_channels=32,
#             n_classes=self.num_classes,
#             exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
#             kernel_size=5,  # Can test kernel_size
#             deep_supervision=True,  # Can be used to test deep supervision
#             do_res=True,  # Can be used to individually test residual connection
#             do_res_up_down=True,
#             # block_counts = [6,6,6,6,4,2,2,2,2],
#             block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
#             checkpoint_style='outside_block'
#         )
#
#         if torch.cuda.is_available():
#             self.network.cuda()
#
#
# class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 5e-4
#
#
# class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_L_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 25e-5
#
#
# class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initial_lr = 1e-4