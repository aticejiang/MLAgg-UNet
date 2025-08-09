from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from torch import nn
import torch
from monai.networks.nets import SwinUNETR
from torch.optim.lr_scheduler import CosineAnnealingLR


class nnUNetTrainer_SwinUNETR_2d(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 5e-4
        self.weight_decay = 1e-3
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
        model = SwinUNETR(
            img_size=configuration_manager.patch_size,
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            feature_size=96,
            spatial_dims= 2,
            use_checkpoint=True,
        )
        return model

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
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler



if __name__ == '__main__':
    # from calflops import calculate_flops
    from thop import profile
    from thop import clever_format

    model = SwinUNETR(
        img_size=[320, 384],
        in_channels=1,
        out_channels=14,
        feature_size=48,
        spatial_dims=2,
        use_checkpoint=True,
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
