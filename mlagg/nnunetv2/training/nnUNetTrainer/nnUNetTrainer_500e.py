from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

import torch
from torch import nn
import numpy as np
import timm
import timm.optim
import timm.scheduler


class nnUNetTrainer_500e(nnUNetTrainer):
    ''' With no DeepSupervision
    Outputs of decoder network is a torch.Tensor.
    If with DeepSupervision, this is a list of torch.Tensor. But haven't been deployed in this file.
    '''
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        ### Some hyperparameters for you to fiddle with
        self.num_epochs = 500  # 1000
