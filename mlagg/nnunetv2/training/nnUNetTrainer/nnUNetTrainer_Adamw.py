import torch
from torch.optim import Adam, AdamW

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import timm
import timm.optim
import timm.scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


class nnUNetTrainer_Adamw(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        ### Some hyperparameters for you to fiddle with
        # self.initial_lr = 1e-4  # 1e-4
        # self.weight_decay = 1e-2  # 1e-3
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 500  # 1000
        self.current_epoch = 0

    def configure_optimizers(self):
        # optimizer = AdamW(self.network.parameters(),
        #                   lr=self.initial_lr,
        #                   weight_decay=self.weight_decay,
        #                   amsgrad=True)
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer, t_initial=self.num_epochs, lr_min=1e-6,
                                                        warmup_t=10, warmup_lr_init=1e-4)
        return optimizer, lr_scheduler

