# MLAgg-UNet: Advancing Medical Image Segmentation with Efficient Transformer and Mamba-Inspired Multi-Scale Sequence


## Requirements: 

Please use `Ubuntu 20.04` for environment setting.

1.python 3.10 + [torch](https://pytorch.org/get-started/locally/) 2.0.0 + torchvision 0.15.0 (cuda 11.7)
```bash
conda create -n mlaggunet python=3.10
conda activate mlaggunet
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

2.Clone this repository
```bash
git clone https://github.com/aticejiang/MLAgg-UNet
cd MLAgg-UNet
pip install -e .
```

3.Install [monai](https://github.com/Project-MONAI/MONAI): 
```bash
pip install monai
``` 

4.Install [Mamba](https://github.com/state-spaces/mamba) : 
```bash
pip install causal-conv1d
pip install mamba-ssm
```

## Dataset preparation
Reference: [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
### Data download
- [AbdomenMRI(U-Mamba)](https://arxiv.org/abs/2401.04722)
- [BTCV](https://www.synapse.org/Synapse:syn3193805/wiki/89480)
- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [Endovis17]()

### Dataset conversion
Conversion via specific programs:
e.g. ```python Dataset027_ACDC.py``` for ACDC dataset

or download from ours [Baidu]()

### Preprocessing
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Train models

```bash
nnUNetv2_train DATASET_ID 2d 0 -tr 
```
or using custom bath size
```bash
nnUNetv2_train DATASET_ID 2d_bsXX 0 -tr
```

## Acknowledgements

Thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba) and [U-mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.

## Citation
