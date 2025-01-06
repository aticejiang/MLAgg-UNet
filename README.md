# MLAgg-UNet: Advancing Medical Image Segmentation with Efficient Transformer and Mamba-Inspired Multi-Scale Sequence


## Requirements: 

1.python 3.10 + [torch](https://pytorch.org/get-started/locally/) 2.0.0 + torchvision 0.15.0 (cuda 11.7)
```
conda create -n mlaggunet python=3.10
conda activate mlaggunet
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

2.Clone this repository
```
git clone https://github.com/aticejiang/MLAgg-UNet
cd MLAgg-UNet
pip install -e .
```

3.Install [monai](https://github.com/Project-MONAI/MONAI): 
```
pip install monai
``` 

4.Install [Mamba](https://github.com/state-spaces/mamba) : 
```
pip install causal-conv1d
pip install mamba-ssm
```
