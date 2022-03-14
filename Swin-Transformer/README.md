# Swin Transformer

This folder contains the implementation of the ACmix based on Swin Transformer models for image classification.

### Requirements

+ Python 3.7

+ PyTorch==1.8.0

+ torchvision==0.9.0

+ timm==0.3.2

+ opencv-python==4.4.0.46

+ termcolor==1.1.0

+ yacs==0.1.8

+ Install Apex:

  ```python
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. The file structure should look like:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

### Run

Train Swin-T + ACmix on ImageNet

```python
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/acmix_swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128
```

Train Swin-S + ACmix on ImageNet

```python
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/acmix_swin_small_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128
```

