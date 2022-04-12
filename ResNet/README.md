# ResNet

This folder contains the implementation of the ACmix based on ResNet models for image classification.

### Requirements

+ Python 3.7
+ PyTorch==1.8.0
+ torchvision==0.9.0

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

Train ResNet + ACmix on ImageNet

```python
python main.py  --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --world-size 1 --rank 0 --data_url <imagenet-path> --batch-size 128
```
