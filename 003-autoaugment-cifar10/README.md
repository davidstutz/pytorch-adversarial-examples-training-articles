# 2.56% on Cifar10 with AutoAugment

Training with AutoAugment and CutOut using ResNets, Wide ResNets or SimpleNet:

    cd examples/
    python3 train.py --architecture=wrn2810|resnet50|simplenet --directory=output_directory

| Architecture | Test Error |
|---|---|
| WRN-28-10 | 2.56% |
| ResNet-50 | 3.13% |
| SimpleNet | 3.85% |

Optionally, different normalization schemes such as group normalization can
be used `examples/train.py`.

Pre-trained models from the article can be downloaded [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/PrsSqnXHD2RyMfG).