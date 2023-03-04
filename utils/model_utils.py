from model import resnet, meta_bdc
from model import mpncovresnet
import torch

model_dict = dict(
    # ResNet10=resnet.resnet10,
    # ResNet12=resnet.pretrained_resnet12(),
    ResNet18=resnet.resnet18,
    ResNet34=resnet.resnet34,
    # ResNet34s=resnet.resNet34s,
    ResNet50=resnet.resnet50,
    ResNet101=resnet.resnet101,
    ResNet152=resnet.resnet152,
    MPNCOVResNet50=mpncovresnet.mpncovresnet50,
    BDCResNet12=meta_bdc.metabdcresnet12)



