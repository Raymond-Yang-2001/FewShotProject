import os

from model import resnet, meta_bdc
from model import mpncovresnet
import torch

model_dict = dict(
    # ResNet10=resnet.resnet10,
    ResNet12=resnet.ResNet12,
    ResNet18=resnet.resnet18,
    ResNet34=resnet.resnet34,
    # ResNet34s=resnet.resNet34s,
    ResNet50=resnet.resnet50,
    ResNet101=resnet.resnet101,
    ResNet152=resnet.resnet152,
    MPNCOVResNet12=mpncovresnet.mpncovresnet12,
    MPNCOVResNet50=mpncovresnet.mpncovresnet50,
    BDCResNet12=meta_bdc.metabdcresnet12,
    BDCResNet18=meta_bdc.metabdcresnet18)


def load_model(model, dir):
    model_dict = model.state_dict()
    if os.path.splitext(dir)[-1] == ".tar":
        file_dict = torch.load(dir)['state']
        print("Get tar")
    if os.path.splitext(dir)[-1] == ".pth":
        file_dict = torch.load(dir)
    for k, v in file_dict.items():
        if k in model_dict:
            print(k)
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model
