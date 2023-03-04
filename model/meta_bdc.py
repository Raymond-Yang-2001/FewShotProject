import torch
import torch.nn as nn

from representation import MPNCOV
from representation.bdc_module import BDC
from .resnet import ResNet12


def load_model(model, dir):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['model']
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model


class META_BDC(nn.Module):
    def __init__(self, model_func, reduced_dim=256):
        super(META_BDC, self).__init__()
        self.model_func = model_func
        self.feat_dim = self.model_func.feat_dim
        self.reduced_dim = reduced_dim
        self.dcov = BDC(is_vec=True, input_dim=self.feat_dim, dimension_reduction=self.reduced_dim)

    def forward(self, x):
        x = self.model_func(x)
        x = self.dcov(x)
        x = x.view(x.size(0), -1)
        return x


def metabdcresnet12(model_path=None, pretrain=False):
    model_func = ResNet12()
    if pretrain:
        model_func = load_model(model_func, model_path)
    model = META_BDC(model_func=model_func)
    return model
