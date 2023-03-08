import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .resnet import Bottleneck, ResNet12, resnet50
from representation import MPNCOV

model_urls = {
    'mpncovresnet50': 'http://jtxie.com/models/mpncovresnet50-15991845.pth',
    'mpncovresnet101': 'http://jtxie.com/models/mpncovresnet101-ade9737a.pth'
}


class _MPNCOVResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(_MPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer_reduce = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(int(256 * (256 + 1) / 2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class covpool(nn.Module):
    def __init__(self):
        super(covpool, self).__init__()

    def forward(self, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1. / M / M) * torch.ones(M, M, device=x.device) + (1. / M) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        return y


class isqrtm(nn.Module):
    def __init__(self, n):
        super(isqrtm, self).__init__()
        self.iterN = n

    def forward(self, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x)).half()
        # Y = torch.zeros(batchSize, self.iterN, dim, dim, requires_grad=False, device=x.device).half().type(dtype)
        # Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, self.iterN, 1, 1).half()#type(dtype)#
        if self.iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y = A.bmm(ZY)
            Z = ZY
            for i in range(1, self.iterN - 1):
                ZY = 0.5 * (I3 - Z.bmm(Y))
                Y = Y.bmm(ZY)
                Z = ZY.bmm(Z)
            YZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x).half()
        return y


class tritovec(nn.Module):
    def __init__(self):
        super(tritovec, self).__init__()

    def forward(self, input):
        """
        reshape the upper triangular of matrix to a vector
        :param ctx:
        :param input:
        :return:
        """
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero()
        y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(dtype)
        y = x[:, index]
        return y


class MPNCOVResNet(nn.Module):
    def __init__(self, model_func, reduced_dim=256):
        super(MPNCOVResNet, self).__init__()
        self.model_func = model_func
        self.feat_dim = self.model_func.feat_dim
        self.reduced_dim = reduced_dim
        self.layer_reduce = nn.Conv2d(self.feat_dim[0], reduced_dim, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(reduced_dim)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        self.meta_layer1 = covpool()
        self.meta_layer2 = isqrtm(n=5)
        self.meta_layer3 = tritovec()

    def forward(self, x):
        x = self.model_func(x)
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)
        '''x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)'''
        x = self.meta_layer1(x)
        x = self.meta_layer2(x)
        x = self.meta_layer3(x)
        x = x.view(x.size(0), -1)
        return x


def mpncovresnet12(reduce_dim):
    model_func = ResNet12()

    model = MPNCOVResNet(model_func=model_func, reduced_dim=reduce_dim)
    return model


def mpncovresnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model_func = resnet50()
    model = _MPNCOVResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def mpncovresnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MPNCOVResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['mpncovresnet101']))
    return model
