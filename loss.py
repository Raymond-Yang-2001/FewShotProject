# coding=utf-8
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np


class PrototypicalLoss(Module):
    """
    Loss class deriving from Module for the prototypical loss function defined below
    """

    def __init__(self, n_way, n_support, n_query):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support
        self.n_way = n_way
        self.n_query = n_query
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input):
        return prototypical_loss(input, self.n_way, self.n_support, self.n_query, self.loss_fn)


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, n_way, n_support, n_query, loss_fn):
    """
    """
    z = input.view(n_way, n_support + n_query, -1)
    z_support = z[:, :n_support]
    z_query = z[:, n_support:]

    # contiguous()函数的作用：把tensor变成在内存中连续分布的形式。
    prototypes = z_support.contiguous().view(n_way, n_support, -1).mean(1)
    query_samples = z_query.contiguous().view(n_way * n_query, -1)
    '''def supp_idxs(c):
        return torch.argwhere(target_cpu.eq(c))[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    # 得到类别索引和类的数量
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # assuming n_query, n_target constants
    # 得到每一类的query的数量
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    # 得到所有support在input中的索引
    support_idxs = list(map(supp_idxs, classes))
    # 每一类的support计算均值，结果stack起来  (n_class,d)
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    # 最后应该为squeeze,或者是view(n_class,-1)

    def quer_idxs(c):
        return torch.argwhere(target_cpu.eq(c))[n_support:]

    query_idxs = torch.stack(list(map(quer_idxs, classes))).view(-1)
    # query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    # query_sample (n_class*n_query,d)
    query_samples = input.to('cpu')[query_idxs]
   
    '''
    '''
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    # gather, 在维度2上按照target_inds索引取元素，得到(n_class,n_query,1)，代表每个类里面 对应的每个query样本和原型的损失，求均值
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()'''
    dists = euclidean_dist(query_samples, prototypes)
    # dist (n_query, n_way)
    y_label = np.repeat(range(n_way), n_query)
    # y_label (n_way*n_query, ) e,g,(0,0,0,0,0,1,1,1,1,1......)
    y_query = torch.from_numpy(y_label).long().cuda()
    y_query = Variable(y_query)
    loss_val = loss_fn(-dists, y_query)

    #  k, dim=None, largest=True, sorted=True    return values， indices
    # 找到每一个query样本距离最近的（最接近）的prototype
    topk_scores, topk_labels = (-dists).data.topk(1, 1, True, True)
    # topk_labels (n_query, 1)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_label)

    # 在维度2上取最大(索引)，得到(n_class,n_query,1)
    # _, y_hat = log_p_y.max(2)
    # acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    acc_val = top1_correct / (topk_ind.shape[0]) * 100

    return loss_val, acc_val
