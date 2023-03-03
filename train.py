import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from params import get_params
from Dataset import ImageNetFolder, PrototypicalBatchSampler, get_transformers
from ProtoNet import ProtoNet
from loss import PrototypicalLoss
from torch.utils.tensorboard import SummaryWriter

from model.mpncovresnet import mpncovresnet50

args = get_params(True, './train_base.json')

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(args.gpu)

print()
print(args)


def train(tr_dataloader, model, optim, lr_scheduler, checkpoint_dir, val_dataloader=None):
    """
    Train the model with the prototypical learning algorithm
    """

    start_epoch = 0

    best_acc = 0

    best_model_path = os.path.join('./runs/' + checkpoint_dir, 'best_model.pth')
    checkpoint_path = os.path.join('./runs/' + checkpoint_dir, 'checkpoint.pth')

    loss_fn = PrototypicalLoss(args.n_way, args.n_support, args.n_query)
    if args.resume:
        checkpoint = torch.load(checkpoint_path)
        assert ValueError("No checkpoint found!")
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    for epoch in range(start_epoch, args.epochs):
        episode_tr_loss = []
        episode_tr_acc = []
        episode_val_loss = []
        episode_val_acc = []

        print('=== Epoch: {} ==='.format(epoch))
        model.train()
        train_iter = iter(tr_dataloader)
        for batch in train_iter:
            optim.zero_grad()
            x, y = batch
            # print(x.shape)
            # x, y = x.to(device), y.to(device)
            x = x.to(device)
            model_output = model(x)
            # print(y)
            loss, acc = loss_fn(model_output)
            loss.backward()
            optim.step()
            episode_tr_loss.append(loss.item())
            episode_tr_acc.append(acc)

        avg_loss = np.mean(episode_tr_loss)
        avg_acc = np.mean(episode_tr_acc)

        print('Avg Train Loss: {:.4f}, Avg Train Acc: {:.4f}%'.format(avg_loss, avg_acc))
        logger.add_scalar("Loss/Train", avg_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Train", avg_acc, global_step=epoch)

        lr_scheduler.step()

        state = {'model': model.state_dict(),
                 'optimizer': optim.state_dict(),
                 'epoch': epoch + 1,
                 'best_acc': best_acc,
                 'lr_scheduler': lr_scheduler.state_dict()}

        torch.save(state, checkpoint_path)
        print("Checkpoint Saved!\n")

        if val_dataloader is None:
            continue

        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x = x.to(device)
            # x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output)
            episode_val_loss.append(loss.item())
            episode_val_acc.append(acc)
        avg_loss = np.mean(episode_val_loss)
        avg_acc = np.mean(episode_val_acc)
        std_acc = np.std(episode_val_acc)
        logger.add_scalar("Loss/Validation", avg_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Validation", avg_acc, global_step=epoch)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {:.2f}%)'.format(
            best_acc * 100)
        print('Avg Val Loss: {:.4f}, Avg Val Acc: {:.2f}%+-{:.2f}%\t{}'.format(
            avg_loss, avg_acc, 1.96 * std_acc / np.sqrt(args.episodes), postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc

    # torch.save(model.state_dict(), last_model_path)


def init_seed():
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)


def init_dataset():
    tr_trans = get_transformers('train')
    val_trans = get_transformers('val')
    train_dataset = ImageNetFolder(root=args.train_root, dataset_name=args.dataset, phase='train',
                                   transformer=tr_trans)
    val_dataset = ImageNetFolder(root=args.val_root, dataset_name=args.dataset, phase='val',
                                 transformer=val_trans)
    n_classes = len(np.unique(train_dataset.targets))
    if n_classes < args.n_way:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen n_way. Decrease the ' +
                         'n_way option and try again.'))
    return train_dataset, val_dataset


def init_sampler(labels, mode):
    classes_per_it = args.n_way
    # 同时产生 support和query
    num_samples = args.n_support + args.n_query
    if 'train' == mode:
        sampler = PrototypicalBatchSampler(labels=labels,
                                           classes_per_episode=classes_per_it,
                                           sample_per_class=num_samples,
                                           iterations=args.episodes,
                                           dataset_name='miniImageNet_train' if args.dataset == "miniImageNet" else
                                           'tieredImageNet_train')
    else:
        sampler = PrototypicalBatchSampler(labels=labels,
                                           classes_per_episode=classes_per_it,
                                           sample_per_class=num_samples,
                                           iterations=args.episodes,
                                           dataset_name='miniImageNet_val' if args.dataset == "miniImageNet" else
                                           'tieredImageNet_val')
    return sampler


def init_dataloader():
    train_dataset, val_dataset = init_dataset()
    tr_sampler = init_sampler(train_dataset.targets, 'train')
    val_sampler = init_sampler(val_dataset.targets, 'val')
    tr_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=tr_sampler)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)
    return tr_dataloader, val_dataloader


def init_protonet():
    """
    Initialize the ProtoNet
    """
    # model = ProtoNet().to(device)
    model = mpncovresnet50()

    class Identity(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    model.fc = Identity()
    model.to(device)
    return model


def init_optim(model):
    """
    Initialize optimizer
    """
    if args.optimizer == "Adam":
        return torch.optim.Adam(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.optimizer == "SGD":
        return torch.optim.SGD(params=model.parameters(),
                               nesterov=args.nesterov,
                               lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)


def init_lr_scheduler(optim):
    """
    Initialize the learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=args.lrG,
                                           step_size=args.lrS)


tr_dataloader, val_dataloader = init_dataloader()

model = init_protonet()
optim = init_optim(model)
lr_scheduler = init_lr_scheduler(optim)

if __name__ == "__main__":
    checkpoint_dir = 'runs/%s/%s_%s' % (args.dataset, args.model, args.method)
    checkpoint_dir += '_%dway_%dshot' % (args.n_way, args.n_support)
    checkpoint_dir += '_metatrain'
    checkpoint_dir += '_'
    checkpoint_dir += args.exp
    if not os.path.exists(os.path.join(os.getcwd(), checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), checkpoint_dir))
    print(checkpoint_dir)

    with open(checkpoint_dir + '/params.json', mode="wt") as f:
        json.dump(vars(args), f, indent=4)

    logger = SummaryWriter(log_dir='./runs/' + checkpoint_dir)

    train(tr_dataloader=tr_dataloader, val_dataloader=val_dataloader, model=model, lr_scheduler=lr_scheduler,
          optim=optim, checkpoint_dir=checkpoint_dir)