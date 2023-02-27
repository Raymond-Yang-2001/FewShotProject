import argparse
import os
import torch

import numpy as np
from tqdm import tqdm

from Dataset import ImageNetFolder, PrototypicalBatchSampler, get_transformers
from ProtoNet import ProtoNet
from loss import prototypical_loss as loss_fn
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', '--dataset_name',
                    type=str,
                    help='dataset name',
                    default='miniImageNet',
                    choices=['miniImageNet', 'tieredImageNet'])
parser.add_argument('-train_root', '--train_dataset_root',
                    type=str,
                    help='path to dataset',
                    default='F:/miniImageNet/train')
parser.add_argument('-val_root', '--val_dataset_root',
                    type=str,
                    help='path to dataset',
                    default='F:/miniImageNet/val')
parser.add_argument('-exp', '--experiment_root',
                    type=str,
                    help='root where to store models, losses and accuracies',
                    default='exp1')

parser.add_argument('-nep', '--epochs',
                    type=int,
                    help='number of epochs to train for',
                    default=100)

parser.add_argument('-lr', '--learning_rate',
                    type=float,
                    help='learning rate for the model, default=0.001',
                    default=0.001)

parser.add_argument('-lrS', '--lr_scheduler_step',
                    type=int,
                    help='StepLR learning rate scheduler step, default=20',
                    default=20)

parser.add_argument('-lrG', '--lr_scheduler_gamma',
                    type=float,
                    help='StepLR learning rate scheduler gamma, default=0.5',
                    default=0.5)

parser.add_argument('-its', '--iterations',
                    type=int,
                    help='number of episodes per epoch, default=100',
                    default=100)

parser.add_argument('-cTr', '--classes_per_it_tr',
                    type=int,
                    help='number of random classes per episode for training, default=60',
                    default=5)

parser.add_argument('-nsTr', '--num_support_tr',
                    type=int,
                    help='number of samples per class to use as support for training, default=5',
                    default=5)

parser.add_argument('-nqTr', '--num_query_tr',
                    type=int,
                    help='number of samples per class to use as query for training, default=5',
                    default=15)

parser.add_argument('-cVa', '--classes_per_it_val',
                    type=int,
                    help='number of random classes per episode for validation, default=5',
                    default=5)

parser.add_argument('-nsVa', '--num_support_val',
                    type=int,
                    help='number of samples per class to use as support for validation, default=5',
                    default=5)

parser.add_argument('-nqVa', '--num_query_val',
                    type=int,
                    help='number of samples per class to use as query for validation, default=15',
                    default=15)

parser.add_argument('-seed', '--manual_seed',
                    type=int,
                    help='input for the manual seeds initializations',
                    default=7)

parser.add_argument('--gpu', type=int,
                    default=0,
                    help='enables cuda')
parser.set_defaults(augment=True)

args = parser.parse_args()
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(args.gpu)

print()
print(args)
logger = SummaryWriter(log_dir='./runs/' + args.experiment_root)


def train(tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    """
    Train the model with the prototypical learning algorithm
    """

    if val_dataloader is None:
        best_state = None

    best_acc = 0

    best_model_path = os.path.join('./runs/' + args.experiment_root, 'best_model.pth')
    last_model_path = os.path.join('./runs/' + args.experiment_root, 'last_model.pth')

    for epoch in range(args.epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            # print(x.shape)
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=args.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-args.iterations:])
        avg_acc = np.mean(train_acc[-args.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        logger.add_scalar("Loss/Train", avg_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Train", avg_acc, global_step=epoch)

        lr_scheduler.step()

        if val_dataloader is None:
            continue

        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=args.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-args.iterations:])
        avg_acc = np.mean(val_acc[-args.iterations:])
        logger.add_scalar("Loss/Validation", avg_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Validation", avg_acc, global_step=epoch)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)


def init_seed():
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)


def init_dataset():
    tr_trans = get_transformers('train')
    val_trans = get_transformers('val')
    train_dataset = ImageNetFolder(root=args.train_dataset_root, dataset_name=args.dataset_name, phase='train',
                                   transformer=tr_trans)
    val_dataset = ImageNetFolder(root=args.val_dataset_root, dataset_name=args.dataset_name, phase='val',
                                 transformer=val_trans)
    n_classes = len(np.unique(train_dataset.targets))
    if n_classes < args.classes_per_it_tr or n_classes < args.classes_per_it_val:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    return train_dataset, val_dataset


def init_sampler(labels, mode):
    if 'train' == mode:
        classes_per_it = args.classes_per_it_tr
        # 同时产生 support和query
        num_samples = args.num_support_tr + args.num_query_tr

        sampler = PrototypicalBatchSampler(labels=labels,
                                           classes_per_episode=classes_per_it,
                                           sample_per_class=num_samples,
                                           iterations=args.iterations,
                                           dataset_name='miniImageNet_train' if args.dataset_name == "miniImageNet" else
                                           'tieredImageNet_train')
    else:
        classes_per_it = args.classes_per_it_val
        num_samples = args.num_support_val + args.num_query_val

        sampler = PrototypicalBatchSampler(labels=labels,
                                           classes_per_episode=classes_per_it,
                                           sample_per_class=num_samples,
                                           iterations=args.iterations,
                                           dataset_name='miniImageNet_val' if args.dataset_name == "miniImageNet" else
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
    '''
    Initialize the ProtoNet
    '''
    model = ProtoNet().to(device)
    return model


def init_optim(model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=args.learning_rate)


def init_lr_scheduler(optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=args.lr_scheduler_gamma,
                                           step_size=args.lr_scheduler_step)


tr_dataloader, val_dataloader = init_dataloader()

model = init_protonet()
optim = init_optim(model)
lr_scheduler = init_lr_scheduler(optim)

if __name__ == "__main__":
    train(tr_dataloader=tr_dataloader, val_dataloader=val_dataloader, model=model, lr_scheduler=lr_scheduler,
          optim=optim)
