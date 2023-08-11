import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import IntegerSortDataset, sparse_seq_collate_fn
from model import PointerNet

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def masked_accuracy(output, target, mask):
    """Computes a batch accuracy with a mask (for padded sequences) """
    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean()

        return accuracy


class mainArguments:
    def __init__(self, **kwargs):
        args={
            'low':0,
            'high':100,
            'min_length':5,
            'max_length':10,
            'train_samples':100000,
            'test_samples':1000,
            'emb_dim':8,
            'batch_size':256,
            'epochs':100,
            'lr':5e-3,
            'wd':1e-5,
            'workers':4,
            'no_cuda':False,
            'seed':None
        }
        assert kwargs.keys() <= args.keys(), f"kwargs contains the following invalid keys: {set(kwargs.keys()).difference(args.keys())}"
        args.update(kwargs) # Update default opts with user requests
        
        for key,val in args.items():
            setattr(self, key, val)
    
        
def main(**kwargs):
    args = mainArguments(**kwargs)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    cudnn.benchmark = True if use_cuda else False

    train_set = IntegerSortDataset(num_samples=args.train_samples, high=args.high, min_len=args.min_length, max_len=args.max_length, seed=1)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=sparse_seq_collate_fn)

    test_set = IntegerSortDataset(num_samples=args.test_samples, high=args.high, min_len=args.min_length, max_len=args.max_length, seed=2)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=sparse_seq_collate_fn)

    model = PointerNet(input_dim=args.high, embedding_dim=args.emb_dim, hidden_size=args.emb_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()

    for epoch in range(args.epochs):
        # Train
        model.train()
        for batch_idx, (seq, length, target) in enumerate(train_loader):
            seq, length, target = seq.to(device), length.to(device), target.to(device)

            optimizer.zero_grad()
            log_pointer_score, argmax_pointer, mask = model(seq, length)

            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            train_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())

            # if batch_idx % 20 == 0:
            #     print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
            #           .format(epoch, batch_idx * len(seq), len(train_loader.dataset),
            #                   100. * batch_idx / len(train_loader), train_loss.avg, train_accuracy.avg))

        # Test
        model.eval()
        for seq, length, target in test_loader:
            seq, length, target = seq.to(device), length.to(device), target.to(device)

            log_pointer_score, argmax_pointer, mask = model(seq, length)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            test_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            test_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())
        print('Epoch {}: Test\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, test_loss.avg, test_accuracy.avg))
