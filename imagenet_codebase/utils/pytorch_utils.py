# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import time

import torch
import torch.nn as nn
import copy

from imagenet_codebase.utils.flops_counter import profile


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return cross_entropy_loss_with_soft_target(pred, soft_target)


def clean_num_batch_tracked(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()
                

def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x
            

""" Net Info """


def get_net_device(net):
    return net.parameters().__next__().device


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


def count_net_flops(net, data_shape=(1, 3, 224, 224)):
    if isinstance(net, nn.DataParallel):
        net = net.module

    net = copy.deepcopy(net)
    
    flop, _ = profile(net, data_shape)
    return flop
    

def get_net_info(net, input_shape=(3, 224, 224), print_info=True):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module
    
    # parameters
    net_info['params'] = count_parameters(net)
    
    # flops
    net_info['flops'] = count_net_flops(net, [1] + list(input_shape))
    
    if print_info:
        print(net)
        print('Total training params: %.2fM' % (net_info['params'] / 1e6))
        print('Total FLOPs: %.2fM' % (net_info['flops'] / 1e6))
    
    return net_info
