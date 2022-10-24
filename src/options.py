#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # pretrained stage parameters
    parser.add_argument('--pretrained', type=int, default=1, help="whether pretrained or not")
    parser.add_argument('--hidden_channel', type=int, default=128,
                        help='hidden channel for DNN')
    parser.add_argument('--pretrained_epochs', type=int, default=50,
                        help="number of rounds of training")
    parser.add_argument('--pretrained_lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--pretrained_bs', type=int, default=10, help="batch size of pretraining")
    parser.add_argument('--pretrained_momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--pretrained_optimizer', type=str, default='sgd', help="type \
                    of optimizer")

    # model arguments
    parser.add_argument('--model', type=str, default='CNN2', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    

    # other arguments
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU id,-1 for CPU")
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")

    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')

    # Adversarial Distillation 
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--iter_discrim', type=int, default=5, help='number of iteration for Imitation Stage')
    parser.add_argument('--iter_gen', type=int, default=1, help='number of iteration for Generation  Stage')
    parser.add_argument('--comm_rounds', type=int, default=10,  #   100
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default=1,    # 100
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=256,
                        help="local batch size: B")
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=33, metavar='S',
                        help='random seed (default: 33)')
    parser.add_argument('--scheduler', action='store_true', default=False)
    args = parser.parse_args()
    return args
