#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=150, help="communication rounds")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")


    # model arguments
    parser.add_argument('--model', type=str, default='vit', help='vit, bert, mvp')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # dataset
    parser.add_argument('--dataset', type=str, default='flower', help="cub, flower")
    parser.add_argument('--pretrain', action='store_false', help='pretrain or not') # store_true : False, store_false:True
    parser.add_argument('--num_classes', type=int, default=102, help="cub: 200, flower: 102")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
 

    #other
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=5481, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_false', help='aggregation over all clients')
    parser.add_argument('--comment', type=str, default='CONFID', help="comment")
    args = parser.parse_args()
    return args
