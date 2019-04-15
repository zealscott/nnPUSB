import argparse
import numpy as np
import torch


# using GPU if available
# device = torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_args():
    parser = argparse.ArgumentParser(
        description='non-negative / unbiased PU learning Chainer implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=30000,
                        help='Mini batch size')
    parser.add_argument('--dataset', '-d', default='mnist', type=str, choices=['mnist', 'cifar10'],
                        help='The dataset name')
    parser.add_argument('--labeled', '-l', default=100, type=int,
                        help='# of labeled data')
    parser.add_argument('--unlabeled', '-u', default=59900, type=int,
                        help='# of unlabeled data')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPUSB')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPUSB')
    parser.add_argument('--loss', type=str, default="sigmoid", choices=['sigmoid'],
                        help='The name of a loss function')
    parser.add_argument('--model', '-m', default='3mlp', choices=['3mlp','6mlp','cnn'],
                        help='The name of a classification model')
    parser.add_argument('--stepsize', '-s', default=1e-3, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--preset', '-p', type=str, default='mnist',
                        choices=['mnist','mnist-6mlp','cifar10'],
                        help="Preset of configuration\n" +
                             "mnist: The setting of MNIST experiment in Experiment")
    args = parser.parse_args()
    if args.preset == "mnist":
        args.labeled = 100
        args.unlabeled = 59900
        args.dataset = "mnist"
        args.batchsize = 30000
        args.epoch = 100
        args.model = "3mlp"
    elif args.preset == "mnist-6mlp":
        args.labeled = 1000
        args.unlabeled = 60000
        args.dataset = "mnist"
        args.batchsize = 30000
        args.epoch = 200
        args.model = "6mlp"
    elif args.preset == "cifar10":
        args.labeled = 1000
        args.unlabeled = 50000
        args.dataset = "cifar10"
        args.batchsize = 500
        args.model = "cnn"
        args.stepsize = 1e-5
        args.epoch = 100

    assert (args.batchsize > 0)
    assert (args.epoch > 0)
    assert (0 < args.labeled < 30000)
    if args.dataset == "mnist":
        assert (0 < args.unlabeled <= 60000)
    else:
        assert (0 < args.unlabeled <= 50000)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
    return args