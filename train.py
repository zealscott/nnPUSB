import copy
import torch
import numpy as np


from model import ThreeLayerPerceptron, MultiLayerPerceptron, CNN
from nnPU_loss import nnPUloss, nnPUSBloss
from dataset import load_dataset, to_dataloader
from args import process_args, device
from plot import draw_losses_test_data, draw_precision_recall


def select_loss(loss_name):
    losses = {
        "sigmoid": lambda x: torch.sigmoid(-x)}
    return losses[loss_name]


def select_model(model_name):
    models = {"3mlp": ThreeLayerPerceptron,
              "6mlp": MultiLayerPerceptron, "cnn": CNN}
    return models[model_name]


def make_optimizer(model, stepsize):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=stepsize, weight_decay=0.005)
    return optimizer


class trainer():
    def __init__(self, models, loss_funcs, optimizers, XYtrainLoader, XYvalidLoader, XYtestLoader, prior):
        self.models = models
        self.loss_funcs = loss_funcs.values()
        self.optimizers = optimizers.values()
        self.XYtrainLoader = XYtrainLoader
        self.XYvalidLoader = XYvalidLoader
        self.XYtestLoader = XYtestLoader
        self.prior = prior

    def train(self):
        nnPUSB_result = []
        nnPU_result = []

        for net, opt, loss_func, model_name in zip(self.models.values(), self.optimizers, self.loss_funcs, self.models.keys()):
            net.train()
            for data, target in self.XYtrainLoader:          # for each training step
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                opt.zero_grad()                # clear gradients for next train
                output = net(data)              # get output for every net
                loss = loss_func(output, target)  # compute loss for every net
                # print("model：{0},loss:{1}".format(model_name, loss.item()))
                loss.backward()                # backpropagation, compute gradients
                opt.step()                     # apply gradients
            if model_name == "nnPUSB":
                nnPUSB_result.extend(net.evalution_with_density(
                    self.XYtestLoader, self.prior))
            else:
                nnPU_result.extend(net.error(self.XYtestLoader))

        return nnPU_result, nnPUSB_result

    def run(self, Epochs):
        # [precision,recall,error_rate]
        nnPU_result = [[], [], []]
        nnPUSB_result = [[], [], []]
        print("Epoch\tnnPU/precision\tnnPU/recall\tnnPU/error\tnnPUSB/precision\tnnPUSB/recall\tnnPUSB/error")

        for epoch in range(Epochs):
            _nnPU, _nnPUSB = self.train()
            print("{0}\t{1:-8}\t{2:-8}\t{3:-8}\t{4:-8}\t{5:-8}\t{6:-8}".format(
                epoch, round(_nnPU[0], 4), round(_nnPU[1], 4), round(_nnPU[2], 4), round(_nnPUSB[0], 4), round(_nnPUSB[1], 4), round(_nnPUSB[2], 4)))

            nnPU_result[0].append(_nnPU[0])
            nnPU_result[1].append(_nnPU[1])
            nnPU_result[2].append(_nnPU[2])

            nnPUSB_result[0].append(_nnPUSB[0])
            nnPUSB_result[1].append(_nnPUSB[1])
            nnPUSB_result[2].append(_nnPUSB[2])

        draw_losses_test_data(nnPU_result[2], nnPUSB_result[2])
        draw_precision_recall(
            nnPU_result[0], nnPU_result[1], nnPUSB_result[0], nnPUSB_result[1])


def main():
    args = process_args()
    print("using:",device)

    # dataset setup
    XYtrainLoader, XYvalidLoader, XYtestLoader, prior, dim = load_dataset(
        args.dataset, args.labeled, args.unlabeled, args.batchsize, with_bias=True, resample_model=args.model)

    # model setup
    loss_type = select_loss(args.loss)
    selected_model = select_model(args.model)
    model = selected_model(dim)
    models = {"nnPU": copy.deepcopy(model).to(
        device), "nnPUSB": copy.deepcopy(model).to(device)}
    loss_funcs = {"nnPU": nnPUloss(prior, loss=loss_type, gamma=args.gamma, beta=args.beta),
                  "nnPUSB": nnPUSBloss(prior, gamma=args.gamma, beta=args.beta)}

    # trainer setup
    optimizers = {k: make_optimizer(v, args.stepsize)
                  for k, v in models.items()}
    print("input dim: {}".format(dim))
    print("prior: {}".format(prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
    print("model: {}".format(selected_model))
    print("beta: {}".format(args.beta))
    print("gamma: {}".format(args.gamma))
    print("")

    # run training
    # training
    PUtrainer = trainer(models, loss_funcs, optimizers,
                        XYtrainLoader, XYvalidLoader, XYtestLoader, prior)
    PUtrainer.run(args.epoch)


if __name__ == '__main__':
    import os
    import sys
    os.chdir(sys.path[0])
    print("working dir: {}".format(os.getcwd()))
    main()
