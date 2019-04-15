import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from args import device


class MyClassifier(nn.Module):

    def zero_one_loss(self, h, t, is_logistic=False):
        self.eval()
        positive = 1
        negative = 0 if is_logistic else -1

        n_p = (t == positive).sum()
        n_n = (t == negative).sum()
        size = n_p + n_n

        t_p = ((h == positive) * (t == positive)).sum()
        t_n = ((h == negative) * (t == negative)).sum()
        f_p = n_n - t_n
        f_n = n_p - t_p

        # print("size:{0},t_p:{1},t_n:{2},f_p:{3},f_n:{4}".format(
        #     size, t_p, t_n, f_p, f_n))

        presicion = (0.0 if t_p == 0 else t_p/(t_p+f_p))
        recall = (0.0 if t_p == 0 else t_p/(t_p+f_n))

        return presicion, recall, 1 - (t_p+t_n)/size

    def error(self, DataLoader, is_logistic=False):
        presicion = []
        recall = []
        error_rate = []
        self.eval()
        for data, target in DataLoader:
            data = data.to(device, non_blocking=True)
            t = target.detach().cpu().numpy()
            size = len(t)
            if is_logistic:
                h = np.reshape(torch.sigmoid(
                    self(data)).detach().cpu().numpy(), size)
                h = np.where(h > 0.5, 1, 0).astype(np.int32)
            else:
                h = np.reshape(torch.sign(
                    self(data)).detach().cpu().numpy(), size)

            result = self.zero_one_loss(h, t, is_logistic)
            presicion.append(result[0])
            recall.append(result[1])
            error_rate.append(result[2])

        return sum(presicion)/len(presicion), sum(recall)/len(recall), sum(error_rate)/len(error_rate)

    def evalution_with_density(self, DataLoader, prior):
        presicion = []
        recall = []
        error_rate = []
        self.eval()
        for data, target in DataLoader:
            data = data.to(device)
            t = target.detach().cpu().numpy()
            size = len(t)
            # get f_x
            h = np.reshape(self(data).detach().cpu().numpy(), size)
            # predict with density ratio and threshold
            h = self.predict_with_density_threshold(h, target, prior)
            # evalution
            result = self.zero_one_loss(h, t)

            presicion.append(result[0])
            recall.append(result[1])
            error_rate.append(result[2])

        return sum(presicion)/len(presicion), sum(recall)/len(recall), sum(error_rate)/len(error_rate)

    def predict_with_density_threshold(self, f_x, target, prior):
        density_ratio = f_x/prior
        # ascending sort
        sorted_density_ratio = np.sort(density_ratio)
        size = len(density_ratio)

        n_pi = int(size * prior)
        # print("size: ", size)
        # print("density_ratio shape: ", density_ratio.shape)
        # print("n in test data: ", n_pi)
        # print("n in real data: ", (target == 1).sum())
        threshold = (
            sorted_density_ratio[size - n_pi] + sorted_density_ratio[size - n_pi - 1]) / 2
        # print("threshold:", threshold)
        h = np.sign(density_ratio - threshold).astype(np.int32)
        return h


class ThreeLayerPerceptron(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(ThreeLayerPerceptron, self).__init__()

        self.input_dim = dim
        self.l1 = nn.Linear(dim, 100)
        self.l2 = nn.Linear(100, 1)

        self.af = F.relu

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.l1(x)
        x = self.af(x)
        x = self.l2(x)
        return x


class MultiLayerPerceptron(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = dim

        self.l1 = nn.Linear(dim, 300, bias=False)
        self.b1 = nn.BatchNorm1d(300)
        self.l2 = nn.Linear(300, 300, bias=False)
        self.b2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300, bias=False)
        self.b3 = nn.BatchNorm1d(300)
        self.l4 = nn.Linear(300, 300, bias=False)
        self.b4 = nn.BatchNorm1d(300)
        self.l5 = nn.Linear(300, 1)
        self.af = F.relu

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h = self.l1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.l2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.l3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.l4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.l5(h)
        return h


class CNN(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(CNN, self).__init__()

        self.af = F.relu
        self.input_dim = dim

        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(96)
        self.b3 = nn.BatchNorm2d(96)
        self.b4 = nn.BatchNorm2d(192)
        self.b5 = nn.BatchNorm2d(192)
        self.b6 = nn.BatchNorm2d(192)
        self.b7 = nn.BatchNorm2d(192)
        self.b8 = nn.BatchNorm2d(192)
        self.b9 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(640, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)

        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
