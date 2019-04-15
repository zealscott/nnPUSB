import numpy
import torch
from torch import nn
from torch.autograd import Function


class nnPUSBloss(nn.Module):
    """Loss function for PUSB learning."""

    def __init__(self, prior, gamma=1, beta=0):
        super(nnPUSBloss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.positive = 1
        self.unlabeled = -1
        self.eps = 1e-7

    def forward(self, x, t):
        # clip the predict value to make the following optimization problem well-defined.
        x = torch.clamp(x, min=self.eps, max=1-self.eps)

        t = t[:, None]
        # positive: if positive,1 else 0
        # unlabeled: if unlabeled,1 else 0
        positive, unlabeled = (t == self.positive).float(
        ), (t == self.unlabeled).float()
        n_positive, n_unlabeled = max(1., positive.sum().item()), max(
            1., unlabeled.sum().item())
        y_positive = -torch.log(x)
        y_unlabeled = -torch.log(1 - x)
        positive_risk = torch.sum(
            self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        # print("positive_risk:", positive_risk.item())
        # print("negative_risk:", negative_risk.item())

        objective = positive_risk + negative_risk
        # nnPU learning
        if negative_risk.item() < -self.beta:
            objective = positive_risk - self.beta
            x_out = -self.gamma * negative_risk
        else:
            x_out = objective
        return x_out


def nnPUSB_loss(x, t, prior):
    """wrapper of loss function for non-negative PU with a select bias learning

        .. math::
            L_[\\-pi E_1[\\log(f(x))]+\\max(-E_X[\\log[1-f(x)]+\\pi E_1[\\log(1-f(x))], \\beta)+ R(f) 

    Args:
        x (~torch.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, 1).
        t (~torch.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, ).
        prior (float): Constant variable for class prior.
        loss (~torch.function): loss function.
            The loss function should be non-increasing.
        R(f):  L2 regularization of f, in pytorch, L2 regularisation is mysteriously added in the Optimization functions like ``Adam``

    Returns:
        ~torch.Variable: A variable object holding a scalar array of the
            PU loss.

    See:
        Masahiro Kato and Takeshi Teshima and Junya Honda. 
        "Learning from Positive and Unlabeled Data with a Selection Bias."
        International Conference on Learning Representations. 2019.
    """
    return nnPUSBloss(prior=prior)(x, t)


class nnPUloss(nn.Module):
    """Loss function for PU learning."""

    def __init__(self, prior, loss, gamma=1, beta=0):
        super(nnPUloss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.positive = 1
        self.unlabeled = -1

    def forward(self, x, t):
        t = t[:, None]
        # positive: if positive,1 else 0
        # unlabeled: if unlabeled,1 else 0
        positive, unlabeled = (t == self.positive).float(
        ), (t == self.unlabeled).float()
        n_positive, n_unlabeled = max(1., positive.sum().item()), max(
            1., unlabeled.sum().item())
        y_positive = self.loss_func(x)
        y_unlabeled = self.loss_func(-x)
        positive_risk = torch.sum(
            self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        objective = positive_risk + negative_risk
        # nnPU learning
        if negative_risk.item() < -self.beta:
            objective = positive_risk - self.beta
            x_out = -self.gamma * negative_risk
        else:
            x_out = objective
        return x_out


def nnPU_loss(x, t, prior, loss):
    """wrapper of loss function for non-negative PU learning

        .. math::
            L_[\\pi E_1[l(f(x))]+\\max(E_X[l(-f(x))]-\\pi E_1[l(-f(x))], \\beta) 

    Args:
        x (~torch.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, 1).
        t (~torch.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, ).
        prior (float): Constant variable for class prior.
        loss (~torch.function): loss function.
            The loss function should be non-increasing.

    Returns:
        ~torch.Variable: A variable object holding a scalar array of the
            PU loss.

    See:
        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
        "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
        Advances in neural information processing systems. 2017.
        du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama.
        "Convex formulation for learning from positive and unlabeled data."
        Proceedings of The 32nd International Conference on Machine Learning. 2015.
    """
    return nnPUloss(prior=prior, loss=loss)(x, t)
