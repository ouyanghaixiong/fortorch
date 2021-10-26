# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: loss.py
@time: 2021/3/19
@desc: 
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class FocalCrossEntropy(torch.nn.Module):
    """
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: int = 2, size_average: bool = True):
        """
        Args:
            alpha: shape [num_classes, 1], the loss weights for classes
            gamma: > 0; reduces the relative loss for well-classified examples (p > .5),
                               putting more focus on hard, mis-classified examples
            size_average: By default, the losses are averaged over observations for each mini batch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each mini batch.
        """
        super().__init__()
        self.alpha = alpha.reshape(-1, 1)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Args:
            logit: shape [batch_size, num_classes]
            y_true: shape [batch_size, 1]

        Returns:
            loss
        """
        if logit.size()[0] != y_true.size()[0]:
            raise ValueError(
                f"y_pred and y_true should have the same batch size, but received {logit.size()[0]} and {y_true.size()[0]}"
            )
        if self.alpha.size()[0] != logit.size()[1]:
            raise ValueError(f"{self.alpha.size()[0]} != {logit.size()[1]}")

        if self.alpha is None:
            self.alpha = Variable(torch.ones(logit.size()[1], 1))
        if not isinstance(self.alpha, Variable):
            self.alpha = Variable(self.alpha)
        if logit.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        y_pred = F.softmax(logit, dim=1)
        y_true = y_true.reshape(-1, 1)
        # shape [batch_size, 1]
        alpha = self.alpha[y_true.type(torch.LongTensor).data.reshape(-1, 1)]

        # label mask
        y_pred_mask = logit.data.new(logit.size()[0], logit.size()[1]).fill_(0)
        y_pred_mask = Variable(y_pred_mask)
        y_pred_mask.scatter_(1, y_true.data, 1.)

        # probabilities for every true label
        p = (y_pred * y_pred_mask).sum(1).reshape(-1, 1)
        log_p = torch.log(p)

        batch_loss = -alpha * torch.pow((1 - p), self.gamma) * log_p

        return batch_loss.mean() if self.size_average else batch_loss.sum()


class FocalBinaryCrossEntropy(torch.nn.Module):
    """
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    """

    def __init__(self, alpha: float = None, gamma: int = 2, size_average: bool = True):
        """
        Args:
            alpha: the loss weight for the positive class
            gamma: > 0; reduces the relative loss for well-classified examples (p > .5),
                               putting more focus on hard, mis-classified examples
            size_average: By default, the losses are averaged over observations for each mini batch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each mini batch.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Args:
            logit: shape [batch_size, 1]
            y_true: shape [batch_size, 1]

        Returns:
            loss
        """
        if logit.size()[0] != y_true.size()[0]:
            raise ValueError(
                f"y_pred and y_true should have the same batch size, but received {logit.size()[0]} and {y_true.size()[0]}"
            )

        if self.alpha is None:
            self.alpha = 0.5

        y_pred = torch.sigmoid(logit.reshape(-1, 1))
        y_true = y_true.reshape(-1, 1)

        batch_loss = - self.alpha * torch.pow((1 - y_pred), self.gamma) * y_true * torch.log(y_pred) \
                     - (1 - self.alpha) * torch.pow(y_pred, self.gamma) * (1 - y_true) * torch.log(1 - y_pred)

        return batch_loss.mean() if self.size_average else batch_loss.sum()


class GHMBinaryCrossEntropy(torch.nn.Module):
    """
    https://arxiv.org/abs/1811.05181
    """

    def __init__(self, bins=10, momentum=0, reduction="mean"):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum_gradient_length = torch.zeros(bins)
            self.register_buffer('acc_sum_gradient_length', acc_sum_gradient_length)
        self.reduction = reduction
        assert self.reduction in (None, 'none', 'mean', 'sum')

    def forward(self, y_pred, y_true, label_weight=None):
        """Calculate the GHM-C loss.
        Args:
            y_pred: shape [batch_size, 1], the direct prediction of classification fc layer
            y_true: shape [batch_size, 1] binary class target for each sample
            label_weight: shape [batch_size, 1]  the value is 1 if the sample is valid and 0 if ignored

        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if y_pred.dim() != y_true.dim() or y_pred.size()[1] != 1:
            raise ValueError(f"the dim of y_pred {y_pred.dim()} != the dim of y_true {y_true.dim()}")
        if label_weight is None:
            label_weight = torch.ones_like(y_pred)
        y_true, label_weight = y_true.float(), label_weight.float()
        weights = torch.zeros_like(y_pred)

        # gradient length, shape [batch_size, 1]
        g = torch.abs(y_pred.sigmoid().detach() - y_true)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum_gradient_length[i] \
                                      + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weights, reduction=self.reduction)

        return loss


class GHMCrossEntropy(torch.nn.Module):
    """
    https://arxiv.org/abs/1811.05181
    """

    def __init__(self, bins=10, momentum=0, reduction="mean"):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum_gradient_length = torch.zeros(bins)
            self.register_buffer('acc_sum_gradient_length', acc_sum_gradient_length)
        self.reduction = reduction
        assert self.reduction in (None, 'none', 'mean', 'sum')

    def forward(self, y_pred, y_true, label_weight=None):
        """Calculate the GHM-C loss.
        Args:
            y_pred: shape [batch_size, num_classes], the direct prediction of classification fc layer
            y_true: shape [batch_size, num_classes] binary class target for each sample
            label_weight: shape [batch_size, 1]  the value is 1 if the sample is valid and 0 if ignored

        Returns:
            The gradient harmonized loss.
        """
        if y_pred.dim() != y_true.dim():
            raise ValueError(f"the dim of y_pred {y_pred.dim()} != the dim of y_true {y_true.dim()}")
        if label_weight is None:
            label_weight = torch.ones(size=(y_pred.size()[0], 1))
        label_weight = label_weight.float()
        weights = torch.zeros(size=(y_pred.size()[0], 1))

        # gradient length, shape [batch_size, 1]
        with torch.no_grad():
            g = torch.sum(
                torch.abs(
                    torch.gather(1 - torch.softmax(y_pred, dim=1), dim=1, index=y_true)
                ),
                dim=1, keepdim=True
            )

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum_gradient_length[i] \
                                      + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss_ce = F.cross_entropy(y_pred, y_true.reshape(-1), reduce=False)
        print(loss_ce)
        loss = weights * loss_ce.reshape(-1, 1)
        print(loss)

        return loss.sum() if self.reduction == 'sum' else loss.mean()
