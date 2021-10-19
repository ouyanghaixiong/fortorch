# -*- coding: utf-8 -*-
"""
@author: @author: ouyhaix@icloud.com
@file: modelling.py
@time: 2021/10/12
@desc: https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf
"""
import torch
import torch.nn.functional as F

from models.RankNet.configuration import RankNetConfig


class RankNetMLP(torch.nn.Module):
    """
    Predict the score of a document.
    """

    def __init__(self, config: RankNetConfig):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=config.in_features, out_features=config.hidden_size)
        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.fc2 = torch.nn.Linear(in_features=config.hidden_size, out_features=config.out_features)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape [batch_size, in_features]

        Returns:
            out: shape [batch_size, out_features]
        """
        # shape [batch_size, hidden_size]
        out = self.fc1(x)
        out = self.bn(out)
        out = F.relu(out)

        # shape [batch_size, out_features]
        out = self.fc2(out)
        out = F.relu6(out)

        return out

    def init_parameters(self):
        for name, w in self.named_parameters():
            if "bn" in name:
                continue
            if 'bias' in name:
                torch.nn.init.constant_(w, 0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(w)


class RankNet(torch.nn.Module):
    """
    Predict the difference of the score between a document pair.
    """

    def __init__(self, config: RankNetConfig):
        super().__init__()
        self.mlp = RankNetMLP(config)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Args:
            x1: shape [batch_size, in_features]
            x2: shape [batch_size, in_features]

        Returns:
            out: shape [batch_size, out_features]
        """
        out1 = self.mlp(x1)
        out2 = self.mlp(x2)

        return torch.sigmoid(out1 - out2)

    def init_parameters(self):
        self.mlp.init_parameters()
