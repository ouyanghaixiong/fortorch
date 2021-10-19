# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: logistic.py
@time: 2021/9/16
@desc: 
"""

import torch.nn as nn


class LogisticClassifier(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1)
        self.bn = nn.BatchNorm1d(num_features=input_size)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.linear.weight.data)

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        output = self.sigmoid(x)

        return output
