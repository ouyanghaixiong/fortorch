# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/9/16
@desc: 
"""
from unittest import TestCase

import torch
import torch.nn as nn

from models.LR.logistic import LogisticClassifier
from utils import get_logger

logger = get_logger(__name__)


class TestLogisticClassifier(TestCase):
    def test_forward(self):
        y = torch.FloatTensor([0, 1, 1, 0, 0])
        x = torch.FloatTensor([[1, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1]])
        model = LogisticClassifier(input_size=3)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for epoch in range(10000):
            model.train()
            y_hat = model(x).view(-1)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            logger.info(f"step: {epoch} loss: {loss}")
            for name, params in model.named_parameters():
                logger.info(f"name: {name} weight {params.data} grad {params.grad}")
