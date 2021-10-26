# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/10/19
@desc: 
"""
from unittest import TestCase

import torch

from consts import DTYPE, DEVICE
from models.loss import FocalCrossEntropy, FocalBinaryCrossEntropy, GHMBinaryCrossEntropy, GHMCrossEntropy


class TestFocalCrossEntropy(TestCase):
    def test_forward(self):
        alpha = torch.tensor([0.25, 0.50, 0.75])
        criterion = FocalCrossEntropy(alpha=alpha, gamma=2)
        logit = torch.tensor([[1, 5, 10], [2, 7, 8], [3, 1, 9], [4, 5, 99], [5, 66, 999]], dtype=DTYPE, device=DEVICE)
        y_true = torch.tensor([[0], [0], [1], [1], [2]], dtype=torch.int64, device=DEVICE)
        loss0 = criterion(logit, y_true)
        self.assertTrue(loss0.item() > 0)
        print(loss0)

        logit = torch.tensor([[1, 5, 10], [2, 7, 8], [3, 1, 9], [4, 5, 99], [5, 66, 999]], dtype=DTYPE, device=DEVICE)
        y_true = torch.tensor([[2], [2], [2], [2], [2]], dtype=torch.int64, device=DEVICE)
        loss1 = criterion(logit, y_true)
        self.assertTrue(loss1.item() > 0)
        self.assertTrue(loss1.item() < loss0.item())
        print(loss1)

        logit = torch.tensor([[0, 0, 10], [0, 0, 8], [0, 0, 9], [0, 0, 99], [0, 0, 999]], dtype=DTYPE, device=DEVICE)
        y_true = torch.tensor([[2], [2], [2], [2], [2]], dtype=torch.int64, device=DEVICE)
        loss2 = criterion(logit, y_true)
        self.assertTrue(loss2.item() < loss1.item())
        print(loss2)


class TestFocalBinaryCrossEntropy(TestCase):
    def test_forward(self):
        criterion = FocalBinaryCrossEntropy(alpha=0.8, gamma=2)

        logit = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32, device=DEVICE)
        y_true = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.int64, device=DEVICE)
        loss0 = criterion(logit, y_true)
        self.assertTrue(loss0.item() > 0)

        logit = torch.tensor([[0], [0], [10], [10], [10]], dtype=torch.float32, device=DEVICE)
        y_true = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.int64, device=DEVICE)
        loss1 = criterion(logit, y_true)
        self.assertTrue(loss0.item() > 0)
        self.assertTrue(loss1.item() < loss0.item())


class TestGHMBinaryCrossEntropy(TestCase):
    def test_forward(self):
        criterion = GHMBinaryCrossEntropy(bins=10, momentum=0, reduction="mean")

        logit = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32, device=DEVICE)
        y_true = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.int64, device=DEVICE)
        loss0 = criterion(logit, y_true)
        self.assertTrue(loss0.item() > 0)

        logit = torch.tensor([[0], [0], [10], [10], [10]], dtype=torch.float32, device=DEVICE)
        y_true = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.int64, device=DEVICE)
        loss1 = criterion(logit, y_true)
        self.assertTrue(loss0.item() > 0)
        self.assertTrue(loss1.item() < loss0.item())


class TestGHMCrossEntropy(TestCase):
    def test_forward(self):
        criterion = GHMCrossEntropy(bins=10, momentum=0, reduction="mean")

        logit = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=torch.float32,
                             device=DEVICE)
        y_true = torch.tensor([[1], [1], [1], [1], [1]], dtype=torch.int64, device=DEVICE)
        loss0 = criterion(logit, y_true)
        print(loss0)
        self.assertTrue(loss0.item() > 0)

        logit = torch.tensor([[1, 5, 10], [2, 7, 8], [3, 1, 9], [4, 5, 99], [5, 66, 999]], dtype=DTYPE, device=DEVICE)
        y_true = torch.tensor([[2], [2], [2], [2], [2]], dtype=torch.int64, device=DEVICE)
        loss1 = criterion(logit, y_true)
        self.assertTrue(loss1.item() > 0)
        self.assertTrue(loss1.item() < loss0.item())
        print(loss1)

        logit = torch.tensor([[0, 0, 10], [0, 0, 8], [0, 0, 9], [0, 0, 99], [0, 0, 999]], dtype=DTYPE, device=DEVICE)
        y_true = torch.tensor([[2], [2], [2], [2], [2]], dtype=torch.int64, device=DEVICE)
        loss2 = criterion(logit, y_true)
        self.assertTrue(loss2.item() < loss1.item())
        print(loss2)
