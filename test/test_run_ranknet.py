# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/10/13
@desc: 
"""
from unittest import TestCase

import torch

from consts import DEVICE
from models.RankNet.run_ranknet import calculate_loss, calculate_lambda


class Test(TestCase):
    def test_calculate_loss(self):
        yhat = torch.tensor([[0.5], [0.8], [2.5], [3.0], [4.1]], device=DEVICE)
        doc_pairs = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3)]
        documents_indexes = [0, 1, 2, 3, 4]
        loss = calculate_loss(yhat, doc_pairs, documents_indexes)
        self.assertTrue(loss >= 0)

    def test_calculate_lambda(self):
        yhat = torch.tensor([[0.5], [0.8], [2.5], [3.0], [4.1]], device=DEVICE)
        doc_pairs = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3)]
        documents_indexes = [0, 1, 2, 3, 4]
        lambda_vector = calculate_lambda(yhat, doc_pairs, documents_indexes)
        self.assertEqual(torch.Size([5, 1]), lambda_vector.size())
