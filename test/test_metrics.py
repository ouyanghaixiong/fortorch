# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/10/14
@desc: 
"""
from unittest import TestCase

import torch

from consts import DEFAULT_DTYPE
from models.metrics import ndcg_score_i, dcg_at_n, ndcg_score


class Test(TestCase):
    def test_dcg_at_n(self):
        scores = torch.tensor([[4], [1], [0], [2]], dtype=DEFAULT_DTYPE)
        dcg = dcg_at_n(scores, k=4)
        self.assertTrue(dcg > 0)
        self.assertIsInstance(dcg, float)

    def test_ndcg_score_i(self):
        y_pred = torch.tensor([[4], [1], [0], [2]], dtype=DEFAULT_DTYPE)
        y_true = torch.tensor([[0], [0], [4], [3]], dtype=DEFAULT_DTYPE)
        ndcg = ndcg_score_i(y_pred, y_true)
        self.assertIsInstance(ndcg, float)
        self.assertTrue(0 <= ndcg <= 1)
        self.assertEqual(round(ndcg, 4), 0.5602)

        y_pred = torch.tensor([[4], [1], [0], [2]], dtype=DEFAULT_DTYPE)
        y_true = torch.tensor([[4], [1], [0], [2]], dtype=DEFAULT_DTYPE)
        ndcg = ndcg_score_i(y_pred, y_true)
        self.assertIsInstance(ndcg, float)
        self.assertTrue(0 <= ndcg <= 1)
        self.assertEqual(round(ndcg, 4), 1.0)

        y_pred = torch.tensor([[0], [1], [2], [3]], dtype=DEFAULT_DTYPE)
        y_true = torch.tensor([[4], [3], [2], [1]], dtype=DEFAULT_DTYPE)
        ndcg = ndcg_score_i(y_pred, y_true)
        self.assertIsInstance(ndcg, float)
        self.assertTrue(0 <= ndcg <= 1)
        self.assertEqual(round(ndcg, 4), 0.6021)

    def test_ndcg_score(self):
        y_pred0 = torch.tensor([[4], [1], [0], [2]], dtype=DEFAULT_DTYPE)
        y_true0 = torch.tensor([[0], [0], [4], [3]], dtype=DEFAULT_DTYPE)

        y_pred1 = torch.tensor([[4], [1], [0], [2]], dtype=DEFAULT_DTYPE)
        y_true1 = torch.tensor([[4], [1], [0], [2]], dtype=DEFAULT_DTYPE)

        y_pred2 = torch.tensor([[0], [1], [2], [3]], dtype=DEFAULT_DTYPE)
        y_true2 = torch.tensor([[4], [3], [2], [1]], dtype=DEFAULT_DTYPE)

        y_preds = [y_pred0, y_pred1, y_pred2]
        y_trues = [y_true0, y_true1, y_true2]

        ndcg = ndcg_score(y_preds, y_trues)
        expected = (ndcg_score_i(y_pred0, y_true0) +
                    ndcg_score_i(y_pred1, y_true1) +
                    ndcg_score_i(y_pred2, y_true2)) / 3
        self.assertEqual(round(expected, 4), round(ndcg, 4))
