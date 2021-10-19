# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/9/15
@desc: 
"""
from unittest import TestCase

import torch

from models.FM.factorization_machine import FM


class TestFM(TestCase):
    def test_forward(self):
        model = FM(num_numeric_features=1, num_cat_features=2, nunique_cat_features=[10, 10], embedding_dim=128)
        y = torch.IntTensor([0, 1, 1, 0, 0])
        numeric_features = torch.FloatTensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        cat_features = torch.IntTensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        output = model(numeric_features, cat_features)
        print(output.shape)
        print(output)
