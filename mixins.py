# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: mixins.py
@time: 2021/5/7 下午4:27
@desc: 
"""
from unittest import TestCase

import torch


class TestMixin:
    class TestModelMixin(TestCase):
        def setUp(self) -> None:
            self.model = None
            self.input = None

        def test_all_parameters_updated(self):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            output = self.model(*self.input)
            loss = output.mean()
            loss.backward()
            optimizer.step()

            for param_name, param in self.model.named_parameters():
                if param.requires_grad:
                    with self.subTest(name=param_name):
                        self.assertIsNotNone(param.grad)
                        self.assertNotEqual(0., torch.sum(param.grad ** 2))
