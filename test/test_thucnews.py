# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/9/27
@desc: 
"""
from unittest import TestCase

from datasets.thucnews import THUCNewsDataset


class TestTHUCNewsDatasets(TestCase):
    def test__process(self):
        train = THUCNewsDataset(mode="train")
        self.assertEqual(180000, len(train))
        print(train.__getitem__(0)[0])
        print(train.__getitem__(0)[1])

        dev = THUCNewsDataset(mode="dev")
        self.assertEqual(10000, len(dev))
        print(dev.__getitem__(0)[0])
        print(dev.__getitem__(0)[1])

        test = THUCNewsDataset(mode="test")
        self.assertEqual(10000, len(test))
        print(test.__getitem__(0)[0])
        print(test.__getitem__(0)[1])
