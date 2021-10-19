# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: test_criteo.py
@time: 2021/9/15
@desc: 
"""
import os
from unittest import TestCase

import torch

from consts import DATA_DIR
from datasets.criteo import CriteoDataset


class TestCriteoDataset(TestCase):
    def test_get_item(self):
        file_path = os.path.join(DATA_DIR, "criteo", "criteo_sampled_data.csv")
        dataset = CriteoDataset(file_path=file_path, is_training=True)
        self.assertTrue(dataset.__len__() > 0)

    def test_discretize(self):
        file_path = os.path.join(DATA_DIR, "criteo", "criteo_sampled_data.csv")
        dataset = CriteoDataset(file_path=file_path, is_training=True, discretize=True)
        features = dataset.__getitem__(0)[1]
        self.assertTrue(torch.Size([39]) == features.shape)

    def test_len(self):
        file_path = os.path.join(DATA_DIR, "criteo", "criteo_sampled_data.csv")
        dataset = CriteoDataset(file_path=file_path, is_training=True)
        self.assertEqual(480 * 1000, dataset.__len__())
        dataset = CriteoDataset(file_path=file_path, is_training=False)
        self.assertEqual(120 * 1000, dataset.__len__())

    def test_nunique_cat_features(self):
        file_path = os.path.join(DATA_DIR, "criteo", "criteo_sampled_data.csv")
        dataset = CriteoDataset(file_path=file_path, is_training=True)
        self.assertEqual(26, dataset.nunique_cat_features.__len__())
