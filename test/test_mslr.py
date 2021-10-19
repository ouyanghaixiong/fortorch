# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/10/11
@desc: 
"""
import os
from unittest import TestCase

import numpy as np
import torch

from consts import DATA_DIR
from datasets.mslr import MSLRPairDataset, MSLRDataset, get_filtered_doc_pairs, parse_raw_data, get_doc_pairs, \
    get_query_documents_map


class TestMSLRPairDataset(TestCase):
    def test_getitem(self):
        train = MSLRPairDataset(file_path=os.path.join(DATA_DIR, "MSLR", "train.txt"))
        labels, documents_i, documents_j = train.__getitem__(0)
        self.assertEqual(torch.Size([12, 1]), labels.size())
        self.assertEqual(torch.Size([12, 136]), documents_i.size())
        self.assertEqual(torch.Size([12, 136]), documents_j.size())

        dev = MSLRPairDataset(file_path=os.path.join(DATA_DIR, "MSLR", "vali.txt"))
        labels, documents_i, documents_j = dev.__getitem__(0)
        self.assertEqual(torch.Size([12, 1]), labels.size())
        self.assertEqual(torch.Size([12, 136]), documents_i.size())
        self.assertEqual(torch.Size([12, 136]), documents_j.size())

        test = MSLRPairDataset(file_path=os.path.join(DATA_DIR, "MSLR", "test.txt"))
        labels, documents_i, documents_j = test.__getitem__(0)
        self.assertEqual(torch.Size([12, 1]), labels.size())
        self.assertEqual(torch.Size([12, 136]), documents_i.size())
        self.assertEqual(torch.Size([12, 136]), documents_j.size())


class TestFunctions(TestCase):
    def test_parse_raw_data(self):
        file_path = os.path.join(DATA_DIR, "MSLR", "train.txt")
        samples = parse_raw_data(file_path)
        self.assertEqual((119, 138), samples.shape)

    def test_get_query_documents_map(self):
        file_path = os.path.join(DATA_DIR, "MSLR", "train.txt")
        samples: np.ndarray = parse_raw_data(file_path)
        unique_qids = np.unique(samples[:, 1])
        query_documents_map = get_query_documents_map(samples, unique_qids)
        self.assertEqual(27, len(query_documents_map))
        print(query_documents_map)

    def test_get_doc_pairs(self):
        file_path = os.path.join(DATA_DIR, "MSLR", "train.txt")
        samples: np.ndarray = parse_raw_data(file_path)
        unique_qids = np.unique(samples[:, 1])
        doc_pairs = get_doc_pairs(samples, unique_qids)
        self.assertEqual(27, len(doc_pairs))

    def test_get_filtered_doc_pairs(self):
        file_path = os.path.join(DATA_DIR, "MSLR", "train.txt")
        samples: np.ndarray = parse_raw_data(file_path)
        unique_qids = np.unique(samples[:, 1])
        filtered_doc_pairs = get_filtered_doc_pairs(samples, unique_qids)
        self.assertEqual(27, len(filtered_doc_pairs))

        file_path = os.path.join(DATA_DIR, "MSLR", "vali.txt")
        samples: np.ndarray = parse_raw_data(file_path)
        unique_qids = np.unique(samples[:, 1])
        filtered_doc_pairs = get_filtered_doc_pairs(samples, unique_qids)
        self.assertEqual(9, len(filtered_doc_pairs))

        file_path = os.path.join(DATA_DIR, "MSLR", "test.txt")
        samples: np.ndarray = parse_raw_data(file_path)
        unique_qids = np.unique(samples[:, 1])
        filtered_doc_pairs = get_filtered_doc_pairs(samples, unique_qids)
        self.assertEqual(9, len(filtered_doc_pairs))


class TestMSLRDataset(TestCase):
    def test_getitem(self):
        train = MSLRDataset(file_path=os.path.join(DATA_DIR, "MSLR", "train.txt"))
        labels, features = train.__getitem__(1)
        self.assertEqual(torch.Size([4, 1]), labels.size())
        self.assertEqual(torch.Size([4, 136]), features.size())

        dev = MSLRDataset(file_path=os.path.join(DATA_DIR, "MSLR", "vali.txt"))
        labels, features = dev.__getitem__(1)
        self.assertEqual(torch.Size([6, 1]), labels.size())
        self.assertEqual(torch.Size([6, 136]), features.size())

        test = MSLRDataset(file_path=os.path.join(DATA_DIR, "MSLR", "test.txt"))
        labels, features = test.__getitem__(1)
        self.assertEqual(torch.Size([6, 1]), labels.size())
        self.assertEqual(torch.Size([6, 136]), features.size())
