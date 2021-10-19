# -*- coding: utf-8 -*-
"""
@author: ouyhaix@icloud.com
@file: mslr.py
@time: 2021/10/11
@desc: 
"""
from typing import List, Tuple, Dict

import numpy as np
import torch.utils.data
from torch.utils.data.dataset import T_co

from consts import DEFAULT_DEVICE
from utils import get_logger

logger = get_logger(__name__)

NUM_FEATURES = 136


def parse_raw_data(file_path: str) -> np.ndarray:
    """
    Read and split the data line by line, parse the feature values.
    Args:
        file_path: raw data file path

    Returns:
        samples
            column names: score pid features_of_document...
            [[0 1 -0.617277 -0.109573 -0.763873 0.369905 -0.257625 0.570188 0.734393...]...]
    """
    samples: List[np.ndarray] = []
    with open(file_path, "r") as file:
        # 0 qid:1 115:-0.617277 5:-0.109573 17:-0.763873 17:0.369905 68:-0.257625 125:0.570188 110:0.734393 ...
        for line in file.readlines():
            # 前两个元素是score和qid,后面的都是是文档特征,空值用0表示
            sample = np.zeros((2 + NUM_FEATURES,))
            values: list = line.split(" ")
            # extract score and qid
            score: int = int(values[0])
            qid: int = int(values[1].split(":")[1])
            sample[0] = score
            sample[1] = qid
            # 提取其他特征
            for value in values[2:]:
                k, v = value.split(":")
                feature_position: int = int(k) + 1
                feature_value: float = float(v)
                sample[feature_position] = feature_value
            samples.append(sample)

    # column names: score pid features_of_document...
    # [[0 1 -0.617277 -0.109573 -0.763873 0.369905 -0.257625 0.570188 0.734393...]...]
    samples: np.ndarray = np.asarray(samples)

    return samples


def get_query_documents_map(samples, qids):
    # key: qid  value: indexes of documents like [1,2,3...] ...
    query_documents_map = {}
    for qid in qids:
        # the indexes along dim 0
        indexes_documents = np.where(samples[:, 1] == qid)[0].tolist()
        query_documents_map[qid] = indexes_documents

    return query_documents_map


def get_doc_pairs(samples: np.ndarray, qids: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    """
    Generate item pairs of the item in every query.
    Args:
        samples: query sample data
        qids: the unique query ids
    Returns:
        doc_pairs: the list of document pairs, each tuple represents a pair: (i, j)
    """
    # key: qid  value: doc pairs list like [(0,1), (1,2)...]
    query_doc_pairs_map = {}
    query_documents_map = get_query_documents_map(samples, qids)

    for qid in qids:
        # Tuple: (i, j) where i,j is the index of the document in samples
        doc_pairs_for_query: List[Tuple[int, int]] = []
        indexes_documents = query_documents_map[qid]
        for i in indexes_documents:
            for j in indexes_documents:
                if i == j:
                    continue
                doc_pairs_for_query.append((i, j))
        query_doc_pairs_map[qid] = doc_pairs_for_query

    return query_doc_pairs_map


def get_filtered_doc_pairs(samples, unique_qids) -> Dict[int, List[Tuple[int, int]]]:
    query_filtered_doc_pairs_map = {}
    query_doc_pairs_map = get_doc_pairs(samples, unique_qids)
    for qid, doc_pairs in query_doc_pairs_map.items():
        filtered_doc_pairs_for_query = []
        for i, j in doc_pairs:
            if samples[i, 0] <= samples[j, 0]:
                continue
            filtered_doc_pairs_for_query.append((i, j))
        query_filtered_doc_pairs_map[qid] = filtered_doc_pairs_for_query

    return query_filtered_doc_pairs_map


class MSLRPairDataset(torch.utils.data.Dataset):
    """
    Every sample represents the documents pairs of a query.
    """

    def __init__(self, file_path):
        # read and split the data line by line
        # column names: score pid features_of_document...
        # [[0 1 -0.617277 -0.109573 -0.763873 0.369905 -0.257625 0.570188 0.734393...]...]
        samples: np.ndarray = parse_raw_data(file_path)

        # 提取文档对
        unique_qids = np.unique(samples[:, 1])
        doc_pairs = get_doc_pairs(samples, unique_qids)

        self.samples: np.ndarray = samples
        self.unique_qids = unique_qids
        self.doc_pairs = doc_pairs

    def __getitem__(self, index: int) -> T_co:
        """
        Get the data for each query. So the number of samples contained is dependent on the number of documents related
        to the query.
        Args:
            index: the sample index

        Returns:
            labels: shape [(n_documents - 1) * (n_documents - 1), 1]
            documents_i: shape [(n_documents - 1) * (n_documents - 1), num_features]
            documents_j: shape [(n_documents - 1) * (n_documents - 1), num_features]
        """
        # [(i, j)...]
        qid: int = self.unique_qids[index]
        doc_pairs_for_query: List[Tuple[int, int]] = self.doc_pairs[qid]
        labels = []
        documents_i = []
        documents_j = []
        for i, j in doc_pairs_for_query:
            label = 0 if self.samples[i, 0] < self.samples[j, 0] else 1
            doc_i = self.samples[i, 2:]
            doc_j = self.samples[j, 2:]
            labels.append([label])
            documents_i.append(doc_i)
            documents_j.append(doc_j)
        labels = torch.FloatTensor(labels, device=DEFAULT_DEVICE)
        documents_i = torch.FloatTensor(documents_i, device=DEFAULT_DEVICE)
        documents_j = torch.FloatTensor(documents_j, device=DEFAULT_DEVICE)

        return labels, documents_i, documents_j

    def __len__(self):
        return len(self.unique_qids)


class MSLRDataset(torch.utils.data.Dataset):
    """
    Every sample represents the documents of a query.
    """

    def __init__(self, file_path):
        # read and split the data line by line
        # column names: score pid features_of_document...
        # [[0 1 -0.617277 -0.109573 -0.763873 0.369905 -0.257625 0.570188 0.734393...]...]
        samples: np.ndarray = parse_raw_data(file_path)

        # 提取文档对
        unique_qids = np.unique(samples[:, 1])

        self.samples = samples
        self.unique_qids = unique_qids
        self.doc_pairs = get_filtered_doc_pairs(samples, unique_qids)
        self.query_documents_map = get_query_documents_map(samples, unique_qids)

    def __getitem__(self, index) -> T_co:
        """
        Args:
            index:

        Returns:
            features: shape [n_documents, num_features]
            labels: shape [n_documents, 1]
        """
        qid: int = self.unique_qids[index]
        indexes_documents: List[int] = self.query_documents_map[qid]
        features = self.samples[indexes_documents, 2:]
        labels = torch.FloatTensor(self.samples[indexes_documents, 0].reshape(-1, 1), device=DEFAULT_DEVICE)
        x = torch.FloatTensor(features, device=DEFAULT_DEVICE)

        return labels, x

    def __len__(self):
        return len(self.unique_qids)
