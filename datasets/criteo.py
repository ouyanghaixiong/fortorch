# -*- coding: utf-8 -*-
"""
@author: bearouyang@icloud.com
@file: criteo.py
@time: 2021/9/15
@desc: 
"""
from typing import List, Tuple, Dict

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset


class CriteoDataset(Dataset):
    def __init__(self, file_path: str, is_training: bool = True, discretize: bool = False):
        self.is_training = is_training
        self.discretize = discretize

        # load data
        self.data = pd.read_csv(file_path)
        self.cols: List[str] = self.data.columns.tolist()
        self.numeric_cols: List[str] = [col for col in self.data.columns.tolist() if col.startswith("I")]
        self.cat_cols: List[str] = [col for col in self.data.columns.tolist() if col.startswith("C")]
        self.preprocess()

        # split dataset
        train_size = int(self.data.shape[0] * 0.8)
        self.train = self.data.iloc[:train_size, :]
        self.test = self.data.iloc[train_size:, :]

    def __getitem__(self, index) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.IntTensor] or Tuple[
            torch.IntTensor, torch.IntTensor]:
        data = self.train if self.is_training else self.test
        label = torch.as_tensor(data.iloc[index, 0], dtype=torch.float32)
        if self.discretize:
            features = torch.FloatTensor(data.iloc[index, 1:].values)
            return label, features
        numeric_features = torch.FloatTensor(data.iloc[index, 1:14].values)
        cat_features = torch.IntTensor(data.iloc[index, 14:].values)
        return label, numeric_features, cat_features

    def __len__(self):
        return self.train.shape[0] if self.is_training else self.test.shape[0]

    @property
    def nunique_cat_features(self) -> List[int]:
        """
        Calculate the number of the unique feature values for all categorical features.
        Returns:
        The list of nunique number. Every index represents a feature.
        """
        res = []
        for cat_col in self.cat_cols:
            res.append(self.data[cat_col].nunique())

        return res

    @property
    def vocabulary(self) -> Dict[str, int]:
        """
        Returns:
        The vocabulary map of every feature in the dataset.
        key: the feature name
        value: the number of unique values of the feature
        """
        res = {}
        for cat_col in self.cat_cols:
            res[cat_col] = self.data[cat_col].nunique()
        for numeric_col in self.numeric_cols:
            res[numeric_col] = 1

        return res

    def preprocess(self):
        """
        Preprocess the DataFrame.
        Returns:
        None
        """
        # fill na
        self.data[self.numeric_cols] = self.data[self.numeric_cols].fillna(0)
        self.data[self.cat_cols] = self.data[self.cat_cols].fillna("NaN")

        if self.discretize:
            # discretization
            for col in self.numeric_cols:
                nunique = self.data[col].nunique()
                q = 10 if nunique >= 10 else nunique
                self.data[col] = pd.qcut(self.data[col], q=q, duplicates="drop")
        else:
            # normalization
            scaler = MinMaxScaler()
            self.data[self.numeric_cols] = scaler.fit_transform(self.data[self.numeric_cols])

        # label encoding
        cols = self.numeric_cols + self.cat_cols if self.discretize else self.cat_cols
        for col in cols:
            label_encoder = LabelEncoder()
            self.data[col] = label_encoder.fit_transform(self.data[col])
