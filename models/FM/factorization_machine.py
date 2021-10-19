# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: factorization_machine.py
@time: 2021/9/15
@desc: 
"""
from typing import List

import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, num_numeric_features: int, num_cat_features: int, nunique_cat_features: List[int],
                 embedding_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(in_features=num_numeric_features + num_cat_features, out_features=1)

        embeddings = []
        for nunique in nunique_cat_features:
            embeddings.append(nn.Embedding(num_embeddings=nunique, embedding_dim=embedding_dim, padding_idx=0))
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, numeric_features: torch.Tensor, cat_features: torch.Tensor):
        """
        Args:
            numeric_features: shape [batch_size, num_numeric_features]
            cat_features: shape [batch_size, num_cat_features]

        Returns:
        shape [batch_size,]
        """
        # 一阶
        # shape [batch_size, ]
        features = torch.cat((numeric_features, cat_features), dim=1)
        order_one_output = self.linear(features)
        order_one_output = order_one_output.view(-1)

        # 二阶
        cat_embeddings = []
        for i, embedding in enumerate(self.embeddings):
            embedding = torch.unsqueeze(embedding(cat_features[:, i]), dim=1)
            cat_embeddings.append(embedding)
        # shape [batch_size, num_cat_features, embedding_dim]
        v = torch.cat(cat_embeddings, dim=1)
        squared_sum = torch.sum(v, 1) ** 2
        sum_squared = torch.sum(v ** 2, 1)
        # shape [batch_size,]
        order_two_output = torch.sum(squared_sum - sum_squared, 1) * 0.5

        # 输出
        sigmoid = nn.Sigmoid()
        y_hat = sigmoid(order_one_output + order_two_output)

        return y_hat
