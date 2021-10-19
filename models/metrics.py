# -*- coding: utf-8 -*-
"""
@author: ouyhaix@icloud.com
@file: metrics.py
@time: 2021/10/14
@desc: 
"""
from typing import Optional, List

import torch


@torch.no_grad()
def dcg_at_n(scores: torch.Tensor, k: int) -> float:
    """
    Calculate the dcg score for the a query.
    Args:
        scores: shape [num_items, 1] the true scores in descending order
        k: top k items to involved in calculation

    Returns:
        dcg score for a query
    """
    if scores.size()[1] != 1:
        raise ValueError(f"Wrong input size of scores: {scores.size()}. " +
                         f"The input size must be torch.Size([num_items, 1]).")

    if scores.numel() > k:
        scores = scores[:k]
    dcg: torch.Tensor = torch.sum(
        (torch.pow(2, scores) - 1) / torch.log2(torch.arange(2, 2 + k).reshape(-1, 1)), dim=0
    )

    return dcg.item()


@torch.no_grad()
def ndcg_score_i(y_pred: torch.Tensor, y_true: torch.Tensor, k: Optional[int] = -1) -> float:
    """
    Calculate the ndcg@k for a query.
    Args:
        y_pred: shape [num_items, 1]
        y_true: shape [num_items, 1]
        k: top k items to involved in calculation

    Returns:
        ndcg score for a query
    """
    if y_pred.size() != y_true.size():
        raise ValueError(f"""The number of element of y_pred must be equal to y_true's,
         not {y_pred.numel()} and {y_true.numel()}""")
    if y_pred.ndim != 2 or y_pred.size()[1] != 1:
        raise ValueError(f"Wrong input size of y_pred: {y_pred.size()}," +
                         f" the size must be torch.Size([num_items, 1])")
    if y_true.ndim != 2 or y_true.size()[1] != 1:
        raise ValueError(f"Wrong input size of y_true: {y_true.size()}," +
                         f" the size must be torch.Size([num_items, 1]).")

    if k == -1: k = y_pred.numel()
    # 将query中的所有item预测分数降序排列, 取排列后的索引
    index_order: torch.Tensor = torch.argsort(y_pred, descending=True, dim=0).reshape(-1)
    # 根据上述索引将真实分数重排序, 用于计算DCG
    dcg = dcg_at_n(y_true[index_order], k=k)
    # 对真实分数降序排列, 用于计算iNDCG
    idcg = dcg_at_n(torch.sort(y_true, descending=True, dim=0)[0], k=k)

    return 0 if idcg == 0 else dcg / idcg


@torch.no_grad()
def ndcg_score(y_preds: List[torch.Tensor], y_trues: List[torch.Tensor]) -> float:
    """
    Calculate the average ndcg@k for all query.
    Args:
        y_preds: the predictive scores of every item for every query
        y_trues: the true scores of every item for every query

    Returns:
        average ndcg score for all query
    """
    if y_preds.__len__() != y_trues.__len__():
        raise ValueError("We expect that y_preds and y_trues has the same number of element.")

    m = y_preds.__len__()
    ndcg = 0
    for i in range(y_preds.__len__()):
        ndcg_i = ndcg_score_i(y_preds[i], y_trues[i])
        ndcg += ndcg_i / m

    return ndcg
