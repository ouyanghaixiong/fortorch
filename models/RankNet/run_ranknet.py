# -*- coding: utf-8 -*-
"""
@author: ouyhaix@icloud.com
@file: run_ranknet.py
@time: 2021/10/11
@desc: https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf
"""
import math
import os
import random
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Union

import numpy as np
import torch.nn

from consts import DATA_DIR, DEFAULT_DEVICE, DEFAULT_DTYPE, SAVE_DIR
from datasets.mslr import MSLRPairDataset, MSLRDataset
from models.RankNet.configuration import RankNetConfig
from models.RankNet.modelling import RankNet, RankNetMLP
from models.metrics import ndcg_score
from models.train import TrainingArgument
from utils import check_sanity, get_logger, set_seed

logger = get_logger(__name__)


def train_normal(train_data: MSLRPairDataset, dev_data: MSLRPairDataset, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, args: TrainingArgument) -> None:
    """
    Train ranknet using normal method, which treat the pairwise ranking problem as classification problem.
    Args:
        train_data:
        dev_data:
        model:
        optimizer:
        criterion:
        args:

    Returns:
        None
    """
    model.init_parameters()
    for epoch in range(args.num_epochs):
        logger.info(f"*** Epoch {epoch} ***")

        model.train()
        for index in range(train_data.__len__()):
            y_for_query, xi_for_query, xj_for_query = train_data[index]
            # 跳过无法形成文档对的query
            if y_for_query.numel() == 0:
                continue
            logger.debug(f"y {y_for_query.size()} xi {xi_for_query.size()} xj {xj_for_query.size()}")
            yhat_for_query = model(xi_for_query, xj_for_query)
            loss = criterion(yhat_for_query, y_for_query)
            optimizer.zero_grad()
            loss.backward()
            for name, params in model.named_parameters():
                logger.debug(
                    f"name: {name} | requires_grad: {params.requires_grad} | mean_grad: {torch.mean(params.grad)}"
                )
            optimizer.step()
            logger.info(f"*Training* loss: {loss.item()}")

        model.eval()
        val_losses = []
        val_ys = []
        val_yhats = []
        with torch.no_grad():
            for index in range(dev_data.__len__()):
                y_for_query, xi_for_query, xj_for_query = dev_data[index]
                # 跳过无法形成文档对的query
                if y_for_query.numel() == 0:
                    continue
                yhat_for_query = model(xi_for_query, xj_for_query)
                loss = criterion(yhat_for_query, y_for_query)
                val_losses.append(loss.item())
                val_ys.append(y_for_query)
                val_yhats.append(yhat_for_query)
        logger.info(f"***Evaluation*** loss: {np.mean(val_losses)} " +
                    f"ndcg:  {ndcg_score(val_yhats, val_ys)}")


def test_normal(test_data: MSLRPairDataset, model: torch.nn.Module, args) -> None:
    """
    Test ranknet using normal method, which treat the pairwise ranking problem as classification problem.
    Args:
        test_data:
        model:
        args:

    Returns:
        None
    """
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    test_ys = []
    test_yhats = []
    with torch.no_grad():
        for index in range(test_data.__len__()):
            y_for_query, xi_for_query, xj_for_query = test_data[index]
            if y_for_query.numel() == 0:
                continue
            test_yhat = model(xi_for_query, xj_for_query)
            test_ys.append(y_for_query)
            test_yhats.append(test_yhat)
    logger.info(f"***Test*** ndcg: {ndcg_score(test_yhats, test_ys)}")


def calculate_loss(yhat: torch.Tensor, doc_pairs: List[Tuple[int, int]], documents_indexes: List[int]) -> float:
    """
    Calculate the sum of loss for item pairs in a query.
    Args:
        yhat: shape [num_items, 1]
        doc_pairs: the item pairs for the current query, notes that the item i is definitely prefer to item j in these pairs
        documents_indexes: the documents indexes for the current query

    Returns:
        loss_for_query: the sum of loss for item pairs in the query
    """
    sigma = 1.0
    loss_for_query = 0
    with torch.no_grad():
        # i j 是item在样本集中的索引
        for index, (i, j) in enumerate(doc_pairs):
            # 需要通过样本索引找到item在yhat中的索引
            index_i, index_j = documents_indexes.index(i), documents_indexes.index(j)
            # 我们只保留i偏好于j的item pair, label一定都是1
            si, sj = yhat[index_i], yhat[index_j]
            # c_ij, loss for one item pair
            loss = math.log(1 + math.exp(-sigma * (si - sj)))
            loss_for_query += loss

    return loss_for_query


def calculate_lambda(yhat: torch.Tensor, doc_pairs: List[Tuple[int, int]],
                     documents_indexes: List[int]) -> torch.Tensor:
    """
    Calculate lambda_ij value for item pairs in a query.
    Args:
        yhat: shape [num_items, 1], the predictive scores for the items in the query
        doc_pairs: the item pairs for the current query, notes that the item i is definitely prefer to item j in these pairs
        documents_indexes: the documents indexes for the current query

    Returns:
        a vector contains the lambda_ij value for item pairs in the query
    """
    sigma = 1.0
    lambda_vector = np.zeros((yhat.numel(), 1))  # 用于保存计算得到的lambdai, key: 文档编号 value: 对应的lambadaij之和
    with torch.no_grad():
        for i, j in doc_pairs:
            # 需要通过样本索引找到yhat当前的索引
            index_i = documents_indexes.index(i)
            index_j = documents_indexes.index(j)
            si, sj = yhat[index_i], yhat[index_j]
            lambda_ij = - sigma * (1 + 1 / (1 + torch.exp(sigma * (si - sj))))
            lambda_vector[index_i, 0] += lambda_ij
            lambda_vector[index_j, 0] -= lambda_ij

    return torch.tensor(lambda_vector, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)


def train_speedup(train_data: MSLRDataset, dev_data: MSLRDataset, model: RankNetMLP, optimizer: torch.optim.Optimizer,
                  args: TrainingArgument) -> None:
    """
    Train ranknet using normal method, which treat the pairwise ranking problem as classification problem.
    Args:
        train_data:
        dev_data:
        model:
        optimizer:
        args:

    Returns:
        None
    """
    # prepare lookup tables
    train_qids: List[int] = train_data.unique_qids.tolist()
    train_doc_pairs: Dict[int, List[Tuple[int, int]]] = train_data.doc_pairs
    train_query_documents_map: Dict[int, List[int]] = train_data.query_documents_map
    val_qids: List[int] = dev_data.unique_qids.tolist()
    val_doc_pairs: Dict[int, List[Tuple[int, int]]] = dev_data.doc_pairs
    val_query_documents_map: Dict[int, List[int]] = dev_data.query_documents_map

    # begin training
    model.init_parameters()
    for epoch in range(args.num_epochs):
        logger.info(f"*** Epoch {epoch} ***")

        model.train()
        # for each query
        query_indexes = list(range(train_data.__len__()))
        random.shuffle(query_indexes)
        for index in query_indexes:
            qid = train_qids[index]
            doc_pairs_for_query = train_doc_pairs[qid]
            documents_for_query = train_query_documents_map[qid]
            if documents_for_query.__len__() <= 1:
                continue
            y_for_query, x_for_query = train_data[index]
            logger.debug(f"y {y_for_query.size()} x {x_for_query.size()}")
            yhat_for_query = model(x_for_query)  # shape [n_item, 1], 即当前query中的每一item的预测得分
            # 当前query的所有 lambda_i, 其中第i个元素对应lambda_i
            lambda_vector: torch.Tensor = calculate_lambda(yhat_for_query, doc_pairs_for_query, documents_for_query)
            optimizer.zero_grad()
            # 注意backward的参数
            yhat_for_query.backward(gradient=lambda_vector)
            for name, params in model.named_parameters():
                logger.debug(
                    f"name: {name} | requires_grad: {params.requires_grad} | mean_grad: {torch.mean(params.grad)}"
                )
            optimizer.step()

        model.eval()
        val_losses = []
        val_ys = []
        val_yhats = []
        with torch.no_grad():
            for index in range(dev_data.__len__()):
                qid = val_qids[index]
                doc_pairs_for_query = val_doc_pairs[qid]
                documents_for_query = val_query_documents_map[qid]
                y_for_query, x_for_query = dev_data[index]
                yhat_for_query = model(x_for_query)
                # 当前query的loss = 当前query中所有在集合I(label=1的pair集合)中对应loss的总和
                loss_for_query: float = calculate_loss(yhat_for_query, doc_pairs_for_query, documents_for_query)
                val_losses.append(loss_for_query)
                val_ys.append(y_for_query)
                val_yhats.append(yhat_for_query)
        logger.info(f"***Evaluation*** loss: {np.mean(loss_for_query)} ndcg: {ndcg_score(val_yhats, val_ys)}")


def test_speedup(test_data: Union[MSLRPairDataset, MSLRDataset], model: torch.nn.Module, args) -> None:
    """
    Test ranknet using normal method, which treat the pairwise ranking problem as classification problem.
    Args:
        test_data:
        model:
        args:

    Returns:
        None
    """
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    ys = []
    yhats = []
    with torch.no_grad():
        for index in range(test_data.__len__()):
            y_for_query, x_for_query = test_data[index]
            if y_for_query.numel() == 0:
                continue
            yhat = model(x_for_query)
            ys.append(y_for_query)
            yhats.append(yhat)
    logger.info(f"***Test*** ndcg: {ndcg_score(yhats, ys)}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, default=__name__, help="任务名称")
    parser.add_argument("--train_file", type=str, default=os.path.join(DATA_DIR, "MSLR", "train.txt"), help="训练集数据文件")
    parser.add_argument("--dev_file", type=str, default=os.path.join(DATA_DIR, "MSLR", "vali.txt"), help="开发集数据文件")
    parser.add_argument("--test_file", type=str, default=os.path.join(DATA_DIR, "MSLR", "test.txt"), help="测试集数据文件")
    parser.add_argument("--model_file", type=str, default=os.path.join(SAVE_DIR, "ranknet", "model.bin"), help="模型文件")

    args = parser.parse_args()
    check_sanity(args)

    return args


def main(training_method="normal"):
    set_seed()
    args = parse_args()

    if training_method == "normal":
        # prepare components
        model_config = RankNetConfig(in_features=136, hidden_size=64, out_features=1)
        training_args = TrainingArgument(num_epochs=10, early_stopping_rounds=100, learning_rate=3e-5, momentum=0.99,
                                         weight_decay=0.1)
        model = RankNet(model_config).to(DEFAULT_DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=training_args.learning_rate, momentum=training_args.momentum,
                                    weight_decay=training_args.weight_decay)
        criterion = torch.nn.BCELoss()

        # train
        train_data = MSLRPairDataset(file_path=args.train_file)
        dev_data = MSLRPairDataset(file_path=args.dev_file)
        train_normal(train_data, dev_data, model, optimizer, criterion, training_args)
        torch.save(model.state_dict(), args.model_file)

        # test
        test_data = MSLRPairDataset(file_path=args.test_file)
        test_normal(test_data, model, args)

    elif training_method == "speedup":
        # prepare components
        model_config = RankNetConfig(in_features=136, hidden_size=64, out_features=1)
        training_args = TrainingArgument(num_epochs=10, early_stopping_rounds=100, learning_rate=3e-5, momentum=0.99,
                                         weight_decay=0.001)
        model = RankNetMLP(model_config).to(DEFAULT_DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=training_args.learning_rate, momentum=training_args.momentum,
                                    weight_decay=training_args.weight_decay)

        # train
        train_data = MSLRDataset(file_path=args.train_file)
        dev_data = MSLRDataset(file_path=args.dev_file)
        train_speedup(train_data, dev_data, model, optimizer, training_args)
        torch.save(model.state_dict(), args.model_file)

        # test
        test_data = MSLRDataset(file_path=args.test_file)
        test_speedup(test_data, model, args)


if __name__ == '__main__':
    main("speedup")
