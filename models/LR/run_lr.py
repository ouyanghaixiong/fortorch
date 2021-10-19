# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: run_lr.py
@time: 2021/9/16
@desc: 
"""

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

from datasets.criteo import CriteoDataset
from models.LR.logistic import LogisticClassifier
from utils import set_seed, get_logger

logger = get_logger(__name__)


def train():
    num_epoch = 128
    input_size = 39
    learning_rate = 0.001
    batch_size = 1024

    # data
    train_dataset = CriteoDataset(is_training=True, discretize=True)
    test_dataset = CriteoDataset(is_training=False, discretize=True)
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=test_dataset.__len__())

    # model
    model = LogisticClassifier(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(num_epoch):
        model.train()
        for step, (train_y, train_x) in enumerate(train_data):
            optimizer.zero_grad()
            train_y_hat = model(train_x).view(-1)
            loss = criterion(train_y_hat, train_y)
            loss.backward()
            optimizer.step()
            logger.info(f"step: {step} loss: {loss}")

        model.eval()
        with torch.no_grad():
            for test_y, test_x in test_data:
                test_y_hat = model(test_x).view(-1)
                test_loss = log_loss(test_y, test_y_hat)
                test_auc = roc_auc_score(test_y, test_y_hat)
                logger.info(f"test: loss {test_loss} auc {test_auc}")


if __name__ == '__main__':
    set_seed()
    train()
