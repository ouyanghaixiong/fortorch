# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: run_fm.py
@time: 2021/9/15
@desc: 
"""
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from consts import DEFAULT_SEED
from models.FM.factorization_machine import FM
from datasets.criteo import CriteoDataset
from utils import get_logger

logger = get_logger(__file__)


def train():
    num_epoch = 10
    batch_size = 512
    num_numeric_features = 13
    num_cat_features = 26
    embedding_dim = 128
    learning_rate = 0.01
    weight_decay = 0.01
    n_split = 5

    # data
    dataset = CriteoDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = FM(num_numeric_features=num_numeric_features, num_cat_features=num_cat_features,
               nunique_cat_features=dataset.nunique_cat_features, embedding_dim=embedding_dim)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # start training with cross validation
    splitter = KFold(n_splits=n_split, shuffle=True, random_state=DEFAULT_SEED)
    # for each epoch
    for epoch in range(num_epoch):
        # for each fold
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(dataset)):
            train_data = DataLoader(dataset=dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
            valid_data = DataLoader(dataset=dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
            # for each step, take a data batch
            for step, ((train_y, train_numeric, train_cat), (valid_y, valid_numeric, valid_cat)) in enumerate(zip(train_data, valid_data)):
                train_y_hat = model(train_numeric, train_cat)
                loss = criterion(train_y_hat, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_auc = roc_auc_score(train_y.numpy(), train_y_hat.numpy())
                    valid_y_hat = model(valid_numeric, valid_cat)
                    valid_auc = roc_auc_score(valid_y.numpy(), valid_y_hat.numpy())
                    logger.info(f"step {step} loss {loss} train: auc {train_auc} valid: auc {valid_auc}")


if __name__ == '__main__':
    train()
