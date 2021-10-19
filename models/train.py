# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: train.py
@time: 2021/9/27
@desc: 
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from utils import get_logger

logger = get_logger(__name__)


class TrainingArgument:
    def __init__(self, num_epochs: int, early_stopping_rounds: int, learning_rate: float, momentum: float,
                 weight_decay: float):
        self.num_epochs = num_epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay


class Trainer:
    def __init__(self, model, args, criterion=None, optimizer=None, scheduler=None):
        self.model = model
        self.args = args

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters())
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        # 记录最佳验证集loss
        self.best_loss = float('inf')
        # 记录已经训练的batch总数
        self.total_batch = 0
        # 记录上次验证集loss下降的batch数
        self.last_improved_batch = 0

        # Tensorboard
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        self.train_writer = SummaryWriter(log_dir="./" + current_time + "/train")
        self.eval_writer = SummaryWriter(log_dir="./" + current_time + "/eval")

    def train(self, dataset):
        # 数据
        train_dataset = dataset(mode="train")
        train = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        dev_dataset = dataset(mode="dev")
        dev = DataLoader(dev_dataset, batch_size=len(dev_dataset))
        test_dataset = dataset(mode="test")
        test = DataLoader(test_dataset, batch_size=len(test_dataset))
        logger.info(f"train_batches: {len(train)} | dev_batches: {len(dev)} | test_batches: {len(test)}")

        early_stopping = False

        self.model.init_parameters()
        self.model.train()
        for epoch in range(self.args.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, self.args.num_epochs))
            for batch, (x, y) in enumerate(train):
                if batch == 0:
                    logger.info(f"Batch: {batch} | x: {x.size()} | y: {y.size()}")
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=5.0)
                self.optimizer.step()
                # 学习率衰减
                self.scheduler.step()
                self.total_batch += 1

                # eval
                if (batch + 1) % 100 == 0:
                    for name, params in self.model.named_parameters():
                        logger.debug(
                            f"name: {name} | requires_grad: {params.requires_grad} | mean_grad: {torch.mean(params.grad)}")
                    self.model.eval()
                    actual = y.data.cpu()
                    prediction = torch.max(y_hat.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(actual, prediction)
                    dev_acc, dev_loss = self.evaluate(dev)
                    if self._early_stopping(batch, dev_loss):
                        logger.info("No optimization for a long time, auto-stopping...")
                        early_stopping = True
                        break
                    msg = 'Batch: {0:>6} | Train Loss: {1:>8.4} | Train Acc: {2:>6.2%} | Val Loss: {3:>8.4} || Val Acc: {4:>6.2%} | {5}'
                    logger.info(msg.format(batch + 1, loss.item(), train_acc, dev_loss, dev_acc,
                                           "*" if self._improved(dev_loss) else "-"))
                    self.train_writer.add_scalar("Loss", loss.item(), self.total_batch)
                    self.eval_writer.add_scalar("Loss", dev_loss, self.total_batch)
                    self.train_writer.add_scalar("Accuracy", train_acc, self.total_batch)
                    self.eval_writer.add_scalar("Accuracy", dev_acc, self.total_batch)
                    self.model.train()

            if early_stopping:
                break

        self.train_writer.close()
        self.eval_writer.close()
        self.test(test)

    def evaluate(self, data: DataLoader, test: bool = False):
        logger.info("********** Evaluation **********")
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for x, y in data:
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss_total += loss
                true = y.data.cpu().numpy()
                prediction = torch.max(y_hat.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, true)
                predict_all = np.append(predict_all, prediction)

        acc = metrics.accuracy_score(labels_all, predict_all)

        if test:
            logger.info("---------- Testing ----------")
            classes = pd.read_csv(self.args.classes_file, header=None).values.reshape(-1).tolist()
            report = metrics.classification_report(labels_all, predict_all, target_names=classes)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data), report, confusion

        return acc, loss_total / len(data)

    def test(self, data: DataLoader):
        self.model.load_state_dict(torch.load(self.args.save_file))
        self.model.eval()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(data, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        logger.info(msg.format(test_loss, test_acc))
        logger.info("Precision, Recall and F1-Score...")
        print(test_report)
        logger.info("Confusion Matrix...")
        print(test_confusion)

    def _improved(self, loss: float):
        return loss < self.best_loss

    def _early_stopping(self, batch: int, loss: float) -> bool:
        if self._improved(loss):
            self.best_loss = loss
            torch.save(self.model.state_dict(), self.args.save_file)
            self.last_improved_batch = batch
            return False

        return self.total_batch - self.last_improved_batch > self.args.early_stopping_rounds
