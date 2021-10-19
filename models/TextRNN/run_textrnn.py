# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: run_textrnn.py
@time: 2021/9/29
@desc: 
"""
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn

from utils import check_sanity, set_seed
from datasets.thucnews import EMBEDDING_FILE, TRAIN_FILE, DEV_FILE, TEST_FILE, CLASSES_FILE, VOCAB_FILE, \
    THUCNewsDataset
from models.train import Trainer


class TextRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.pretrained_embedding_file is not None:
            pretrained_embedding = torch.Tensor(np.load(EMBEDDING_FILE)["embeddings"].astype('float32'))
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(args.n_vocab, args.embedding_dim, padding_idx=args.n_vocab - 1)

        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_size, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.fc = nn.Linear(args.hidden_size * 2, args.num_classes)
        self.bn = nn.BatchNorm1d(num_features=args.num_classes)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, sequence_len, embedding_dim]
        Returns:
        out: shape [batch_size, num_classes]
        """
        # shape [batch_size, sequence_len, embedding_dim]
        out = self.embedding(x)

        # shape [batch_size, sequence_len, hidden_size * 2]
        out, _ = self.lstm(out)

        # shape [batch_size, hidden_size * 2]
        out = out[:, -1, :]

        # shape [batch_size, num_classes]
        out = self.fc(out)

        out = self.bn(out)

        return out

    def init_parameters(self):
        for name, w in self.named_parameters():
            if name.split(".")[0] in ["embedding", "bn"]:
                continue
            if 'bias' in name:
                nn.init.constant_(w, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(w)


def parse_args():
    parser = ArgumentParser()
    # 任务参数
    parser.add_argument("--task_name", type=str, default="", help="任务名称")
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE, help="训练集数据文件")
    parser.add_argument("--dev_file", type=str, default=DEV_FILE, help="开发集数据文件")
    parser.add_argument("--test_file", type=str, default=TEST_FILE, help="测试集数据文件")
    parser.add_argument("--classes_file", type=str, default=CLASSES_FILE, help="类别名称数据文件")
    parser.add_argument("--vocab_file", type=str, default=VOCAB_FILE, help="词典文件")
    parser.add_argument("--pretrained_embedding_file", type=str, default=EMBEDDING_FILE, help="预训练的词向量文件")

    # 模型参数
    parser.add_argument("--n_vocab", type=int, default=10000, help="词表大小")
    parser.add_argument("--embedding_dim", type=int, default=300, help="字向量维度")
    parser.add_argument("--hidden_size", type=int, default=128, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM层数量")
    parser.add_argument("--dropout", type=float, default=0.1, help="随机失活")
    parser.add_argument("--num_classes", type=int, default=10, help="类别数")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--pad_size", type=int, default=32, help="每句话处理成的长度(短填长切)")
    parser.add_argument("--early_stopping_rounds", type=int, default=1000, help="早停")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="学习率")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--save_file", type=str, default=f"./model.ckpt", help="保存模型文件")

    args = parser.parse_args()
    check_sanity(args)

    return args


def main():
    args = parse_args()

    set_seed()

    model = TextRNN(args)
    trainer = Trainer(model, args)
    trainer.train(THUCNewsDataset)


if __name__ == '__main__':
    main()
