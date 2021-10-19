# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: run_bilstmattention.py
@time: 2021/10/4
@desc: 
"""
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn

from utils import check_sanity, set_seed
from consts import DEFAULT_SEED
from datasets.thucnews import EMBEDDING_FILE, TRAIN_FILE, DEV_FILE, TEST_FILE, CLASSES_FILE, VOCAB_FILE, \
    THUCNewsDataset
from models.train import Trainer


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w_q = nn.Linear(2 * args.hidden_size, args.attention_size)
        self.w_k = nn.Linear(2 * args.hidden_size, args.attention_size)
        self.w_v = nn.Linear(2 * args.hidden_size, args.attention_size)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, sequence_len, embedding_dim]

        Returns:
            out: shape [batch_size, sequence_len, embedding_dim]
        """
        # shape [batch_size, sequence_len, attention_size]
        q: torch.Tensor = self.w_q(x)

        # shape [batch_size, sequence_len, attention_size]
        k: torch.Tensor = self.w_k(x)

        # shape [batch_size, sequence_len, attention_size]
        v: torch.Tensor = self.w_v(x)

        # shape [batch_size, sequence_len ,sequence_len]
        dk = k.shape[2]
        q = q / np.sqrt(dk)
        attention_score = torch.softmax(torch.bmm(q, torch.transpose(k, 2, 1)), dim=-1)

        # shape [batch_size, sequence_len, attention_size]
        out = torch.bmm(attention_score, v)

        return out

    def init_parameters(self):
        for name, w in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(w, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(w)


class BiLSTMAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.pretrained_embedding_file is not None:
            pretrained_embedding = torch.Tensor(np.load(EMBEDDING_FILE)["embeddings"].astype('float32'))
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(args.n_vocab, args.embedding_dim, padding_idx=args.n_vocab - 1)

        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_size, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.self_attention = SelfAttention(args)
        self.fc = nn.Linear(args.attention_size, args.num_classes)
        self.bn = nn.BatchNorm1d(num_features=args.num_classes)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, sequence_len]

        Returns:
            out: shape [batch_size, num_classes]
        """
        # shape [batch_size, sequence_len, embedding_dim]
        x_embeddings = self.embedding(x)

        # shape [batch_size, sequence_len, 2 * hidden_size]
        out, _ = self.lstm(x_embeddings)

        # shape [batch_size, sequence_len, attention_size]
        out = self.self_attention(out)

        # shape [batch_size, attention_size]
        out = torch.mean(out, dim=1)

        # shape [batch_size, num_classes]
        out = self.fc(out)

        return self.bn(out)

    def init_parameters(self):
        self.self_attention.init_parameters()
        for name, w in self.named_parameters():
            if "lstm" in name:
                if 'bias' in name:
                    nn.init.constant_(w, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(w)
            elif "fc" in name:
                if "bias" in name:
                    nn.init.constant_(w, 0)
                elif "weight" in name:
                    nn.init.xavier_normal_(w)


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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")

    # 模型参数
    parser.add_argument("--n_vocab", type=int, default=10000, help="词表大小")
    parser.add_argument("--embedding_dim", type=int, default=300, help="字向量维度")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层大小")
    parser.add_argument("--attention_size", type=int, default=256, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM层数量")
    parser.add_argument("--dropout", type=float, default=0.1, help="随机失活")
    parser.add_argument("--num_classes", type=int, default=10, help="类别数")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10, help="epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--pad_size", type=int, default=50, help="每句话处理成的长度(短填长切)")
    parser.add_argument("--early_stopping_rounds", type=int, default=1000, help="早停")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="学习率")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--save_file", type=str, default=f"./model.ckpt", help="保存模型文件")

    args = parser.parse_args()
    check_sanity(args)

    return args


def main():
    args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    model = BiLSTMAttention(args)
    trainer = Trainer(model, args)
    trainer.train(THUCNewsDataset)


if __name__ == '__main__':
    main()
