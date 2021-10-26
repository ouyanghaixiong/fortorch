# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: run_bert.py
@time: 2021/10/4
@desc: 
"""
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn.functional as F

import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import check_sanity, set_seed, get_logger
from consts import SEED
from datasets.thucnews import TRAIN_FILE, DEV_FILE, TEST_FILE, CLASSES_FILE, VOCAB_FILE, EMBEDDING_FILE

logger = get_logger(__file__)


def parse_args():
    parser = ArgumentParser()
    # 任务参数
    parser.add_argument("--task_name", type=str, default="", help="任务名称")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="预训练模型")
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE, help="训练集数据文件")
    parser.add_argument("--dev_file", type=str, default=DEV_FILE, help="开发集数据文件")
    parser.add_argument("--test_file", type=str, default=TEST_FILE, help="测试集数据文件")
    parser.add_argument("--classes_file", type=str, default=CLASSES_FILE, help="类别名称数据文件")
    parser.add_argument("--vocab_file", type=str, default=VOCAB_FILE, help="词典文件")
    parser.add_argument("--pretrained_embedding_file", type=str, default=EMBEDDING_FILE, help="预训练的词向量文件")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")

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

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    bert_model = BertModel.from_pretrained(args.pretrained_model)
