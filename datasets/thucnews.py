# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: thucnews.py
@time: 2021/9/27
@desc: 
"""

import os
import pickle

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from consts import DATA_DIR, DEVICE

from utils import get_logger

MAX_VOCAB_SIZE = 10000  # 词表长度限制
PAD_SIZE = 32
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

DIR = os.path.join(DATA_DIR, "THUCNews")

RAW_TRAIN_FILE = os.path.join(DIR, "train.txt")
RAW_DEV_FILE = os.path.join(DIR, "dev.txt")
RAW_TEST_FILE = os.path.join(DIR, "test.txt")
CLASSES_FILE = os.path.join(DIR, "class.txt")
PRETRAINED_EMBEDDING_FILE = os.path.join(DIR, "sgns.sogou.char")

TRAIN_FILE = os.path.join(DIR, "train.pkl")
DEV_FILE = os.path.join(DIR, "dev.pkl")
TEST_FILE = os.path.join(DIR, "test.pkl")
VOCAB_FILE = os.path.join(DIR, "vocab.pkl")
EMBEDDING_FILE = os.path.join(DIR, "embedding_SougouNews.npz")

logger = get_logger(__name__)


class THUCNewsDataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        self.mode = mode

        data = self._process()
        self.x = torch.LongTensor([_[0] for _ in data]).to(DEVICE)
        self.y = torch.LongTensor([_[1] for _ in data]).to(DEVICE)

    def _process(self):
        file_map = {
            "train": TRAIN_FILE,
            "dev": DEV_FILE,
            "test": TEST_FILE
        }
        if self.mode not in file_map:
            raise KeyError(f"找不到mode: {self.mode}")

        with open(file_map[self.mode], "rb") as f:
            return pickle.load(f)

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size()[0]


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def load_dataset(path, vocab, tokenizer, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            words_line = []
            token = tokenizer(content)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label)))

    return contents  # [([...], 0), ([...], 1), ...]


def main():
    tokenizer = lambda x: [y for y in x]  # char-level

    # 构建vocab
    vocab = build_vocab(RAW_TRAIN_FILE, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    logger.info(f"Vocab size: {len(vocab)}")
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump(vocab, f)

    # 构建数据集train / dev / test
    files = ((RAW_TRAIN_FILE, TRAIN_FILE), (RAW_DEV_FILE, DEV_FILE), (RAW_TEST_FILE, TEST_FILE))
    for file in files:
        raw_data_file = file[0]
        saved_data_file = file[1]
        data = load_dataset(raw_data_file, vocab, tokenizer)
        with open(saved_data_file, "wb") as f:
            pickle.dump(data, f)

    # 提取预训练词向量
    embeddings = np.random.rand(len(vocab), 300)
    with open(PRETRAINED_EMBEDDING_FILE) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split(" ")
            if line[0] in vocab:
                idx = vocab[line[0]]
                emb = [float(x) for x in line[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
    np.savez_compressed(EMBEDDING_FILE, embeddings=embeddings)


if __name__ == "__main__":
    main()
