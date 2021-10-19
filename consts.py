# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: consts.py
@time: 2021/4/25 上午11:04
@desc: 
"""
import logging
import os

import torch

# the absolute root dir of the project
ROOT_DIR: str = os.path.dirname(__file__)

# the absolute data dir
DATA_DIR = os.path.join(ROOT_DIR, "data")

# the absolute save dir
SAVE_DIR = os.path.join(ROOT_DIR, "save")

# the device to use
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random seed
DEFAULT_SEED = 2021

# default data type of torch.Tensor
DEFAULT_DTYPE = torch.float32

# logging.level
LOGGING_LEVEL = logging.INFO
