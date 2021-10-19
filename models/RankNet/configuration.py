# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: configuration.py
@time: 2021/10/12
@desc: 
"""


class RankNetConfig:
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
