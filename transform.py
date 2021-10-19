# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: transform.py
@time: 2021/7/10 下午10:00
@desc: 
"""
import re
from typing import List

import pandas as pd
from sklearn.base import TransformerMixin


class StrToRealList(TransformerMixin):
    """
    将值为形为列表的字符串转为真正的列表
    """

    def __init__(self, cols: str or List[str]):
        """
        Args:
            cols: 需要转换的列名
        """
        self.cols = cols

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if isinstance(self.cols, str):
            self.cols = [self.cols]

        for col in self.cols:
            df[col] = df[col].apply(lambda x: str(x).replace('[', '').replace(']', '').split(","))

        return df


class CommonGroupbyAgg(TransformerMixin):
    """
    实现常见的聚合统计，如'mean', 'max', 'min', 'var', 'median', 'count'
    """

    def __init__(self, group_keys: str or List[str], agg_column: str or List[str]):
        """
        Args:
            group_keys: 指定分组, 可以同时以多列为keys
            agg_column: 指定需要聚合的列
        """
        self.group_keys = group_keys
        self.agg_column = agg_column

    def fit_transform(self, df: pd.DataFrame, **fit_params):
        agg_list = {self.agg_column: ['mean', 'max', 'min', 'var', 'median', 'count']}
        df = df.groupby(self.group_keys).agg(agg_list)
        df.columns = [x[0] + '_' + x[1] for x in df.columns]
        df.reset_index(inplace=True)

        return df


def replace_english_char_to_chinese(text: str) -> str:
    """
    Replace the English punctuations in the text into Chinese if the punctuation followers Chinese character;
    Args:
        text: text episode

    Returns: text episode

    """
    english_chinese_punctuation_map: dict = {
        ",": '，',
        '.': '。',
        '?': '？',
        ':': '：',
        ';': '；',
        '!': '！',
        '"': '”',
        "'": "’",
    }
    chars = list(text)
    for i in range(len(chars)):
        if i == 0:
            continue
        pre = chars[i - 1]
        cur = chars[i]
        if re.match("[\u4e00-\u9fa5]", pre):
            if re.match("[\u4e00-\u9fa5]", cur):
                continue
            chars[i] = english_chinese_punctuation_map.get(chars[i])

    r = "[a-zA-Z'!\"#$%&'()*+,-./:;<=>?@★[\\]^_`{|}~]+"
    return re.sub(r, "", "".join(chars))


def values_to_str(values: list) -> str:
    """
    Transform a list of values into string.
    Args:
        values: list of values which need to be transformed

    Returns:
        string that joined by values
    """
    print("Starts change values to string...")
    char_list: list = list()
    for row in values:
        n: int = len(row)
        for i in range(n):
            char_list.append(str(row[i]))
            if i != n - 1:
                char_list.append("\t")
            else:
                char_list.append("\n")
    print("Changing values to string finished.")

    return "".join(char_list)


def join_by_comma(values: list) -> str:
    """
    Join all elements in a list with comma.
    Args:
        values: list of values which need to be transformed

    Returns:
        string that joined by values
    """
    values = [str(value) for value in values]

    return ", ".join(values)
