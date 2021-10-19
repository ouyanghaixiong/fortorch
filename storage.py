# -*- coding: utf-8 -*-
"""
@author: ouyhaix@icloud.com
@file: storage.py
@time: 2021/3/22 下午4:48
@desc: 
"""
import abc
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from consts import SAVE_DIR, DATA_DIR
from utils import get_logger

logger = get_logger(__name__)


class StorageLevel:
    MEMORY = "memory"
    CSV = "csv"


class Storage(abc.ABC):
    """
    The interface of storage.
    """

    @abc.abstractmethod
    def put(self, key: str, value):
        pass

    @abc.abstractmethod
    def get(self, key: str):
        pass

    @abc.abstractmethod
    def __contains__(self, key):
        pass


class MemoryStorage(Storage):
    """
    A storage that caches data in memory.
    """
    _data_map = dict()

    def put(self, key: str, value: Any) -> None:
        """
        Put the key value pair into the data map.
        Args:
            key: the key of data
            value: the value of data

        Returns:
            None
        """
        self._data_map[key] = value
        logger.info("put {key}:{value}".format(key=key, value=value.__class__))

    def get(self, key: str) -> Any:
        """
        Get the value from the data map by the key.
        Args:
            key: the key of data

        Returns:
            the value of data
        """
        if key not in self._data_map:
            raise Exception("Unrecognized key: {key}".format(key=key))

        value: Any = self._data_map[key]
        logger.info("got {key}:{value}".format(key=key, value=value.__class__))

        return value

    def __contains__(self, key):
        return key in self._data_map


class CsvStorage(Storage):
    """
    A storage that saves data as csv format.
    """

    def __init__(self, save_dir: str = DATA_DIR, sep: str = ",", index: bool = False,
                 header=0):
        self.save_dir = save_dir
        self.sep = sep
        self.index = index
        self.header = header
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def put(self, key: str, value) -> None:
        if not isinstance(value, pd.DataFrame):
            raise Exception("Unsupported type of value: {type}".format(type=value.__class__))

        file_path = os.path.join(self.save_dir, key + ".csv")
        value.to_csv(file_path, columns=value.columns, sep=self.sep, index=self.index)
        logger.info("put {key}:{value}".format(key=key, value=value.__class__))

    def get(self, key: str) -> pd.DataFrame:
        file_path: str = os.path.join(self.save_dir, key + ".csv")
        if not os.path.exists(file_path):
            raise Exception("Unrecognized key: {key}".format(key=key))

        value = pd.read_csv(file_path, sep=self.sep, header=self.header)
        logger.info("got {key}:{value}".format(key=key, value=value.__class__))

        return value

    def __contains__(self, key):
        file_path: str = os.path.join(self.save_dir, key + ".csv")
        return os.path.exists(file_path)


class FigureStorage(Storage):
    """
    A storage that saves matplotlib.pylot.Figure.
    """

    def __init__(self, save_dir: str = SAVE_DIR, fig_format: str = "png"):
        self.save_dir = save_dir
        self.fig_format = fig_format
        if not os.path.exists(self.save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def put(self, key: str, value: Figure):
        fig_path: str = os.path.join(self.save_dir, key) + "." + self.fig_format
        value.savefig(fig_path)

    def get(self, key: str):
        fig_path: str = os.path.join(self.save_dir, key) + "." + self.fig_format
        if not os.path.exists(fig_path):
            raise Exception("Unrecognized key: {key}".format(key=key))

        fig: Figure = plt.imread(fig_path, format=self.fig_format)

        return fig

    def __contains__(self, key):
        fig_path: str = os.path.join(self.save_dir, key) + "." + self.fig_format
        return os.path.exists(fig_path)
