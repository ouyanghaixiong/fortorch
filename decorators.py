# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: decorators.py
@time: 2021/3/8 下午5:25
@desc: 
@Software: PyCharm
"""
import functools
import os

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from consts import DATA_DIR
from storage import MemoryStorage, StorageLevel, CsvStorage


def execute_periodically(trigger: str = "cron", coalesce: bool = True, max_instances: int = 1, **trigger_args):
    """
    A decorator that enable the func to be executed periodically.
    Args:
        trigger:
        coalesce:
        max_instances:
        **trigger_args:

    Examples:
        @execute_periodically(year="*", month="*", day="*", hour="0", minute="0", second="0")

    Returns:
        decorator
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            scheduler = BlockingScheduler()
            scheduler.add_job(func, trigger, args=args, kwargs=kwargs, coalesce=coalesce,
                              max_instances=max_instances, **trigger_args)
            scheduler.start()

        return wrapper

    return decorator


def save_as_csv(path: str, sep: str = ",", index: bool = False):
    """
    A decorator that save pd.DataFrame as csv

    Args:
        path: file path to save
        sep: separator in csv file
        index: whether to save the index of the DataFrame or not

    Returns:
        decorator
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df: pd.DataFrame = func(*args, **kwargs)
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
            df.to_csv(path, sep=sep, columns=df.columns, index=index)
            return df

        return wrapper

    return decorator


def persist(level=StorageLevel.MEMORY, **decorator_args):
    """
    A decorator that persist data returned by calling func.
    Args:
        level: StorageLevel, i.e. StorageLevel.MEMORY
        **decorator_args: save_dir / sep / index

    Returns:
        decorator
    """
    if level == StorageLevel.CSV:
        save_dir = decorator_args.get("save_dir", DATA_DIR)
        sep = decorator_args.get("sep", ",")
        index = decorator_args.get("index", False)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key: str = func.__name__
            if level == StorageLevel.MEMORY:
                storage = MemoryStorage()
            elif level == StorageLevel.CSV:
                storage = CsvStorage(save_dir=save_dir, sep=sep, index=index)
            else:
                raise Exception("Unsupported storage level: {level}.".format(level=level))

            if key not in storage:
                data = func(*args, **kwargs)
                storage.put(key, data)
                return data

            return storage.get(key)

        return wrapper

    return decorator
