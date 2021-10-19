import datetime
import logging
import os
import random

import numpy as np
import requests
import torch

from consts import DEFAULT_SEED, LOGGING_LEVEL


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def set_seed(seed: int = DEFAULT_SEED):
    """
    Avoid randomicity according to setting the random state.
    Args:
        seed: random seed

    Returns:
        None
    """
    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def check_sanity(args) -> None:
    if args.task_name is None and args.train_file is None and args.dev_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    if args.model_file is not None:
        saved_model_dir = os.path.dirname(args.model_file)
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)


def download_from_url(url, save_dir: str) -> None:
    """
    从指定的URL下载文件并保存
    Args:
        url: 下载地址
        save_dir: 保存地址

    Returns: None
    """
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception("wrong status code: {status_code}".format(status_code=req.status_code))
    file_path = os.path.join(save_dir, url.split('/')[-1])
    with open(file_path, "wb") as f:
        f.write(req.content)


def get_logger(name: str, level=LOGGING_LEVEL):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


def get_first_datetime_of_year(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(year=dt.year, month=1, day=1, hour=0, minute=0, second=0)


def get_last_datetime_of_year(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(year=dt.year, month=12, day=31, hour=23, minute=59, second=59)
