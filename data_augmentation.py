# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: data_augmentation.py
@time: 2021/5/19
@desc: 
"""
import hashlib
import random

import numpy as np
import pandas as pd
import requests

from config import TRANSLATION_APPID, TRANSLATION_SECRET_KEY, TRANSLATION_URL


class Translation:
    @staticmethod
    def translate(text, from_language: str, to_language: str):
        salt = random.randint(32768, 65536)
        sign = TRANSLATION_APPID + text + str(salt) + TRANSLATION_SECRET_KEY
        sign = hashlib.md5(sign.encode()).hexdigest()

        url = TRANSLATION_URL
        params = {
            "appid": TRANSLATION_APPID,
            "q": text,
            "from": from_language,
            "to": to_language,
            "salt": salt,
            "sign": sign,
        }
        r = requests.get(url, params)
        data = r.json()
        src = data["trans_result"][0]["src"]
        dst = data["trans_result"][0]["dst"]

        return {"src": src, "dst": dst}

    def generate(self, src_file: str, text_col: str, dst_file: str):
        src_texts: np.ndarray = pd.read_csv(src_file)[text_col].values
        with open(dst_file, "a") as f:
            for src_text in src_texts:
                tmp_text = self.translate(src_text, from_language="zh", to_language="en")["dst"]
                dst_text = self.translate(tmp_text, from_language="en", to_language="zh")["dst"]
                f.write(dst_text + "\n")
