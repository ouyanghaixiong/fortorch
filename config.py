# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: config.py
@time: 2021/10/19
@desc: 
"""
import os

TRANSLATION_APPID = os.environ.get("translation_appid")

TRANSLATION_SECRET_KEY = os.environ.get("translation_secret_key")

TRANSLATION_URL = "https://fanyi-api.baidu.com/api/trans/vip/translate"
