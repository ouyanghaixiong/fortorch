# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@forchange.tech
@file: ${FILE_NAME}
@time: 2021/10/19
@desc: 
"""
import time
from unittest import TestCase

from data_augmentation import Translation


class TestTranslation(TestCase):
    def test_translate(self):
        t = Translation()
        tmp = t.translate("我觉得不错", from_language="zh", to_language="en")["dst"]
        time.sleep(1)
        dst = t.translate(tmp, from_language="en", to_language="zh")["dst"]
        print(dst)
