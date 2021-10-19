# -*- coding: utf-8 -*-
"""
@author: ouyanghaixiong@icloud.com
@file: parsers.py
@time: 2021/3/10 上午9:09
@desc: 
@Software: PyCharm
"""


class SQLParser:
    def __init__(self):
        pass

    @staticmethod
    def get_columns_from(sql: str) -> list:
        """
        Parses a dml to get the selected column names.
        :param sql: a dml sql
        :return: the columns names
        """
        raw_str = sql.split("from ")[0].replace("select ", "")
        columns = raw_str.split(",")
        cleaned_columns: list = []
        for column in columns:
            if " as " in column:
                column = column.split(" as ")[-1].strip()
            if "." in column:
                column = column.split(".")[-1].strip()
            column = column.strip("`")
            cleaned_columns.append(column.strip())

        return cleaned_columns
