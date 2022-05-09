# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了一个用于读取初始化参数存放的yml文件的类。

import yaml


class Params(object):
    """
    从yml文件中读取参数
    """

    def __init__(self, project_file: str):
        """
        输入：
            @param project_file: yml文件路径
        """
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
