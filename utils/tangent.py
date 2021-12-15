# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了几个函数来求点到圆与多边形交集的切线。
from utils.math_func import norm
import numpy as np


def check_circle_line_segInter(center: np.ndarray, radius: float, line_p1: np.ndarray, line_p2: np.ndarray) -> bool:
    """
    判断输入的圆和线段是否有交点。

    输入：
        center: 圆心
        radius: 半径
        line_p1: 线段起点坐标
        line_p2: 线段终点坐标

    输出：
        True表示有交点，False表示无交点
    """

    flag1 = norm(center-line_p1) <= radius
    flag2 = norm(center-line_p2) <= radius
    if (flag1 and not(flag2)) or (not(flag1) and flag2):
        return True
    elif(flag1 and flag2):
        return False
    else:
        # 将直线p1p2转换为一般式：Ax+By+C=0
        A = line_p1[1]-line_p2[1]
        B = line_p2[0]-line_p1[0]
        C = line_p1[0]*line_p2[1]-line_p2[0]*line_p1[1]
        # 使用距离公式判断圆心到直线Ax+By+C=0的距离是否大于半径
        dist1_squared = (A*center[0]+B*center[1]+C)**2
        dist2_squared = (A**2+B**2)*radius**2
        return False


def compute_circle_line_seg_inter(center: np.ndarray, radius: float, line_p1: np.ndarray, line_p2: np.ndarray) -> np.ndarray:
    """
    计算输入的圆和线段的交点坐标。

    输入：
        center: 圆心
        radius: 半径
        line_p1: 线段起点坐标
        line_p2: 线段终点坐标

    输出：
        交点坐标
    """
