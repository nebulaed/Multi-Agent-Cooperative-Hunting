# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了一些用于数学计算的函数。
# 注：@jit(nopython=True)用于加速numpy数学计算

import numpy as np
from numba import jit

PI = np.pi

@jit(nopython=True)
def correct(angle: float) -> float:
    """
    将角度调整到[0,2π)的范围内

    输入：
        angle: 角度(弧度制)

    输出：
        angle: 角度∈[0,2π)
    """
    if angle < 0:
        angle += 2*PI
    elif angle >= 2*PI:
        angle -= 2*PI
    else:
        pass
    return angle


@jit(nopython=True)
def peri_arctan(c):
    """
    将向量转换到[0,2π)间的角度，把向量的起点定为圆心，终点定在圆上，则其方向角度∈[0,2π)计算如该函数。

    输入：
        c: 长度为2的list或np.ndarray，c[0]为向量的x分量，c[1]为向量的y分量
    输出：
        angle: 向量c的方向角度∈[0,2π)
    """
    # 圆的右上半周(0,π/2)
    if c[0] > 0 and c[1] >= 0:
        angle = np.arctan(c[1]/c[0])
    # 圆的右下半周(3π/2,2π)
    elif c[0] > 0 and c[1] < 0:
        angle = np.arctan(c[1]/c[0])+2*PI
    # 圆的左半周(-π/2,π/2)=>(π/2,3π/2)
    elif c[0] < 0:
        angle = np.arctan(c[1]/c[0])+PI
    # 圆上半周奇异点赋值
    elif c[0] == 0 and c[1] > 0:
        angle = PI/2
    # 圆下半周奇异点赋值
    elif c[0] == 0 and c[1] < 0:
        angle = 3*PI/2
    else:
        angle = 0
    return angle


@jit(nopython=True)
def inc_angle(angle1, angle2):
    """
    计算两个方向角之间的夹角

    输入：
        angle1∈[0,2π)
        angle2∈[0,2π)
    输出∈[0,π]
    """
    angle1, angle2 = correct(angle1), correct(angle2)
    if abs(angle1-angle2) > PI:
        return 2*PI-abs(angle1-angle2)
    else:
        return abs(angle1-angle2)


def intervals_merge(ang_intervals: list, ang_intervals_index: list, ang_intervals_dists: list):
    """
    将有交叠的角度区间进行合并，所有输入角度区间的起点和终点都∈[0,2π)

    输入：
        ang_intervals: 角度区间列表
        ang_intervals_index: 角度区间的索引列表
        ang_intervals_dists: 角度区间的距离列表

    输出：
        merged_ang_interval: 合并后的角度区间列表
        merged_ang_interval_index: 合并后的角度区间索引列表
        merged_ang_interval_dists: 合并后的角度区间距离列表
    """
    # 若右边界小于左边界, 则令右边界+2π
    for i in range(len(ang_intervals)):
        if ang_intervals[i][0] > ang_intervals[i][1]:
            ang_intervals[i][1] += 2*PI
    # if mark==1: print(ang_intervals)
    if len(ang_intervals) >= 2:
        duplication = True
        while duplication:
            for i in range(len(ang_intervals)):
                for j in range(i+1, len(ang_intervals)):
                    if ang_intervals[i][0] != -1 and ang_intervals[i][1] != -1 and ang_intervals[j][0] != -1 and ang_intervals[j][1] != -1:
                        if max(ang_intervals[i][0], ang_intervals[j][0]) <= min(ang_intervals[i][1], ang_intervals[j][1]):
                            # 将两个交叠区间进行合并
                            ang_intervals[i][0] = min([ang_intervals[i][0], ang_intervals[i][1], ang_intervals[j][0], ang_intervals[j][1]])
                            ang_intervals[i][1] = max([ang_intervals[i][0], ang_intervals[i][1], ang_intervals[j][0], ang_intervals[j][1]])
                            # 将被合并的区间标记为(-1,-1)
                            ang_intervals[j][0] = -1
                            ang_intervals[j][1] = -1
                        elif ang_intervals[i][1] > 2*PI and ang_intervals[j][1] < 2*PI:
                            # 将两个交叠区间进行合并
                            if ang_intervals[j][0] < ang_intervals[i][1]-2*PI:
                                ang_intervals[i][0] = ang_intervals[i][0]
                                if ang_intervals[j][1]+2*PI > ang_intervals[i][1]:
                                    ang_intervals[i][1] = ang_intervals[j][1]+2*PI
                                # 将被合并的区间标记为(-1,-1)
                                ang_intervals[j][0] = -1
                                ang_intervals[j][1] = -1
                        elif ang_intervals[j][1] > 2*PI and ang_intervals[i][1] < 2*PI:
                            # 将两个交叠区间进行合并
                            if ang_intervals[i][0] < ang_intervals[j][1]-2*PI:
                                ang_intervals[i][0] = ang_intervals[j][0]
                                if ang_intervals[i][1]+2*PI > ang_intervals[j][1]:
                                    ang_intervals[i][1] = ang_intervals[i][1]+2*PI
                                else:
                                    ang_intervals[i][1] = ang_intervals[j][1]
                                # 将被合并的区间标记为(-1,-1)
                                ang_intervals[j][0] = -1
                                ang_intervals[j][1] = -1
                        if ang_intervals[i][1]-ang_intervals[i][0]>2*PI:
                            ang_intervals[i][0] = 0
                            ang_intervals[i][1] = 2*PI
                            for k in range(len(ang_intervals)):
                                if k != i:
                                    ang_intervals[k][0] = -1
                                    ang_intervals[k][1] = -1
                            break
                # 若内层j循环正常执行完毕未被break，则外层i循环继续
                else:
                    continue
                # 否则外层i循环也break
                break
            new_ang_interval, new_ang_interval_index, new_ang_interval_dists = [], [], []
            for i in range(len(ang_intervals)):
                if ang_intervals[i][0] != -1 or ang_intervals[i][1] != -1:
                    new_ang_interval.append(ang_intervals[i])
                    # 因为dangerous_ranges[i]的距离一定比dangerous_ranges[j]更小, 故保留i的index
                    new_ang_interval_index.append(ang_intervals_index[i])
                    new_ang_interval_dists.append(ang_intervals_dists[i])
            ang_intervals, ang_intervals_index, ang_intervals_dists = new_ang_interval, new_ang_interval_index, new_ang_interval_dists
            duplication = False
            for i in range(len(ang_intervals)):
                for j in range(i+1, len(ang_intervals)):
                    if ang_intervals[i][0] != -1 and ang_intervals[i][1] != -1 and ang_intervals[j][0] != -1 and ang_intervals[j][1] != -1:
                        if max(ang_intervals[i][0], ang_intervals[j][0]) <= min(ang_intervals[i][1], ang_intervals[j][1]):
                            duplication = True
                        elif ang_intervals[i][1] > 2*PI and ang_intervals[j][1] < 2*PI:
                            # 将两个交叠区间进行合并
                            if ang_intervals[j][0] < ang_intervals[i][1]-2*PI:
                                duplication = True
                        elif ang_intervals[j][1] > 2*PI and ang_intervals[i][1] < 2*PI:
                            if ang_intervals[i][0] < ang_intervals[j][1]-2*PI:
                                duplication = True
        merged_ang_interval, merged_ang_interval_index, merged_ang_interval_dists = new_ang_interval, new_ang_interval_index, new_ang_interval_dists
    else:
        merged_ang_interval, merged_ang_interval_index, merged_ang_interval_dists = ang_intervals, ang_intervals_index, ang_intervals_dists
    return merged_ang_interval, merged_ang_interval_index, merged_ang_interval_dists


@jit(nopython=True)
def cos(x):
    return np.cos(x)


@jit(nopython=True)
def sin(x):
    return np.sin(x)


@jit(nopython=True)
def sqrt(x):
    return np.sqrt(x)


@jit(nopython=True)
def arcsin(x):
    return np.arcsin(x)


@jit(nopython=True)
def norm(x):
    return np.linalg.norm(x)


@jit(nopython=True)
def exp(x):
    return np.exp(x)
