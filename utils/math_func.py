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
    思路：先检查列表中的角度区间，若有右端点大于左端点，将其拆分为[右端点,2π)和[0,左端点]，然后对列表中的区间按照左端点升序排序，同时index和dists也按照前面排好的顺序排序，然后按顺序依次考虑每个区间：1.假如是第一个区间或当前区间的左端点在结果列表中最后一个区间的右端点之后，那么它们不重合，直接把当前区间放入结果列表中，2.否则重合，更新结果列表中最后一个区间的右端点，变为(结果列表中最后一个区间的右端点,当前区间的右端点)中的最大值。最后将[右端点,2π)和[0,左端点]合并为[左端点，右端点+2π]

    输入：
        ang_intervals: 角度区间列表
        ang_intervals_index: 角度区间的索引列表
        ang_intervals_dists: 角度区间的距离列表

    输出：
        merged_ang_interval: 合并后的角度区间列表
        merged_ang_interval_index: 合并后的角度区间索引列表
        merged_ang_interval_dists: 合并后的角度区间距离列表
    """
    # 若右边界小于左边界, 则将该区间拆开为[右端点,2π)和[0,左端点]
    for i in range(len(ang_intervals)):
        if ang_intervals[i][0] > ang_intervals[i][1]:
            ang_intervals.append([0, ang_intervals[i][1]])
            ang_intervals_index.append(ang_intervals_index[i])
            ang_intervals_dists.append(ang_intervals_dists[i])
            ang_intervals[i][1] = 2*PI
    # 若区间数小于2，不用合并交叠区间
    if len(ang_intervals) < 2:
        return ang_intervals, ang_intervals_index, ang_intervals_dists
    # 三个列表都按照角度区间的左端点进行排序
    ang_intervals_index = [x for _,x in sorted(zip(ang_intervals, ang_intervals_index))]
    ang_intervals_dists = [x for _,x in sorted(zip(ang_intervals, ang_intervals_dists))]
    ang_intervals.sort(key=lambda x: x[0])
    # 初始化结果列表
    merged_ang_intervals, merged_ang_interval_indexs, merged_ang_interval_dists = [],[],[]
    # 合并过程
    for i in range(len(ang_intervals)):
        if not merged_ang_intervals or merged_ang_intervals[-1][1] < ang_intervals[i][0]:
            merged_ang_intervals.append(ang_intervals[i])
            merged_ang_interval_indexs.append(ang_intervals_index[i])
            merged_ang_interval_dists.append(ang_intervals_dists[i])
        else:
            merged_ang_intervals[-1][1] = max(merged_ang_intervals[-1][1], ang_intervals[i][1])
            merged_ang_interval_indexs[-1] = merged_ang_interval_indexs[-1] if merged_ang_interval_dists[-1] < ang_intervals_dists[i] else ang_intervals_index[i]
            merged_ang_interval_dists[-1] = min(merged_ang_interval_dists[-1],ang_intervals_dists[i])
    # 将[右端点,2π)和[0,左端点]合并为[左端点，右端点+2π]
    for i in range(len(merged_ang_intervals)):
        flag = False
        if merged_ang_intervals[i][0] == 0:
            for j in range(len(merged_ang_intervals)):
                if j != i and merged_ang_intervals[j][1] == 2*PI:
                    merged_ang_intervals[i][0] = merged_ang_intervals[j][0]
                    merged_ang_intervals[i][1] += 2*PI
                    merged_ang_interval_indexs[i] = merged_ang_interval_indexs[i] if merged_ang_interval_dists[i] < merged_ang_interval_dists[j] else merged_ang_interval_indexs[j]
                    merged_ang_interval_dists[i] = min(merged_ang_interval_dists[i], merged_ang_interval_dists[j])
                    merged_ang_intervals.pop(j)
                    merged_ang_interval_indexs.pop(j)
                    merged_ang_interval_dists.pop(j)
                    flag = True
                    break
        if(flag): break
    return merged_ang_intervals, merged_ang_interval_indexs, merged_ang_interval_dists


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
