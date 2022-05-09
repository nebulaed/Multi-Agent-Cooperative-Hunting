# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了一些用于数学计算的函数。
# 注：@jit(nopython=True)用于加速numpy数学计算

import numpy as np
from numba import jit
from typing import List
from utils.params import PI


@jit(nopython=True)
def correct(angle: float) -> float:
    """
    将角度调整到[0,2π)的范围内

    输入：
        @param angle: 角度(单位为rad)

    输出：
        @return angle: 角度∈[0,2π)
    """
    if angle < 0:
        angle += 2*PI
    elif angle >= 2*PI:
        angle -= 2*PI
    else:
        pass
    return angle


@jit(nopython=True)
def peri_arctan(c: np.ndarray) -> float:
    """
    将向量转换到[0,2π)间的角度，把向量的起点定为圆心，终点定在圆上，则其方向角度∈[0,2π)计算如该函数。

    输入：
        @param c: 长度为2的list或np.ndarray，c[0]为向量的x分量，c[1]为向量的y分量
    输出：
        @return angle: 向量c的方向角度∈[0,2π)
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
def inc_angle(angle1: float, angle2: float) -> float:
    """
    计算两个方向角之间的夹角

    输入：
        @param angle1: ∈[0,2π)
        @param angle2: ∈[0,2π)
    输出
        @return abs(angle1-angle2): ∈[0,π]
    """
    angle1, angle2 = correct(angle1), correct(angle2)
    if abs(angle1-angle2) > PI:
        return 2*PI-abs(angle1-angle2)
    else:
        return abs(angle1-angle2)


def intervals_merge(ang_intervals: List, ang_intervals_index: List[int], ang_intervals_dists: List[float]):
    """
    将有交叠的角度区间进行合并，所有输入角度区间的起点和终点都∈[0,2π)
    思路：先检查列表中的角度区间，若有右端点大于左端点，将其拆分为[右端点,2π)和[0,左端点]，然后对列表中的区间按照左端点升序排序，同时index和dists也按照前面排好的顺序排序，然后按顺序依次考虑每个区间：1.假如是第一个区间或当前区间的左端点在结果列表中最后一个区间的右端点之后，那么它们不重合，直接把当前区间放入结果列表中，2.否则重合，更新结果列表中最后一个区间的右端点，变为(结果列表中最后一个区间的右端点,当前区间的右端点)中的最大值。最后将[右端点,2π)和[0,左端点]合并为[左端点，右端点+2π]

    输入：
        @param ang_intervals: 角度区间列表
        @param ang_intervals_index: 角度区间的索引列表
        @param ang_intervals_dists: 角度区间的距离列表

    输出：
        @return merged_ang_interval: 合并后的角度区间列表
        @return merged_ang_interval_index: 合并后的角度区间索引列表
        @return merged_ang_interval_dists: 合并后的角度区间距离列表
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


@jit(nopython = True)
def rotate_update(t: np.ndarray, samples_num: int, r_num: np.ndarray, R: float, vertex_x: np.ndarray, vertex_y: np.ndarray, pos_x: float, pos_y: float, pose_order: np.ndarray, variation: float) -> np.ndarray:
    """
    更新旋转后的不规则障碍物的构成点

    输入：
        @param t: 圆内顶点相对于圆心的方向数组
        @param samples_num: 不规则障碍物形状的顶点个数
        @param r_num: 0.07到1间的随机数序列，在障碍物初始化时已确定
        @param R: 不规则障碍物顶点生成圆的半径
        @param vertex_x: 不规则障碍物顶点的x坐标数组
        @param vertex_y: 不规则障碍物顶点的y坐标数组
        @param pos_x: 不规则障碍物圆心的x坐标数组
        @param pos_y: 不规则障碍物圆心的y坐标数组
        @param pose_order: 顶点索引顺序list: self.pose_order
        @param variation: 不规则障碍物旋转角度，单位为rad

    输出：
        @return elements: 不规则障碍物形状构成点的坐标数组
    """
    elements = np.zeros((samples_num * 5, 2))
    count = 0
    # 各顶点围绕生成点旋转
    t += variation
    for item in t:
        item = correct(item)
    unit_x = cos(t)
    unit_y = sin(t)
    # 计算出新顶点i为(vertex_x[i],vertex_y[i])
    for i in range(samples_num):
        polar_r = sqrt(r_num[i])*(R)
        vertex_x[i] = unit_x[i]*polar_r+pos_x
        vertex_y[i] = unit_y[i]*polar_r+pos_y
    # 按照当前顶点在[0,2π)的角度顺序进行连接，并在每条边上取5个点作为该边的构成点
    for i in range(samples_num-1):
        side_x = np.linspace(vertex_x[pose_order[i]], vertex_x[pose_order[i+1]], 5)
        side_y = np.linspace(vertex_y[pose_order[i]], vertex_y[pose_order[i+1]], 5)
        for j in range(5):
            elements[count] = np.array([side_x[j], side_y[j]])
            count += 1
    side_x = np.linspace(vertex_x[pose_order[-1]], vertex_x[pose_order[0]], 5)
    side_y = np.linspace(vertex_y[pose_order[-1]], vertex_y[pose_order[0]], 5)
    for i in range(5):
        elements[count] = np.array([side_x[i], side_y[i]])
        count += 1
    return elements


@jit(nopython = True)
def expand_angle(psi: float, Delta: float, shortest_dist: float, D_DANGER_W: float):
    """
    根据机器人与障碍物的最短距离shortest_dist以及紧急避障距离D_DANGER_W对危险角度区间进行扩展

    输入：
        @param psi: 原危险角度区间角平分线的角度(单位为rad)
        @param Delta: 原危险角度区间长度的一半
        @param shortest_dist: 机器人与障碍物的最短距离(单位为m)
        @param D_DANGER_W: 机器人的紧急避障距离(单位为m)

    输出：
        @return dangerous_range: 扩展后的危险角度区间
        @return flag: 目前机器人与障碍物的最短距离shortest_dist是否已小于紧急避障距离D_DANGER_W，是则改用arctan，避免arcsin出现计算错误
    """
    flag = True
    if (D_DANGER_W > shortest_dist):
        flag = False
        expanded = arctan(D_DANGER_W/shortest_dist)
    else:
        # 扩展的角度值
        expanded = arcsin(D_DANGER_W/shortest_dist)
    dangerous_range = [correct(psi-Delta-expanded), correct(psi+Delta+expanded)]
    return dangerous_range, flag


@jit(nopython=True)
def cos(x: float) -> float:
    return np.cos(x)


@jit(nopython=True)
def sin(x: float) -> float:
    return np.sin(x)


@jit(nopython=True)
def sqrt(x: float) -> float:
    return np.sqrt(x)


@jit(nopython=True)
def arctan(x: float) -> float:
    return np.arctan(x)


@jit(nopython=True)
def arcsin(x: float) -> float:
    return np.arcsin(x)


@jit(nopython=True)
def norm(x: np.ndarray) -> float:
    return np.linalg.norm(x)


@jit(nopython=True)
def exp(x: float) -> float:
    return np.exp(x)
