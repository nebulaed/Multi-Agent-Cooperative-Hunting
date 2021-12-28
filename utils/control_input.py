# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了一个控制输入计算函数

from utils.math_func import correct
from utils.params import PI, TS


def saturator(ori: float, vel_max: float, ang_vel_max: float, vel_desired: float, theta_desired: float):
    """
    控制输入计算函数，用当前小车方向ori，最大线速度vel_max，最大角速度ang_vel_max，算法给出的期望速度vel_desired，算法给出的期望方向theta_desired，计算出实际的速度vel和角速度ang_vel

    输入：
        ori: 小车当前车头方向∈[0,2π)
        vel_max: 小车最大线速度(单位为m/s)
        ang_vel_max: 小车最大角速度(单位为rad/s)
        vel_desired: 算法给出的期望速度(单位为m/s)
        theta_desired: 算法给出的期望方向∈[0,2π)

    输出：
        vel: 小车实际下个step的速度(单位为m/s)
        ang_vel: 小车实际下个step的角速度(单位为rad/s)
    """

    # 速率的饱和机制
    vel_desired = min(vel_desired, vel_max)
    # 期望方向角和当前方向角的差
    ori_dif = correct(ori-theta_desired)
    # 若期望方向角和当前方向角的差超出最大角速度则取最大值
    if ang_vel_max*TS < ori_dif <= PI:
        ang_vel = -ang_vel_max
        vel_desired = (1-ori_dif/PI)*vel_desired
    # 若期望方向角和当前方向角的差超出最大角速度则取最大值
    elif PI < ori_dif < 2*PI-ang_vel_max*TS:
        ang_vel = ang_vel_max
        vel_desired = (1-(2*PI-ori_dif)/PI)*vel_desired
    # 若期望方向角和当前方向角的差未超出最大角速度则小车的角速度定为小车刚好能在转到期望方向
    elif 0 <= ori_dif <= ang_vel_max*TS:
        ang_vel = (theta_desired-ori)/TS
        vel_desired = (1-ori_dif/PI)*vel_desired
    # 若期望方向角和当前方向角的差未超出最大角速度则小车的角速度定为小车刚好能在转到期望方向
    elif 2*PI-ang_vel_max*TS <= ori_dif <= 2*PI:
        ang_vel = (theta_desired-ori)/TS
        vel_desired = (1-(2*PI-ori_dif)/PI)*vel_desired

    # # 若期望方向角和当前方向角的差超出最大角速度则取最大值
    # if 0<=ori_dif<=PI:
    #     ang_vel = -ang_vel_max*ori_dif/PI
    #     vel_desired = (1-ori_dif/PI)*vel_desired
    # # 若期望方向角和当前方向角的差超出最大角速度则取最大值
    # elif PI<ori_dif<2*PI:
    #     ang_vel = ang_vel_max*(2*PI-ori_dif)/PI
    #     vel_desired = (1-(2*PI-ori_dif)/PI)*vel_desired
    # 角速度调整到一个周期内
    if ang_vel*TS > PI:
        ang_vel -= 2*PI/TS
    elif ang_vel*TS < -PI:
        ang_vel += 2*PI/TS
    # 速率的饱和机制
    vel = min(vel_desired, vel_max)
    return vel, ang_vel
