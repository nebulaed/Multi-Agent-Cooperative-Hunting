# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了AllMove函数，实现围捕机器人和目标的实际移动。

from typing import List
import numpy as np
from model import Robot, Target, StaObs, MobObs, IrregularObs, MobIrregularObs, Border
from utils.params import WOLF_NUM, TARGET_NUM

# TODO: 速度优化重点
def all_move(wolves: List[Robot], targets: List[Target], sta_obss: List[StaObs], mob_obss: List[MobObs], irr_obss: List[IrregularObs], m_irr_obss: List[MobIrregularObs], rectangle_border: Border, t: int, vel_wolves: np.ndarray, ang_vel_wolves: np.ndarray, vel_targets: np.ndarray, ang_vel_targets: np.ndarray, **kwargs) -> None:
    """
    围捕机器人和目标的实际移动

    输入：
        @param wolves: 存放所有围捕机器人对象的list
        @param targets: 存放所有目标对象的list
        @param sta_obss: 存放所有固定障碍物对象的list
        @param mob_obss: 存放所有移动障碍物对象的list
        @param irr_obss: 存放所有不规则障碍物对象的list
        @param m_irr_obss: 存放所有移动不规则障碍物对象的list
        @param rectangle_border: 边界对象
        @param t: 当前仿真步数(单位为step)
        @param vel_wolves: 围捕机器人速度(单位为m/s)
        @param ang_vel_wolves: 围捕机器人角速度(单位为rad/s)
        @param vel_targets: 目标速度(单位为m/s)
        @param ang_vel_targets: 目标角速度(单位为rad/s)
    """
    for i in range(WOLF_NUM):
        # 个体移动
        if t != 0:
            # 若移动后位置不超出边界范围且不与其他机器人碰撞，按照输入速度和角速度移动
            if wolves[i].check_feasibility(vel_wolves[i], ang_vel_wolves[i], rectangle_border, wolves, sta_obss, mob_obss, irr_obss, m_irr_obss, i):
                wolves[i].move(vel_wolves[i], ang_vel_wolves[i])
            # 若移动后位置超出了边界范围或与其他机器人碰撞则令线速度为零, 角速度按照原定输入来
            else:
                wolves[i].move(0, ang_vel_wolves[i])
    for i in range(TARGET_NUM):
        # 目标移动
        if t != 0:
            if targets[i].check_feasibility(vel_targets[i], ang_vel_targets[i], rectangle_border, targets, sta_obss, mob_obss, irr_obss, m_irr_obss, i):
                targets[i].move(vel_targets[i], ang_vel_targets[i])
            else:
                targets[i].move(0, ang_vel_targets[i])
