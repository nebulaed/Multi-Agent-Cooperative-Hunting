# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了AllMove函数，实现围捕机器人和目标的实际移动。

from utils.read_yml import Params
from utils.relative_pos import vector_count
from utils.init import ParamsTable


def all_move(wolves: list, targets: list, sta_obss: list, mob_obss: list, irr_obss: list, m_irr_obss: list, rectangle_border: object, t: int, vel_wolves: float, ang_vel_wolves: float, vel_targets: float, ang_vel_targets: float, **kwargs) -> None:
    """
    围捕机器人和目标的实际移动

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        rectangle_border: 边界对象
        t: 当前仿真步数(单位为step)
        vel_wolves: 围捕机器人速度
        ang_vel_wolves: 围捕机器人角速度
        vel_targets: 目标速度
        ang_vel_targets: 目标角速度
    """
    WOLF_NUM, TARGET_NUM = ParamsTable.WOLF_NUM, ParamsTable.TARGET_NUM
    for i in range(WOLF_NUM):
        # 个体移动
        if t != 0:
            vel_try = vel_wolves[i]
            infeasibleFlag = 0
            while (not wolves[i].check_feasibility(vel_try, ang_vel_wolves[i], rectangle_border, wolves, sta_obss, mob_obss, irr_obss, m_irr_obss, i)):
                vel_try *= 0.5
                if vel_try < 0.01:
                    infeasibleFlag = 1
                    break
            # 若移动后位置不超出边界范围且不与其他机器人碰撞，按照输入速度和角速度移动
            if infeasibleFlag == 0:
                wolves[i].move(vel_try, ang_vel_wolves[i])
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
    # 更新向量差
    vector_count(wolves, targets, sta_obss, mob_obss, irr_obss, m_irr_obss, rectangle_border)
