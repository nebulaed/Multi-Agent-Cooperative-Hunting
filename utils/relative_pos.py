# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了计算一些对象(包括围捕机器人、目标、固定和移动障碍物)间的向量差以及围捕机器人和目标到边界距离的函数。


import numpy as np
from numba import jit
from typing import List
from model import Robot, Target, StaObs, MobObs, IrregularObs, MobIrregularObs, Border
from utils.params import WOLF_NUM, TARGET_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM
from utils.math_func import norm


def vector_count(wolves: List[Robot], targets: List[Target], sta_obss: List[StaObs], mob_obss: List[MobObs], irr_obss: List[IrregularObs], m_irr_obss: List[MobIrregularObs], border: Border) -> None:
    """
    计算一些对象(包括围捕机器人、目标、固定和移动障碍物)间的向量差以及围捕机器人和目标到边界距离。

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        border: 边界对象
    """

    # 目标到各边界距离以及最近边界
    for i in range(TARGET_NUM):
        targets[i].border_d[0] = abs(targets[i].pos[0]-border.X_MIN)
        targets[i].border_d[1] = abs(targets[i].pos[1]-border.Y_MIN)
        targets[i].border_d[2] = abs(targets[i].pos[0]-border.X_MAX)
        targets[i].border_d[3] = abs(targets[i].pos[1]-border.Y_MAX)
        # 找出最近墙
        w = np.argmin(targets[i].border_d)
        # 墙上最近点
        if w == 0:
            targets[i].nearest = np.array([border.X_MIN, targets[i].pos[1]])
        elif w == 1:
            targets[i].nearest = np.array([targets[i].pos[0], border.Y_MIN])
        elif w == 2:
            targets[i].nearest = np.array([border.X_MAX, targets[i].pos[1]])
        elif w == 3:
            targets[i].nearest = np.array([targets[i].pos[0], border.Y_MAX])
        else:
            raise ValueError(f'Numerical error in variable w: {w}.')


    # 个体到各边界距离以及最近边界
    for i in range(WOLF_NUM):
        wolves[i].border_d[0] = abs(wolves[i].pos[0]-border.X_MIN)
        wolves[i].border_d[1] = abs(wolves[i].pos[1]-border.Y_MIN)
        wolves[i].border_d[2] = abs(wolves[i].pos[0]-border.X_MAX)
        wolves[i].border_d[3] = abs(wolves[i].pos[1]-border.Y_MAX)
        # 找出最近墙
        w = np.argmin(wolves[i].border_d)
        # 墙上最近点
        if w == 0:
            wolves[i].nearest = np.array([border.X_MIN, wolves[i].pos[1]])
        elif w == 1:
            wolves[i].nearest = np.array([wolves[i].pos[0], border.Y_MIN])
        elif w == 2:
            wolves[i].nearest = np.array([border.X_MAX, wolves[i].pos[1]])
        elif w == 3:
            wolves[i].nearest = np.array([wolves[i].pos[0], border.Y_MAX])
        else:
            raise ValueError(f'Numerical error in variable w: {w}.')

    import matplotlib.pyplot as plt
    for i in range(WOLF_NUM):
        # 个体到移动障碍物的向量
        # random_gamma = 15*(PI/180)*np.sin(np.random.rand(M_OBS_NUM)*2*PI)
        wolves[i].danger_m = []
        for j in range(M_OBS_NUM):
            # RM = np.array([[np.cos(random_gamma[j]),-np.sin(random_gamma[j])],[np.sin(random_gamma[j]),np.cos(random_gamma[j])]])
            # measure_noise = np.random.uniform(0.90,1.10)
            # wolves[i].wolf_to_m_obs[j] = np.dot(mob_obss[j].pos-wolves[i].pos,RM.T)*measure_noise
            wolves[i].wolf_to_m_obs[j] = mob_obss[j].pos-wolves[i].pos
            # 个体i和各移动障碍物j的距离
            wolves[i].d_m[j] = norm(wolves[i].wolf_to_m_obs[j])-mob_obss[j].R
            if 0 < wolves[i].d_m[j] < wolves[i].AVOID_DIST:
                wolves[i].danger_m.append(j)
        # 个体到固定障碍物的向量
        # random_gamma = 15*(PI/180)*np.sin(np.random.rand(S_OBS_NUM)*2*PI)
        wolves[i].danger_s = []
        for j in range(S_OBS_NUM):
            # RM = np.array([[np.cos(random_gamma[j]),-np.sin(random_gamma[j])],[np.sin(random_gamma[j]),np.cos(random_gamma[j])]])
            # measure_noise = np.random.uniform(0.90,1.10)
            # wolves[i].wolf_to_s_obs[j] = np.dot(sta_obss[j].pos-wolves[i].pos,RM.T)*measure_noise
            wolves[i].wolf_to_s_obs[j] = sta_obss[j].pos-wolves[i].pos
            # 个体i和各固定障碍物j的距离
            wolves[i].d_s[j] = norm(wolves[i].wolf_to_s_obs[j])-sta_obss[j].R
            if 0 < wolves[i].d_s[j] < wolves[i].AVOID_DIST:
                wolves[i].danger_s.append(j)
        wolves[i].danger_ir = []
        for j in range(IRR_OBS_NUM):
            wolves[i].d_ir[j], wolves[i].wolf_to_irr_obs[j] = find_closest_point(irr_obss[j].elements, wolves[i].pos)
            if 0 < wolves[i].d_ir[j] < wolves[i].AVOID_DIST:
                # plt.plot(irr_obss[j].elements[mind_point][0],irr_obss[j].elements[mind_point][1],'ko')
                wolves[i].danger_ir.append(j)
        wolves[i].danger_m_ir = []
        for j in range(M_IRR_OBS_NUM):
            wolves[i].d_m_ir[j], wolves[i].wolf_to_m_irr_obs[j] = find_closest_point(m_irr_obss[j].elements, wolves[i].pos)
            if 0 < wolves[i].d_m_ir[j] < wolves[i].AVOID_DIST:
                # plt.plot(m_irr_obss[j].elements[mind_point][0],m_irr_obss[j].elements[mind_point][1],'ko')
                wolves[i].danger_m_ir.append(j)
        # 个体i距离较近的墙
        wolves[i].danger_border = []
        for j in range(np.size(wolves[i].border_d)):
            if 0 < wolves[i].border_d[j] < wolves[i].DIS_AVOID_BORDER:
                wolves[i].danger_border.append(j)
        # 个体到目标的向量及是否观察到目标
        # random_gamma = 15*(PI/180)*np.sin(np.random.rand(TARGET_NUM)*2*PI)
        for j in range(TARGET_NUM):
            # RM = np.array([[np.cos(random_gamma[j]),-np.sin(random_gamma[j])],[np.sin(random_gamma[j]),np.cos(random_gamma[j])]])
            # measure_noise = np.random.uniform(0.90,1.10)
            # wolves[i].wolf_to_target[j] = np.dot(targets[j].pos-wolves[i].pos,RM.T)*measure_noise
            wolves[i].wolf_to_target[j] = targets[j].pos-wolves[i].pos
        # 个体间的向量差
        # random_gamma = 15*(PI/180)*np.sin(np.random.rand(WOLF_NUM)*2*PI)
        for j in range(WOLF_NUM):
            # RM = np.array([[np.cos(random_gamma[j]),-np.sin(random_gamma[j])],[np.sin(random_gamma[j]),np.cos(random_gamma[j])]])
            # measure_noise = np.random.uniform(0.90,1.10)
            # wolves[i].wolves_dif[j] = np.dot(wolves[i].pos-wolves[j].pos,RM.T)*measure_noise
            wolves[i].wolves_dif[j] = wolves[i].pos-wolves[j].pos

    for i in range(TARGET_NUM):
        # 目标到个体的向量
        # random_gamma = 15*(PI/180)*np.sin(np.random.rand(WOLF_NUM)*2*PI)
        for j in range(WOLF_NUM):
            # RM = np.array([[np.cos(random_gamma[j]),-np.sin(random_gamma[j])],[np.sin(random_gamma[j]),np.cos(random_gamma[j])]])
            # measure_noise = np.random.uniform(0.90,1.10)
            # targets[i].target_to_wolf[j] = np.dot(wolves[j].pos-targets[i].pos,RM.T)*measure_noise
            targets[i].target_to_wolf[j] = wolves[j].pos-targets[i].pos
        # 目标到移动障碍物的向量
        # random_gamma = 15*(PI/180)*np.sin(np.random.rand(M_OBS_NUM)*2*PI)
        targets[i].danger_m = []
        for j in range(M_OBS_NUM):
            # RM = np.array([[np.cos(random_gamma[j]),-np.sin(random_gamma[j])],[np.sin(random_gamma[j]),np.cos(random_gamma[j])]])
            # measure_noise = np.random.uniform(0.90,1.10)
            # targets[i].target_to_m_obs[j] = np.dot(mob_obss[j].pos-targets[i].pos,RM.T)*measure_noise
            targets[i].target_to_m_obs[j] = mob_obss[j].pos-targets[i].pos
            # 目标和各移动障碍物j的距离
            targets[i].d_m[j] = norm(targets[i].target_to_m_obs[j])-mob_obss[j].R
            if 0 < targets[i].d_m[j] < targets[i].AVOID_DIST:
                targets[i].danger_m.append(j)
        # 目标到固定障碍物的向量
        # random_gamma = 15*(PI/180)*np.sin(np.random.rand(S_OBS_NUM)*2*PI)
        targets[i].danger_s = []
        for j in range(S_OBS_NUM):
            # RM = np.array([[np.cos(random_gamma[j]),-np.sin(random_gamma[j])],[np.sin(random_gamma[j]),np.cos(random_gamma[j])]])
            # measure_noise = np.random.uniform(0.90,1.10)
            # targets[i].target_to_s_obs[j] = np.dot(sta_obss[j].pos-targets[i].pos,RM.T)*measure_noise
            targets[i].target_to_s_obs[j] = sta_obss[j].pos-targets[i].pos
            # 目标和各固定障碍物j的距离
            targets[i].d_s[j] = norm(targets[i].target_to_s_obs[j])-sta_obss[j].R
            if 0 < targets[i].d_s[j] < targets[i].AVOID_DIST:
                targets[i].danger_s.append(j)
        targets[i].danger_ir = []
        for j in range(IRR_OBS_NUM):
            targets[i].d_ir[j], _ = find_closest_point(irr_obss[j].elements, targets[i].pos)
            if 0 < targets[i].d_ir[j] < targets[i].AVOID_DIST:
                targets[i].danger_ir.append(j)
        targets[i].danger_m_ir = []
        for j in range(M_IRR_OBS_NUM):
            targets[i].d_m_ir[j], _ = find_closest_point(m_irr_obss[j].elements, targets[i].pos)
            if 0 < targets[i].d_m_ir[j] < targets[i].AVOID_DIST:
                targets[i].danger_m_ir.append(j)
        # 目标距离较近的墙
        targets[i].danger_border = []
        for j in range(np.size(targets[i].border_d)):
            if 0 < targets[i].border_d[j] < targets[i].DIS_AVOID_BORDER:
                targets[i].danger_border.append(j)


@jit(nopython = True)
def find_closest_point(elements: np.ndarray, pos: np.ndarray):
    points_dist = np.zeros(len(elements))
    for i in range(len(elements)):
        points_dist[i] = norm(elements[i]-pos)
    # 个体i和各不规则障碍物j的距离
    mind_point = np.argmin(points_dist)
    return points_dist[mind_point], elements[mind_point]-pos
