# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了一些函数用来初始化地图以及初始化所有围捕机器人、目标、障碍物。
# RandomSpawn: 计算目标的初始位置
# StaObsInit: 计算固定障碍物的初始位置
# MobObsInit: 计算移动障碍物的初始位置
# IrrObsInit: 计算不规则障碍物的初始位置
# MobIrrObsInit: 计算移动不规则障碍物的初始位置
# Init: 用于初始化地图以及调用以上函数初始化所有围捕机器人、目标、障碍物


import numpy as np
from utils.math_func import correct, peri_arctan, arcsin, norm, sin, cos, exp, inc_angle, sqrt
from time import time
from model import Robot, Target, StaObs, MobObs, IrregularObs, MobIrregularObs, Border
from utils.read_yml import Params


# 读取参数
ParamsTable = Params('params.yml')
PI = np.pi

def random_spawn(wolves: list) -> np.ndarray:
    """
    在随机个体的观察范围内随机选择一点作为目标的出生点

    输入：
        wolves: 存放所有围捕机器人对象的list

    输出：
        np.array([x,y]): 目标的坐标
    """
    # 天选之子, 目标将诞生在你附近
    chosen_son = np.random.randint(ParamsTable.WOLF_NUM)
    # 锁定天选之子的坐标
    cen_p = wolves[chosen_son].pos
    # 计算目标在天选之子的哪个方向
    polar_a = np.random.random()*2*PI-PI
    x = cos(polar_a)
    y = sin(polar_a)
    # 计算目标和天选之子的距离
    polar_r = np.sqrt(np.random.random())*wolves[chosen_son].R_VISION
    # 确定目标出生坐标
    x = x*polar_r+cen_p[0]
    y = y*polar_r+cen_p[1]
    return np.array([x, y])


def sta_obs_init(wolves: list, targets: list, border: Border) -> list:
    """
    计算固定障碍物的初始位置

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        border: 边界对象

    输出：
        sta_obss: 存放所有固定障碍物对象的list
    """
    t1 = time()
    WOLF_NUM, TARGET_NUM, S_OBS_NUM, INIT_D = ParamsTable.WOLF_NUM, ParamsTable.TARGET_NUM, ParamsTable.S_OBS_NUM, ParamsTable.INIT_D
    sta_obss = []
    for i in range(S_OBS_NUM):
        obs_r = np.random.uniform(0.25, 1.25)
        ok = False
        while(not(ok)):
            t2 = time()
            if t2-t1 > 2:
                return 0
            ok = True
            instance = [np.random.uniform(border.X_MIN+2, border.X_MAX-2), np.random.uniform(border.Y_MIN+2, border.Y_MAX-2)]
            # 检查固定障碍物出生点是否和个体有安全距离
            for i in range(WOLF_NUM):
                dist = norm(instance-wolves[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查固定障碍物出生点是否和目标有安全距离
            for i in range(TARGET_NUM):
                dist = norm(instance-targets[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查固定障碍物出生点是否和已出生固定障碍物有安全距离
            if len(sta_obss) != 0:
                for item in sta_obss:
                    dist = norm(instance-item.pos)
                    if dist <= obs_r+INIT_D+item.R:
                        ok = False
                        break
            if not(ok):
                continue
            ok = True
        sta_obss.append(StaObs([obs_r, instance[0], instance[1]]))
    return sta_obss


def mob_obs_init(wolves: list, targets: list, sta_obss: list, border: Border) -> list:
    """
    计算移动障碍物的初始位置

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        border: 边界对象

    输出：
        mob_obss: 存放所有移动障碍物对象的list
    """
    t1 = time()
    WOLF_NUM, TARGET_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, INIT_D = ParamsTable.WOLF_NUM, ParamsTable.TARGET_NUM, ParamsTable.S_OBS_NUM, ParamsTable.M_OBS_NUM, ParamsTable.IRR_OBS_NUM, ParamsTable.INIT_D
    mob_obss = []
    for i in range(M_OBS_NUM):
        obs_r = np.random.uniform(0.25, 1.25)
        ok = False
        while(not(ok)):
            t2 = time()
            if t2-t1 > 2:
                return 0
            ok = True
            instance = [np.random.uniform(border.X_MIN+2, border.X_MAX-2), np.random.uniform(border.Y_MIN+2, border.Y_MAX-2)]
            # 检查移动障碍物出生点是否和个体有安全距离
            for i in range(WOLF_NUM):
                dist = norm(instance-wolves[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动障碍物出生点是否和目标有安全距离
            for i in range(TARGET_NUM):
                dist = norm(instance-targets[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动障碍物出生点是否和固定障碍物有安全距离
            for item in sta_obss:
                dist = norm(instance-item.pos)
                if dist <= obs_r+INIT_D+item.R:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动障碍物出生点是否和已出生的移动障碍物有安全距离
            if len(mob_obss) != 0:
                for item in mob_obss:
                    dist = norm(instance-item.pos)
                    if dist <= obs_r+INIT_D+item.R:
                        ok = False
                        break
            if not(ok):
                continue
            ok = True
        mob_obss.append(MobObs([obs_r, instance[0], instance[1]]))
    return mob_obss


def irr_obs_init(wolves: list, targets: list, sta_obss: list, mob_obss: list, border: Border) -> list:
    """
    计算不规则障碍物的初始位置

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        border: 边界对象

    输出：
        irr_obss: 存放所有不规则障碍物对象的list
    """
    t1 = time()
    WOLF_NUM, TARGET_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, INIT_D = ParamsTable.WOLF_NUM, ParamsTable.TARGET_NUM, ParamsTable.S_OBS_NUM, ParamsTable.M_OBS_NUM, ParamsTable.IRR_OBS_NUM, ParamsTable.INIT_D
    irr_obss = []
    for i in range(IRR_OBS_NUM):
        obs_r = np.random.uniform(0.5, 1.25)
        ok = False
        while (not(ok)):
            t2 = time()
            if t2-t1 > 2:
                return 0
            ok = True
            instance = [np.random.uniform(border.X_MIN+2, border.X_MAX-2), np.random.uniform(border.Y_MIN+2, border.Y_MAX-2)]
            # 检查不规则障碍物出生点是否和个体有安全距离
            for i in range(WOLF_NUM):
                dist = norm(instance-wolves[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查不规则障碍物出生点是否和目标有安全距离
            for i in range(TARGET_NUM):
                dist = norm(instance-targets[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查不规则障碍物出生点是否和固定障碍物有安全距离
            for item in sta_obss:
                dist = norm(instance-item.pos)
                if dist <= obs_r+INIT_D+item.R:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查不规则障碍物出生点是否和移动障碍物有安全距离
            for item in mob_obss:
                dist = norm(instance-item.pos)
                if dist <= obs_r+INIT_D+item.R:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查不规则障碍物出生点是否和已出生的不规则障碍物有安全距离
            if len(irr_obss) != 0:
                for item in irr_obss:
                    dist = norm(instance-item.pos)
                    if dist <= obs_r+INIT_D+item.R:
                        ok = False
                        break
            if not(ok):
                continue
            ok = True
        irr_obss.append(IrregularObs([obs_r, instance[0], instance[1]]))
    return irr_obss


def mob_irr_obs_init(wolves: list, targets: list, sta_obss: list, mob_obss: list, irr_obss: list, border: Border) -> list:
    """
    计算移动不规则障碍物的初始位置

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        border: 边界对象

    输出：
        m_irr_obss: 存放所有移动不规则障碍物对象的list
    """
    t1 = time()
    WOLF_NUM, TARGET_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM, INIT_D = ParamsTable.WOLF_NUM, ParamsTable.TARGET_NUM, ParamsTable.S_OBS_NUM, ParamsTable.M_OBS_NUM, ParamsTable.IRR_OBS_NUM, ParamsTable.M_IRR_OBS_NUM, ParamsTable.INIT_D
    m_irr_obss = []
    for i in range(M_IRR_OBS_NUM):
        obs_r = np.random.uniform(0.5, 1.25)
        ok = False
        while (not(ok)):
            t2 = time()
            if t2-t1 > 2:
                return 0
            ok = True
            instance = [np.random.uniform(border.X_MIN+2, border.X_MAX-2), np.random.uniform(border.Y_MIN+2, border.Y_MAX-2)]
            # 检查移动不规则障碍物出生点是否和个体有安全距离
            for i in range(WOLF_NUM):
                dist = norm(instance-wolves[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动不规则障碍物出生点是否和目标有安全距离
            for i in range(TARGET_NUM):
                dist = norm(instance-targets[i].pos)
                if dist <= obs_r+INIT_D:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动不规则障碍物出生点是否和固定障碍物有安全距离
            for item in sta_obss:
                dist = norm(instance-item.pos)
                if dist <= obs_r+INIT_D+item.R:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动不规则障碍物出生点是否和移动障碍物有安全距离
            for item in mob_obss:
                dist = norm(instance-item.pos)
                if dist <= obs_r+INIT_D+item.R:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动不规则障碍物出生点是否和不规则障碍物有安全距离
            for item in irr_obss:
                dist = norm(instance-item.pos)
                if dist <= obs_r+INIT_D+item.R:
                    ok = False
                    break
            if not(ok):
                continue
            # 检查移动不规则障碍物出生点是否和已出生的移动不规则障碍物有安全距离
            if len(m_irr_obss) != 0:
                for item in m_irr_obss:
                    dist = norm(instance-item.pos)
                    if dist <= obs_r+INIT_D+item.R:
                        ok = False
                        break
            if not(ok):
                continue
            ok = True
        m_irr_obss.append(MobIrregularObs([obs_r, instance[0], instance[1]]))
    return m_irr_obss


def init():
    """
    初始化地图以及调用以上函数初始化所有围捕机器人、目标、障碍物

    输出：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        rectangle_border: 边界对象
    """
    init_fail = True
    # 如果初始化失败(例如随机初始化的障碍物太大导致无论如何摆放在地图边界范围内都放不下)，重新初始化
    while(init_fail):
        rectangle_border = Border(ParamsTable.border)
        wolves = [Robot([np.random.uniform(0, 2*PI), np.random.uniform(rectangle_border.X_MIN/4*3, rectangle_border.X_MAX/4*3),
                        np.random.uniform(rectangle_border.Y_MIN/4*3, rectangle_border.Y_MAX/4*3)]) for i in range(ParamsTable.WOLF_NUM)]
        spawn_p = np.zeros((ParamsTable.TARGET_NUM, 2))
        for i in range(ParamsTable.TARGET_NUM):
            spawn_p[i] = random_spawn(wolves)
        targets = [Target([np.random.uniform(0, 2*PI), spawn_p[i, 0], spawn_p[i, 1]])
                   for i in range(ParamsTable.TARGET_NUM)]
        sta_obss = sta_obs_init(wolves, targets, rectangle_border)
        # 初始化过程中有重叠，重新初始化
        if sta_obss == 0:
            continue
        mob_obss = mob_obs_init(wolves, targets, sta_obss, rectangle_border)
        # 初始化过程中有重叠，重新初始化
        if mob_obss == 0:
            continue
        irr_obss = irr_obs_init(wolves, targets, sta_obss, mob_obss, rectangle_border)
        # 初始化过程中有重叠，重新初始化
        if irr_obss == 0:
            continue
        m_irr_obss = mob_irr_obs_init(wolves, targets, sta_obss, mob_obss, irr_obss, rectangle_border)
        if m_irr_obss == 0:
            continue
        init_fail = False
    return wolves, targets, sta_obss, mob_obss, irr_obss, m_irr_obss, rectangle_border
