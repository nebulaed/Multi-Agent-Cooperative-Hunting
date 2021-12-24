# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义的函数主要根据算法计算出围捕机器人下一步的速度和角速度。
# AxisTransform: 计算输入矢量在以目标方向为y'轴，逆时针旋转90°为x'轴的x'Oy'坐标系中的x'轴分量的函数。
# FindNear: 围捕机器人计算同伴由近到远的次序。
# Assignment: 在多目标情况下为围捕机器人分配围捕目标
# Attractive: 计算围捕机器人的前进速度
# Repulsion: 计算同伴对围捕机器人的排斥速度
# WolvesHunt: 围捕机器人运动的主函数，在以上函数的基础上计算出围捕机器人下一步运动的速度和角速度。


import numpy as np
from typing import List
from model import Robot, Target, StaObs, MobObs, IrregularObs, MobIrregularObs, Border
from utils.math_func import correct, peri_arctan, arcsin, norm, sin, cos, exp, inc_angle, sqrt, intervals_merge
from utils.control_input import saturator
from utils.params import WOLF_NUM, TARGET_NUM
from utils.collision_avoidance import robot_avoid_obs


def axis_transform(xb: float, yb: float, theta: float):
    """
    计算输入矢量在以目标方向为y'轴，逆时针旋转90°为x'轴的x'Oy'坐标系中的x'轴分量

    输入：
        xb: 向量的x轴分量
        yb: 向量的y轴
        theta: y'轴的方向角度∈[0,2π)
    输出：
        po2[0,0]: 计算得到x'轴分量转换回全局xOy坐标系后的x轴分量。
        po2[1,0]: 计算得到x'轴分量转换回全局xOy坐标系后的x轴分量。
    """
    cosx, sinx = cos(theta), sin(theta)
    # 坐标变换矩阵
    Tab = np.matrix([[cosx, -1*sinx, 0], [sinx, cosx, 0], [0, 0, 1]])
    # 坐标逆变换矩阵
    Tba = np.matrix([[cosx, sinx, 0], [-1*sinx, cosx, 0], [0, 0, 1]])
    po2 = np.matrix([[xb], [yb], [1]])
    # 转换到自主坐标系
    po1 = np.dot(Tba, po2)
    # y'轴分量置零
    po1[1, 0] = 0
    # 转换回全局坐标系
    po2 = np.dot(Tab, po1)
    return po2[0, 0], po2[1, 0]


def find_near(wolves: List) -> None:
    """
    围捕机器人确定同伴由近到远的次序

    输入：
        wolves: 存放所有围捕机器人对象的list
    """
    # 执行个体的成员函数计算同伴由近到远的次序
    for wolf in wolves:
        wolf.find_near()


def assignment(wolves: List) -> List[int]:
    """
    分配围捕目标，原理是各个机器人就近选择目标，假如选择某个目标的机器人数量超过了[机器人数/总目标数]，则离该目标最远的机器人选择离自己次近的目标

    输入：
        wolves: 存放所有围捕机器人对象的list
    输出：
        my_target: 存放各个围捕机器人选择的目标序号的list
    """
    my_target = []
    target_dist = np.zeros((WOLF_NUM, TARGET_NUM))
    for i in range(WOLF_NUM):
        for tar in range(TARGET_NUM):
            target_dist[i, tar] = norm(wolves[i].wolf_to_target[tar])
        my_target.append(np.argmin(target_dist[i]))
    for i in set(my_target):
        if my_target.count(i) > (WOLF_NUM/TARGET_NUM)+1:
            my_target_dists = [[], []]
            for j in range(len(my_target)):
                if my_target[j] == i:
                    my_target_dists[0].append(j)
                    my_target_dists[1].append(target_dist[j, i])
            exclusion = np.argsort(my_target_dists[1], kind='heapsort')
            for item in range(len(my_target_dists[0])):
                if item > (WOLF_NUM/TARGET_NUM):
                    exclusion_index = np.argsort(target_dist[my_target_dists[0][exclusion[item]]], kind='heapsort')
                    my_target[my_target_dists[0]
                              [exclusion[item]]] = exclusion_index[1]
    return my_target


def attractive(i: int, my_t: List[int], wolves: List, RADIUS: float, VARSIGMA: float, TAU_1: float, TAU_2: float, old_track_target: np.ndarray, old_attract: np.ndarray, interact: List):
    """
    计算围捕机器人的前进速度

    输入：
        i: 围捕机器人序号
        my_t: 存放围捕机器人选择的目标list
        wolves: 存放所有围捕机器人对象的list
        RADIUS: 围捕半径(单位为m)
        VARSIGMA: 跟随同伴速度的系数
        TAU_1: 吸引速度的系数
        TAU_2: 追踪速度的系数
        old_track_target: 实际未用到的量，可忽略
        old_attract: 上一步的吸引速度矢量
        interact: 交互拓扑

    输出：
        attract_v: 计算得到的吸引速度矢量
        track_target: 长度为2的np.ndarray，第一个元素为0表示在追击目标，为1表示在跟随同伴，第二个元素表示目标或围捕机器人的序号
        interact: 交互拓扑
    """
    attract_v = np.zeros(2)
    track_target = np.zeros(2)
    # 若该个体能观察到目标
    if norm(wolves[i].wolf_to_target[my_t[i]]) <= wolves[i].R_VISION:
        # 该个体与目标发生了交互
        interact[i][5] = 1
        # 个体到目标的向量
        predict = wolves[i].wolf_to_target[my_t[i]]
        # 个体到目标的距离
        dist = norm(predict)
        # 吸引速度
        # attract_v = (dist-RADIUS)*predict/(dist**2)
        attract_v = (dist-RADIUS)*predict/(dist**TAU_1)
        track_target[0] = 0
        track_target[1] = my_t[i]
    # 若该个体无法观察到目标
    else:
        # 按照由近到远的顺序判断同伴是否能观察到目标
        for j in range(WOLF_NUM-1):
            x = int(wolves[i].neighbor[j])-1
            # 该个体与与x个体发生了交互
            interact[i][x] = 1
            # 若该同伴能观察到的目标是该机器人选择的目标
            if wolves[x].detection and my_t[x] == my_t[i]:
                # attract_v = -VARSIGMA*wolves[i].wolves_dif[x]
                attract_v = -VARSIGMA * \
                    wolves[i].wolves_dif[x] * \
                    (norm(wolves[i].wolves_dif[x])**TAU_2)
                track_target[0] = 1
                track_target[1] = x
                break
    # 若没有同伴观察得到目标则按照上一步吸引速度继续运动
    # and old_track_target[0]==0:
    if attract_v[0] == 0 and attract_v[1] == 0 and norm(old_attract) != 0:
        attract_v = 1.5*old_attract/norm(old_attract)
        track_target[0] = 0
        track_target[1] = my_t[i]
    return attract_v, track_target, interact


def repulsion(i: int, my_t: int, wolves: List, TAU_3: float, interact: List, attract_v: np.ndarray):
    """
    计算同伴对围捕机器人的排斥速度

    输入：
        i: 围捕机器人序号
        my_t: 存放围捕机器人选择的目标list
        wolves: 存放所有围捕机器人对象的list
        TAU_3: 排斥速度的参数
        interact: 交互矩阵
        attract_v: 前进速度

    输出：
        horizontal_v: 计算得到的排斥速度矢量
        interact: 交互矩阵
    """
    horizontal_v = np.zeros(2)
    loose = np.zeros(2)
    # 最近同伴的索引
    x = int(wolves[i].neighbor[0])
    # 次近同伴的索引
    y = int(wolves[i].neighbor[1])
    # 与最近同伴的距离
    dist1 = norm(wolves[i].wolves_dif[x-1])
    # 与次近同伴的距离
    dist2 = norm(wolves[i].wolves_dif[y-1])
    # 若次近同伴已在个体的观察范围内, 则两个同伴均产生排斥
    if dist2 < wolves[i].R_VISION:
        interact[i][x-1] = 1
        interact[i][y-1] = 1
        loose[0:2] = wolves[i].wolves_dif[x-1] / \
            (dist1**TAU_3)+wolves[i].wolves_dif[y-1]/(dist2**TAU_3)
    # 若最近同伴在个体的观察范围内, 而次近同伴不在, 则前者产生排斥
    elif dist1 < wolves[i].R_VISION and dist2 > wolves[i].R_VISION:
        interact[i][x-1] = 1
        loose[0:2] = wolves[i].wolves_dif[x-1]/(dist1**TAU_3)
    # 若最近同伴都不在个体的观察范围内, 则无排斥
    elif dist1 > wolves[i].R_VISION:
        pass
    else:
        raise ValueError(f'Numerical error in variable dist1 {dist1} or dist2 {dist2}.')
    # 计算目标所在方向角
    target_theta = correct(peri_arctan(attract_v)-np.pi/2)
    # 通过坐标系转换算出实际的排斥速度
    horizontal_v[0], horizontal_v[1] = axis_transform(loose[0], loose[1], target_theta)
    return horizontal_v, interact


def robots_movement_strategy(wolves: List[Robot], targets: List[Target], mob_obss: List[MobObs], sta_obss: List[StaObs], irr_obss: List[IrregularObs], m_irr_obss: List[MobIrregularObs], rectangle_border: Border, VARSIGMA: float, ALPHA: float, BETA: float, D_DANGER: float, D_DANGER_W: float, TAU_1: float, TAU_2: float, TAU_3: float, RADIUS: float, t: int, global_my_t: List[int], old_track_target: np.ndarray, old_attract: np.ndarray, interact: List, ASSIGN_CYCLE: int, EXPANSION3: List[float], EXPANSION4: List[float], **kwargs):
    """
    围捕机器人运动的主函数，在以上函数的基础上计算出围捕机器人下一步运动的速度和角速度

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        mob_obss: 存放所有移动障碍物对象的list
        sta_obss: 存放所有固定障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        rectangle_border: 边界对象
        VARSIGMA: 跟随同伴速度的系数
        ALPHA: 前进速度attract_v的系数
        BETA: 同伴排斥速度horizontal_v的系数
        D_DANGER: 围捕机器人启动紧急避障的距离(单位为m)
        D_DANGER_W: 围捕机器人启动紧急避免互撞的距离(单位为m)
        TAU_1: 吸引速度的系数
        TAU_2: 追踪速度的系数
        TAU_3: 排斥速度的参数
        RADIUS: 围捕半径(单位为m)
        t: 当前仿真步数(单位为step)
        global_my_t: 当前步各围捕机器人选择的目标list
        old_track_target: 上一步中各围捕机器人跟踪或追击的对象, 长度为2的np.ndarray，第一个元素为0表示在追击目标，为1表示在跟随同伴，第二个元素表示目标或围捕机器人的序号
        old_attract: 上一步中各围捕机器人的前进速度向量(单位为m/s)
        interact: 初始化的交互拓扑

    输出：
        track_target: 当前步各围捕机器人跟踪或追击的对象, 长度为2的np.ndarray，第一个元素为0表示在追击目标，为1表示在跟随同伴，第二个元素表示目标或围捕机器人的序号
        my_target: 多目标情况下当前步各围捕机器人选择的目标
        vel_wolves: 围捕机器人计算得到的当前步的控制输入线速度(单位为m/s)，尚未实际移动
        ang_vel_wolves: 围捕机器人计算得到的当前步的控制输入角速度(单位为rad/s)，尚未实际移动
        w_d_rs: 围捕机器人的观察范围内的危险角度范围区间∈[0,2π)
        v_vector: 算法计算出的围捕机器人的期望速度向量，可用于画图，便于debug
        save_attract: 当前步的前进速度向量(单位为m/s)
    """
    # 个体将同伴按距离由近到远排序
    find_near(wolves)
    # 机器人个体的跟踪目标
    track_target = np.zeros((WOLF_NUM, 2))
    if t % ASSIGN_CYCLE == 0:
        my_target = assignment(wolves)
    else:
        my_target = global_my_t
    vel_wolves = np.zeros(WOLF_NUM)
    ang_vel_wolves = np.zeros(WOLF_NUM)
    w_d_rs = []
    v_vector, save_attract = np.zeros((WOLF_NUM, 2)), np.zeros((WOLF_NUM, 2))
    targets_hunter = [[] for _ in range(TARGET_NUM)]
    for i in range(WOLF_NUM):
        for j in range(TARGET_NUM):
            if my_target[i] == j:
                targets_hunter[j].append(i)

    for j in range(WOLF_NUM):
        interact[j] = [0] * (WOLF_NUM+TARGET_NUM)
        if norm(wolves[j].wolf_to_target[my_target[j]]) < wolves[j].R_VISION:
            wolves[j].detection = True
        else:
            wolves[j].detection = False
        attract_v, track_target[j], interact = attractive(j, my_target, wolves, RADIUS, VARSIGMA, TAU_1, TAU_2, old_track_target[j], old_attract[j], interact)
        save_attract[j] = attract_v
        attract_v = ALPHA*attract_v
        horizontal_v, interact = repulsion(j, my_target[j], wolves, TAU_3, interact, attract_v)
        horizontal_v = BETA*horizontal_v
        # 将速度矢量转换为期望速度和期望航向角
        variation = attract_v+horizontal_v
        vel_wolf_desired = norm(variation)
        theta_wolf_desired = peri_arctan(variation)
        # 避障
        vel_wolf_desired, theta_wolf_desired, w_d_r = robot_avoid_obs(t, j, vel_wolf_desired, theta_wolf_desired, my_target[j], wolves, mob_obss, sta_obss, irr_obss, m_irr_obss, rectangle_border, D_DANGER, D_DANGER_W, EXPANSION3, EXPANSION4)
        w_d_rs.append(w_d_r)

        v_vector[j, 0] = vel_wolf_desired*cos(theta_wolf_desired)
        v_vector[j, 1] = vel_wolf_desired*sin(theta_wolf_desired)
        # import matplotlib.pyplot as plt
        # plt.arrow(wolves[j].pos[0],wolves[j].pos[1],0.5*cos(theta_wolf_desired),0.5*sin(theta_wolf_desired),color='k', width=0.025, head_width=0.2, head_length=0.3)
        vel_wolves[j], ang_vel_wolves[j] = saturator(wolves[j].ori, wolves[j].vel_max, wolves[j].ang_vel_max, vel_wolf_desired, theta_wolf_desired)

    return track_target, my_target, vel_wolves, ang_vel_wolves, w_d_rs, v_vector, save_attract
