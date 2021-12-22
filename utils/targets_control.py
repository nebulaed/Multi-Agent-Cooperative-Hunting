# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义的函数主要用于根据算法计算出目标下一步的速度和角速度。
# Ang2w: 由障碍物和当前车头方向的角度差计算出期望运动角度
# AroundTarget: 目标的避障算法
# TargetGo: 目标运动的主函数，在以上函数的基础上计算出目标下一步运动的速度和角速度


import numpy as np
import matplotlib.pyplot as plt
from utils.math_func import correct, peri_arctan, arcsin, norm, sin, cos, exp, inc_angle, intervals_merge
from utils.robots_control import saturator
from utils.params import WOLF_NUM, TARGET_NUM, PI, TS


def target_avoid_obs(t: int, mark: int, vel_target_desired: float, theta_target_desired: float, targets: list, mob_obss: list, sta_obss: list, irr_obss: list, m_irr_obss: list, border: object, EXPANSION1: list, EXPANSION2: list, danger_direction: list = [], danger_index: list = []):
    """
    目标的避障算法

    输入：
        t: 当前仿真步数(单位为step)
        mark: 目标的序号
        vel_target_desired: 在不考虑避障情况下，目标的期望速度(单位为m/s)
        theta_target_desired: 在不考虑避障情况下，目标的期望方向角∈[0,2π)
        targets: 存放所有目标对象的list
        mob_obss: 存放所有移动障碍物对象的list
        sta_obss: 存放所有固定障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        border: 边界对象
        danger_direction: 将围捕机器人也当成障碍物判断危险角度范围，实际未用到，是一个空list，逃避围捕在TargetGo函数中已实现，可忽略
        danger_index: 目标已观察到的围捕机器人序号(索引)，实际未用到，是一个空list，可忽略

    输出：
        vel_target_desired: 考虑避障下，目标的期望速度(单位为m/s)
        theta_target_desired: 考虑避障下，目标的期望方向角∈[0,2π)
        dangerous_ranges_organized: 目标的观察范围内的危险角度范围区间∈[0,2π)
    """
    # 用字典同时记录障碍物和目标的距离和障碍物的索引
    dict_d_obs = {}
    for i in range(len(targets[mark].danger_m)):
        obs_ind = targets[mark].danger_m[i]
        dict_d_obs[(0, obs_ind)] = targets[mark].d_m[obs_ind]
    for i in range(len(targets[mark].danger_s)):
        obs_ind = targets[mark].danger_s[i]
        dict_d_obs[(1, obs_ind)] = targets[mark].d_s[obs_ind]
    for i in range(len(targets[mark].danger_ir)):
        obs_ind = targets[mark].danger_ir[i]
        dict_d_obs[(2, obs_ind)] = targets[mark].d_ir[obs_ind]
    for i in range(len(targets[mark].danger_m_ir)):
        obs_ind = targets[mark].danger_m_ir[i]
        dict_d_obs[(3, obs_ind)] = targets[mark].d_m_ir[obs_ind]
    for i in range(len(targets[mark].danger_border)):
        obs_ind = targets[mark].danger_border[i]
        dict_d_obs[(4, obs_ind)] = targets[mark].border_d[obs_ind]
    for i in range(len(danger_index)):
        obs_ind = danger_index[i]
        dict_d_obs[(5, obs_ind)] = norm(targets[mark].target_to_wolf[obs_ind])
    if len(dict_d_obs) == 1:
        safety_bor, safety_mob, safety_sta, safety_m_irr, safety_enemy = EXPANSION1[0], EXPANSION1[1], EXPANSION1[2], EXPANSION1[3], EXPANSION1[4]
    elif len(dict_d_obs) >= 2:
        safety_bor, safety_mob, safety_sta, safety_m_irr, safety_enemy = EXPANSION2[0], EXPANSION2[1], EXPANSION2[2], EXPANSION2[3], EXPANSION2[4]
    # 按照障碍物的距离将其和对应的索引进行排序
    dict_d_obs = sorted(dict_d_obs.items(), key=lambda item: item[1])
    dangerous_ranges_dists, dangerous_ranges, dangerous_ranges_indexs = [], [], []
    if len(dict_d_obs) != 0:
        for item in dict_d_obs:
            index = item[0]
            dangerous_range = None
            # 若是移动障碍物
            if index[0] == 0:
                i_m = index[1]
                dist_m = item[1]
                # 计算个体到移动障碍物的方向角
                psi1 = peri_arctan(targets[mark].target_to_m_obs[i_m])
                # 移动障碍物中心连线与切线的夹角
                Delta1 = arcsin(mob_obss[i_m].R/(dist_m+mob_obss[i_m].R))*safety_mob
                # 记录危险角度
                dangerous_range = [correct(psi1-Delta1), correct(psi1+Delta1)]
            # 若是固定障碍物
            elif index[0] == 1:
                i_s = index[1]
                dist_s = item[1]
                # 计算目标到固定障碍物的方向角
                psi2 = peri_arctan(targets[mark].target_to_s_obs[i_s])
                # 固定障碍物中心连线与切线的夹角
                Delta2 = arcsin(sta_obss[i_s].R/(dist_s+sta_obss[i_s].R))*safety_sta
                # 记录危险角度
                dangerous_range = [correct(psi2-Delta2), correct(psi2+Delta2)]
            # 若是不规则障碍物
            elif index[0] == 2:
                # 获取该不规则障碍物的索引
                i_ir = index[1]
                psi_points = []
                for point in irr_obss[i_ir].elements:
                    if norm(point-targets[mark].pos) < targets[mark].AVOID_DIST:
                        psi_point = peri_arctan(point-targets[mark].pos)
                        psi_points.append(psi_point)
                if len(psi_points) >= 6:
                    psi_points_difs, psi_points_difs_index = [], []
                    for i in range(len(psi_points)):
                        for j in range(i+1, len(psi_points)):
                            psi_points_dif = psi_points[i]-psi_points[j] if psi_points[i] - \
                                psi_points[j] >= 0 else psi_points[i]-psi_points[j]+2*PI
                            psi_points_dif = 2*PI-psi_points_dif if psi_points_dif > PI else psi_points_dif
                            psi_points_difs.append(psi_points_dif)
                            psi_points_difs_index.append([i, j])
                    tangent_1 = psi_points[psi_points_difs_index[np.argmax(psi_points_difs)][0]]
                    tangent_2 = psi_points[psi_points_difs_index[np.argmax(psi_points_difs)][1]]
                    if abs(tangent_1-tangent_2) > PI:
                        bisector = correct((tangent_1+tangent_2)/2+PI)
                        half_ang = (PI-abs(tangent_1-tangent_2)/2)*safety_mob
                    else:
                        bisector = (tangent_1+tangent_2)/2
                        half_ang = abs(tangent_1-tangent_2)/2*safety_mob
                    # 记录危险角度
                    dangerous_range = [correct(bisector-half_ang), correct(bisector+half_ang)]
            # 若是移动不规则障碍物
            elif index[0] == 3:
                # 获取距离最近移动不规则障碍物的索引
                i_m_ir = index[1]
                psi_points = []
                for point in m_irr_obss[i_m_ir].elements:
                    if norm(point-targets[mark].pos) < targets[mark].AVOID_DIST:
                        psi_point = peri_arctan(point-targets[mark].pos)
                        psi_points.append(psi_point)
                if len(psi_points) >= 6:
                    psi_points_difs, psi_points_difs_index = [], []
                    for i in range(len(psi_points)):
                        for j in range(i+1, len(psi_points)):
                            psi_points_dif = psi_points[i]-psi_points[j] if psi_points[i] - \
                                psi_points[j] >= 0 else psi_points[i]-psi_points[j]+2*PI
                            psi_points_dif = 2*PI-psi_points_dif if psi_points_dif > PI else psi_points_dif
                            psi_points_difs.append(psi_points_dif)
                            psi_points_difs_index.append([i, j])
                    tangent_1 = psi_points[psi_points_difs_index[np.argmax(psi_points_difs)][0]]
                    tangent_2 = psi_points[psi_points_difs_index[np.argmax(psi_points_difs)][1]]
                    if abs(tangent_1-tangent_2) > PI:
                        bisector = correct((tangent_1+tangent_2)/2+PI)
                        half_ang = (PI-abs(tangent_1-tangent_2)/2)*safety_m_irr
                    else:
                        bisector = (tangent_1+tangent_2)/2
                        half_ang = abs(tangent_1-tangent_2)/2*safety_m_irr
                    # 记录危险角度
                    dangerous_range = [correct(bisector-half_ang), correct(bisector+half_ang)]
            # 若是边界
            elif index[0] == 4:
                # 若是左边界
                if index[1] == 0:
                    # 计算感知范围圆与边界的交点1的y坐标
                    cut_off_point1_y = np.sqrt(targets[mark].DIS_AVOID_BORDER**2-
                                       (border.X_MIN-targets[mark].pos[0])**2)+targets[mark].pos[1]
                    # 算得交点1坐标
                    cut_off_point1 = [border.X_MIN, cut_off_point1_y]
                    # 计算感知范围圆与边界的交点2的y坐标
                    cut_off_point2_y = -np.sqrt(targets[mark].DIS_AVOID_BORDER**2-
                                       (border.X_MIN-targets[mark].pos[0])**2)+targets[mark].pos[1]
                    # 算得交点2坐标
                    cut_off_point2 = [border.X_MIN, cut_off_point2_y]
                # 若是下边界
                elif index[1] == 1:
                    # 计算感知范围圆与边界的交点1的x坐标
                    cut_off_point1_x = np.sqrt(targets[mark].DIS_AVOID_BORDER**2-
                                       (border.Y_MIN-targets[mark].pos[1])**2)+targets[mark].pos[0]
                    # 算得交点1坐标
                    cut_off_point1 = [cut_off_point1_x, border.Y_MIN]
                    # 计算感知范围圆与边界的交点2的x坐标
                    cut_off_point2_x = -np.sqrt(targets[mark].DIS_AVOID_BORDER**2-
                                       (border.Y_MIN-targets[mark].pos[1])**2)+targets[mark].pos[0]
                    # 算得交点2坐标
                    cut_off_point2 = [cut_off_point2_x, border.Y_MIN]
                # 若是右边界
                elif index[1] == 2:
                    # 计算感知范围圆与边界的交点1的y坐标
                    cut_off_point1_y = np.sqrt(targets[mark].DIS_AVOID_BORDER**2-
                                       (border.X_MAX-targets[mark].pos[0])**2)+targets[mark].pos[1]
                    # 算得交点1坐标
                    cut_off_point1 = [border.X_MAX, cut_off_point1_y]
                    # 计算感知范围圆与边界的交点2的y坐标
                    cut_off_point2_y = -np.sqrt(targets[mark].DIS_AVOID_BORDER**2
                                       -(border.X_MAX-targets[mark].pos[0])**2)+targets[mark].pos[1]
                    # 算得交点2坐标
                    cut_off_point2 = [border.X_MAX, cut_off_point2_y]
                # 若是上边界
                elif index[1] == 3:
                    # 计算感知范围圆与边界的交点1的x坐标
                    cut_off_point1_x = np.sqrt(targets[mark].DIS_AVOID_BORDER**2
                                       -(border.Y_MAX-targets[mark].pos[1])**2)+targets[mark].pos[0]
                    # 算得交点1坐标
                    cut_off_point1 = [cut_off_point1_x, border.Y_MAX]
                    # 计算感知范围圆与边界的交点2的x坐标
                    cut_off_point2_x = -np.sqrt(targets[mark].DIS_AVOID_BORDER**2
                                       -(border.Y_MAX-targets[mark].pos[1])**2)+targets[mark].pos[0]
                    # 算得交点2坐标
                    cut_off_point2 = [cut_off_point2_x, border.Y_MAX]
                # 计算交点1、2相对于目标位置的方向角
                tangent_1 = peri_arctan(cut_off_point1-targets[mark].pos)
                tangent_2 = peri_arctan(cut_off_point2-targets[mark].pos)
                if abs(tangent_1-tangent_2) > PI:
                    bisector = correct((tangent_1+tangent_2)/2+PI)
                    half_ang = (PI-abs(tangent_1-tangent_2)/2)*safety_bor
                else:
                    bisector = (tangent_1+tangent_2)/2
                    half_ang = abs(tangent_1-tangent_2)/2*safety_bor
                # 记录危险角度
                dangerous_range = [correct(bisector-half_ang), correct(bisector+half_ang)]
            elif index[0] == 5:
                # 获取距离危险目标的索引
                i_t = index[1]
                dangerous_range = [correct(danger_direction[i_t]-PI/12*safety_enemy),
                                   correct(danger_direction[i_t]+PI/12*safety_enemy)]
            if dangerous_range is not None:
                dangerous_ranges.append(dangerous_range)
                dangerous_ranges_indexs.append(index)
                dangerous_ranges_dists.append(item[1])
    if len(dangerous_ranges) != 0:
        dangerous_ranges_organized, dangerous_index_organized, dangerous_dists_organized = intervals_merge(dangerous_ranges, dangerous_ranges_indexs, dangerous_ranges_dists)
        target_direction = targets[mark].ori
        find_target = -1
        target_in_danger = False
        nearest_side = -1
        for i in range(len(dangerous_ranges_organized)):
            if dangerous_ranges_organized[i][1] <= 2*PI:
                if dangerous_ranges_organized[i][0] < target_direction < dangerous_ranges_organized[i][1]:
                    target_in_danger = True
                    find_target = i
                    if inc_angle(dangerous_ranges_organized[i][0], target_direction) <= inc_angle(dangerous_ranges_organized[i][1], target_direction):
                        nearest_side = correct(dangerous_ranges_organized[i][0]-0.01)
                    else:
                        nearest_side = correct(dangerous_ranges_organized[i][1]+0.01)
                    break
            else:
                if dangerous_ranges_organized[i][0] < target_direction <= 2*PI or 0 <= target_direction < dangerous_ranges_organized[i][1]-2*PI:
                    target_in_danger = True
                    find_target = i
                    if inc_angle(dangerous_ranges_organized[i][0], target_direction) <= inc_angle(dangerous_ranges_organized[i][1], target_direction-2*PI):
                        nearest_side = correct(dangerous_ranges_organized[i][0]-0.01)
                    else:
                        nearest_side = correct(dangerous_ranges_organized[i][1]+0.01)
                    break
        if target_in_danger:
            theta_target_desired = nearest_side
            # plt.arrow(targets[mark].pos[0],targets[mark].pos[1],1*Cos(theta_target_desired),1*Sin(theta_target_desired),color='k', width=0.05, head_width=0.2, head_length=0.4)
        else:
            # 期望方向角和当前方向角的差
            ori_dif = correct(targets[mark].ori-theta_target_desired)
            # 最大角速度由当前速度和轮子最大转速求出
            w_max = 7.0
            # 若期望方向角和当前方向角的差超出最大角速度则取最大值
            if w_max*TS < ori_dif <= PI:
                ang_vel_target = -w_max
                # 角速度调整到一个周期内
                if ang_vel_target*TS > PI:
                    ang_vel_target -= 2*PI/TS
                elif ang_vel_target*TS < -PI:
                    ang_vel_target += 2*PI/TS
                new_ori = correct(targets[mark].ori+ang_vel_target*TS)
                new_ori_in_danger = False
                for i in range(len(dangerous_ranges_organized)):
                    if (dangerous_ranges_organized[i][1] <= 2*PI and dangerous_ranges_organized[i][0] < new_ori < dangerous_ranges_organized[i][1]) or (dangerous_ranges_organized[i][1] > 2*PI and (dangerous_ranges_organized[i][0] < new_ori <= 2*PI or 0 <= new_ori < dangerous_ranges_organized[i][1]-2*PI)):
                        new_ori_in_danger = True
                        break
                if new_ori_in_danger:
                    theta_target_desired = targets[mark].ori
                # plt.arrow(targets[mark].pos[0],targets[mark].pos[1],1*Cos(theta_target_desired),1*Sin(theta_target_desired),color='b', width=0.05, head_width=0.2, head_length=0.4)
            elif PI < ori_dif < 2*PI-w_max*TS:
                ang_vel_target = w_max
                # 角速度调整到一个周期内
                if ang_vel_target*TS > PI:
                    ang_vel_target -= 2*PI/TS
                elif ang_vel_target*TS < -PI:
                    ang_vel_target += 2*PI/TS
                new_ori = correct(targets[mark].ori+ang_vel_target*TS)
                new_ori_in_danger = False
                for i in range(len(dangerous_ranges_organized)):
                    if (dangerous_ranges_organized[i][1] <= 2*PI and dangerous_ranges_organized[i][0] < new_ori < dangerous_ranges_organized[i][1]) or (dangerous_ranges_organized[i][1] > 2*PI and (dangerous_ranges_organized[i][0] < new_ori <= 2*PI or 0 <= new_ori < dangerous_ranges_organized[i][1]-2*PI)):
                        new_ori_in_danger = True
                        break
                if new_ori_in_danger:
                    theta_target_desired = targets[mark].ori
                # plt.arrow(targets[mark].pos[0],targets[mark].pos[1],1*Cos(theta_target_desired),1*Sin(theta_target_desired),color='b', width=0.05, head_width=0.2, head_length=0.4)
            elif 0 <= ori_dif <= w_max*TS:
                ang_vel_target = (theta_target_desired-targets[mark].ori)/TS
                # 角速度调整到一个周期内
                if ang_vel_target*TS > PI:
                    ang_vel_target -= 2*PI/TS
                elif ang_vel_target*TS < -PI:
                    ang_vel_target += 2*PI/TS
                new_ori = correct(targets[mark].ori+ang_vel_target*TS)
                new_ori = correct(targets[mark].ori+ang_vel_target*TS)
                new_ori_in_danger = False
                for i in range(len(dangerous_ranges_organized)):
                    if (dangerous_ranges_organized[i][1] <= 2*PI and dangerous_ranges_organized[i][0] < new_ori < dangerous_ranges_organized[i][1]) or (dangerous_ranges_organized[i][1] > 2*PI and (dangerous_ranges_organized[i][0] < new_ori <= 2*PI or 0 <= new_ori < dangerous_ranges_organized[i][1]-2*PI)):
                        new_ori_in_danger = True
                        break
                if new_ori_in_danger:
                    theta_target_desired = targets[mark].ori
                # plt.arrow(targets[mark].pos[0],targets[mark].pos[1],1*Cos(theta_target_desired),1*Sin(theta_target_desired),color='b', width=0.05, head_width=0.2, head_length=0.4)
            elif 2*PI-w_max*TS <= ori_dif <= 2*PI:
                ang_vel_target = (theta_target_desired-targets[mark].ori)/TS
                # 角速度调整到一个周期内
                if ang_vel_target*TS > PI:
                    ang_vel_target -= 2*PI/TS
                elif ang_vel_target*TS < -PI:
                    ang_vel_target += 2*PI/TS
                new_ori = correct(targets[mark].ori+ang_vel_target*TS)
                new_ori = correct(targets[mark].ori+ang_vel_target*TS)
                new_ori_in_danger = False
                for i in range(len(dangerous_ranges_organized)):
                    if (dangerous_ranges_organized[i][1] <= 2*PI and dangerous_ranges_organized[i][0] < new_ori < dangerous_ranges_organized[i][1]) or (dangerous_ranges_organized[i][1] > 2*PI and (dangerous_ranges_organized[i][0] < new_ori <= 2*PI or 0 <= new_ori < dangerous_ranges_organized[i][1]-2*PI)):
                        new_ori_in_danger = True
                        break
                if new_ori_in_danger:
                    theta_target_desired = targets[mark].ori
                # plt.arrow(targets[mark].pos[0],targets[mark].pos[1],1*Cos(theta_target_desired),1*Sin(theta_target_desired),color='b', width=0.05, head_width=0.2, head_length=0.4)
            else:
                theta_target_desired = targets[mark].ori
    else:
        dangerous_ranges_organized = []
    # plt.arrow(targets[mark].pos[0],targets[mark].pos[1],2*Cos(theta_target_desired),2*Sin(theta_target_desired),color='r', width=0.05, head_width=0.2, head_length=0.4)
    return vel_target_desired, theta_target_desired, dangerous_ranges_organized


def target_go(targets: list, mob_obss: list, sta_obss: list, irr_obss: list, m_irr_obss: list, rectangle_border: object, t: int, interact: list, EXPANSION1: list, EXPANSION2: list, **kwargs):
    """
    目标运动的主函数，在以上函数的基础上计算出目标下一步运动的速度和角速度

    输入：
        targets: 存放所有目标对象的list
        mob_obss: 存放所有移动障碍物对象的list
        sta_obss: 存放所有固定障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        rectangle_border: 边界对象
        t: 当前仿真步数(单位为step)
        interact: 拓扑矩阵

    输出：
        vel_target: 目标计算得到的当前步的控制输入线速度(单位为m/s)，尚未实际移动
        ang_vel_target: 目标计算得到的当前步的控制输入角速度(单位为rad/s)，尚未实际移动
        t_d_rs: 目标当前步的观察范围内的危险角度范围区间∈[0,2π)
    """
    vel_target, ang_vel_target = np.zeros(TARGET_NUM), np.zeros(TARGET_NUM)
    # 最大角速度
    random_ang_vel_max = np.pi/6
    t_d_rs = []
    for i in range(TARGET_NUM):
        dir_variation = np.zeros(2)
        if targets[i].death:
            t_d_rs.append([])
            continue
        targets[i].attacked = 0
        gap = np.zeros(WOLF_NUM)
        for j in range(WOLF_NUM):
            gap[j] = norm(targets[i].target_to_wolf[j])
            targets[i].attacked += 1 if gap[j] <= targets[i].R_ATTACKED else 0
        if targets[i].attacked >= 4:
            targets[i].death = True
            targets[i].t_death = t
        if min(gap) < targets[i].R_VISION and targets[i].t_ob == 0:
            targets[i].t_ob = t
        if targets[i].t_ob != 0:
            hunter = []
            for j in range(WOLF_NUM):
                if gap[j] < targets[i].R_VISION:
                    interact[5][j] = 1
                    hunter.append(j)
            if len(hunter) != 0:
                danger_direction = []
                for k in hunter:
                    danger_direction.append(peri_arctan(targets[i].target_to_wolf[k]))
                    dir_variation += -targets[i].target_to_wolf[k]
                variation = 0.9*(-1/(1+exp(0.25*(100+targets[i].t_ob-t)))+2)
                theta = peri_arctan(dir_variation)
            else:
                variation = (-1/(1+exp(0.25*(100+targets[i].t_ob-t)))+2)/(-1/(1+exp(0.25*(100+targets[i].t_ob-t+1)))+2)*targets[i].vel
                theta = targets[i].ori
        else:
            variation = np.random.uniform(0, targets[i].vel_max)
            w = np.random.uniform(-random_ang_vel_max, random_ang_vel_max)
            theta = correct(targets[i].ori+w)

        vel_target_desired, theta_target_desired, t_d_r = target_avoid_obs(t, i, variation, theta, targets, mob_obss, sta_obss, irr_obss, m_irr_obss, rectangle_border, EXPANSION1, EXPANSION2)
        t_d_rs.append(t_d_r)
        vel_target[i], ang_vel_target[i] = saturator(targets[i].ori, targets[i].vel_max, targets[i].ang_vel_max, vel_target_desired, theta_target_desired)
    return vel_target, ang_vel_target, t_d_rs
