# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义的函数主要是实现围捕机器人的避碰。

# AroundWolves: 围捕机器人的避障算法


import numpy as np
from utils.math_func import correct,peri_arctan,arcsin,norm,inc_angle,intervals_merge
from utils.params import WOLF_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM, PI


def robot_avoid_obs(t: int, mark: int, vel_wolf_desired: float, theta_wolf_desired: float, my_t: int, wolves: list, mob_obss: list, sta_obss: list, irr_obss: list, m_irr_obss: list, border: object, D_DANGER: float, D_DANGER_W: float,dict_d_obs: dict):
    """
    围捕机器人的避障算法

    输入：
        @param t: 当前仿真步数(单位为step)
        @param mark: 围捕机器人序号
        @param vel_wolf_desired: 在不考虑避障的情况下围捕机器人期望速度(单位为m/s)
        @param theta_wolf_desired: 在不考虑避障的情况下围捕机器人期望速度方向∈[0,2π)
        @param my_t: 存放围捕机器人选择的目标list
        @param wolves: 存放所有围捕机器人对象的list
        @param mob_obss: 存放所有移动障碍物对象的list
        @param sta_obss: 存放所有固定障碍物对象的list
        @param irr_obss: 存放所有不规则障碍物对象的list
        @param m_irr_obss: 存放所有移动不规则障碍物对象的list
        @param border: 边界对象
        @param D_DANGER: 围捕机器人启动紧急避障的距离(单位为m)
        @param D_DANGER_W: 围捕机器人避免互碰启动紧急避障的距离(单位为m)

    输出：
        @return vel_wolf_desired: 考虑避障的情况下围捕机器人的期望速度(单位为m/s)
        @return theta_wolf_desired: 考虑避障的情况下围捕机器人期望速度方向∈[0,2π)
        @return dangerous_ranges_organized: 围捕机器人的观察范围内的危险角度范围区间∈[0,2π)
    """
    # 目标受攻击数预登记为0
    attack = 0
    for i in range(len(wolves)):
        # 若个体目标间距小于受攻击距离则目标受攻击数+1
        attack += 1 if norm(wolves[i].wolf_to_target[my_t]) <= 0.8 else 0
    # 若个体与目标的距离小于0.6且有超过4个机器人接近目标
    if (norm(wolves[mark].wolf_to_target[my_t]) < 0.8 and attack >= 4):
        return vel_wolf_desired, theta_wolf_desired, []
    if len(dict_d_obs) == 1:
        key, = dict_d_obs
        if key[0] == 4 and norm(wolves[mark].wolf_to_target[my_t]) < 0.8:
            return vel_wolf_desired, theta_wolf_desired, []
        safety_bor, safety_mob, safety_sta, safety_m_irr = 1.6, 2.4, 2.4, 3.2
    elif len(dict_d_obs) == 2:
        safety_bor, safety_mob, safety_sta, safety_m_irr = 1.3, 1.8, 1.8, 2.0
    elif len(dict_d_obs) >= 3:
        safety_bor, safety_mob, safety_sta, safety_m_irr = 1.2, 1.6, 1.6, 1.8
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
                psi1 = peri_arctan(wolves[mark].wolf_to_m_obs[i_m])
                # 移动障碍物中心连线与切线的夹角
                Delta1 = arcsin(mob_obss[i_m].R/(dist_m+mob_obss[i_m].R))*safety_mob
                # 记录危险角度
                dangerous_range = [correct(psi1-Delta1), correct(psi1+Delta1)]
            # 若是固定障碍物
            elif index[0] == 1:
                i_s = index[1]
                dist_s = item[1]
                # 计算个体到固定障碍物的方向角
                psi2 = peri_arctan(wolves[mark].wolf_to_s_obs[i_s])
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
                    if norm(point-wolves[mark].pos) < wolves[mark].AVOID_DIST:
                        psi_point = peri_arctan(point-wolves[mark].pos)
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
                    if norm(point-wolves[mark].pos) < wolves[mark].AVOID_DIST:
                        psi_point = peri_arctan(point-wolves[mark].pos)
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
                    cut_off_point1_y = np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.X_MIN-wolves[mark].pos[0])**2)+wolves[mark].pos[1]
                    # 算得交点1坐标
                    cut_off_point1 = [border.X_MIN, cut_off_point1_y]
                    # 计算感知范围圆与边界的交点2的y坐标
                    cut_off_point2_y = -np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.X_MIN-wolves[mark].pos[0])**2)+wolves[mark].pos[1]
                    # 算得交点2坐标
                    cut_off_point2 = [border.X_MIN, cut_off_point2_y]
                # 若是下边界
                elif index[1] == 1:
                    # 计算感知范围圆与边界的交点1的x坐标
                    cut_off_point1_x = np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.Y_MIN-wolves[mark].pos[1])**2)+wolves[mark].pos[0]
                    # 算得交点1坐标
                    cut_off_point1 = [cut_off_point1_x, border.Y_MIN]
                    # 计算感知范围圆与边界的交点2的x坐标
                    cut_off_point2_x = -np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.Y_MIN-wolves[mark].pos[1])**2)+wolves[mark].pos[0]
                    # 算得交点2坐标
                    cut_off_point2 = [cut_off_point2_x, border.Y_MIN]
                # 若是右边界
                elif index[1] == 2:
                    # 计算感知范围圆与边界的交点1的y坐标
                    cut_off_point1_y = np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.X_MAX-wolves[mark].pos[0])**2)+wolves[mark].pos[1]
                    # 算得交点1坐标
                    cut_off_point1 = [border.X_MAX, cut_off_point1_y]
                    # 计算感知范围圆与边界的交点2的y坐标
                    cut_off_point2_y = -np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.X_MAX-wolves[mark].pos[0])**2)+wolves[mark].pos[1]
                    # 算得交点2坐标
                    cut_off_point2 = [border.X_MAX, cut_off_point2_y]
                # 若是上边界
                elif index[1] == 3:
                    # 计算感知范围圆与边界的交点1的x坐标
                    cut_off_point1_x = np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.Y_MAX-wolves[mark].pos[1])**2)+wolves[mark].pos[0]
                    # 算得交点1坐标
                    cut_off_point1 = [cut_off_point1_x, border.Y_MAX]
                    # 计算感知范围圆与边界的交点2的x坐标
                    cut_off_point2_x = -np.sqrt(wolves[mark].DIS_AVOID_BORDER**2-(border.Y_MAX-wolves[mark].pos[1])**2)+wolves[mark].pos[0]
                    # 算得交点2坐标
                    cut_off_point2 = [cut_off_point2_x, border.Y_MAX]
                # 计算目标到边界最近点的方向角
                # psi3 = PeriArctan(wolves[mark].nearest-wolves[mark].pos)
                # 计算交点1、2相对于围捕机器人位置的方向角
                tangent_1 = peri_arctan(cut_off_point1-wolves[mark].pos)
                tangent_2 = peri_arctan(cut_off_point2-wolves[mark].pos)
                if abs(tangent_1-tangent_2) > PI:
                    bisector = correct((tangent_1+tangent_2)/2+PI)
                    half_ang = (PI-abs(tangent_1-tangent_2)/2)*safety_bor
                else:
                    bisector = (tangent_1+tangent_2)/2
                    half_ang = abs(tangent_1-tangent_2)/2*safety_bor
                # 记录危险角度
                dangerous_range = [correct(bisector-half_ang), correct(bisector+half_ang)]
            # 若是同伴
            elif index[0] == 5:
                i_p = index[1]
                psi5 = peri_arctan(-wolves[mark].wolves_dif[i_p])
                Delta5 = PI/12
                dangerous_range = [correct(psi5-Delta5), correct(psi5+Delta5)]
            if dangerous_range is not None:
                dangerous_ranges.append(dangerous_range)
                dangerous_ranges_indexs.append(index)
                dangerous_ranges_dists.append(item[1])
    # 若至少有一个危险角度范围
    if len(dangerous_ranges) != 0:
        dangerous_ranges_organized, dangerous_index_organized, dangerous_dists_organized = intervals_merge(dangerous_ranges, dangerous_ranges_indexs, dangerous_ranges_dists)
        for i in range(len(dangerous_ranges_organized)):
            if dangerous_index_organized[i][0] != 1:
                # 若与障碍物的距离小于危险距离, 打上标签
                if 0 < dangerous_dists_organized[i] < D_DANGER:
                    wolves[mark].danger[t] = 1
                else:
                    wolves[mark].danger[t] = 0
            # 若障碍物距离太近
            if wolves[mark].danger[t] == 1 or wolves[mark].danger[t-1] == 1 or wolves[mark].danger[t-2] == 1:
                # 两个角度的中值
                bisector = (dangerous_ranges_organized[i][0]+dangerous_ranges_organized[i][1])/2
                # 期望方向为危险角度范围扇形的角平分线的反向
                theta_wolf_desired = correct(bisector-PI)
                # 若目前车头方向与期望速度方向相差大于π/2，则期望速度为0
                if abs(wolves[mark].ori-theta_wolf_desired) > PI/2:
                    vel_wolf_desired = 0
                # 若目前车头方向与期望速度方向相差小于π/2，则期望速度为vel_max
                else:
                    avoid_v = wolves[mark].vel_max
                    vel_wolf_desired = avoid_v
                return vel_wolf_desired, theta_wolf_desired, dangerous_ranges_organized

        # 期望方向
        target_direction = theta_wolf_desired
        # 初始化期望方向是否在危险角度范围内的标签
        target_in_danger = False
        # 检查所有的危险角度范围
        for i in range(len(dangerous_ranges_organized)):
            # 若危险角度区间的右界小于等于2π
            if dangerous_ranges_organized[i][1] <= 2*PI:
                # 若危险角度区间的左界<期望方向<危险角度区间的右界
                if dangerous_ranges_organized[i][0] < target_direction < dangerous_ranges_organized[i][1]:
                    # 则标签为True
                    target_in_danger = True
                    # 若危险角度区间左界与期望方向的夹角小于危险角度区间右界与期望方向的夹角
                    if inc_angle(dangerous_ranges_organized[i][0], target_direction) <= inc_angle(dangerous_ranges_organized[i][1], target_direction):
                        # 则最近的边为危险角度区间左界
                        nearest_side = correct(dangerous_ranges_organized[i][0]-0.01)
                    # 若危险角度区间左界与期望方向的夹角大于危险角度区间右界与期望方向的夹角
                    else:
                        # 则最近的边为危险角度区间右界
                        nearest_side = correct(dangerous_ranges_organized[i][1]+0.01)
                    break
                
            # 若危险角度区间的右界大于2π
            else:
                # 若危险角度区间的左界<期望方向<2π或0<=期望方向<危险角度区间的右界-2π
                if dangerous_ranges_organized[i][0] < target_direction <= 2*PI or 0 <= target_direction < dangerous_ranges_organized[i][1]-2*PI:
                    # 则标签为True
                    target_in_danger = True
                    # 若危险区间左界与期望方向的夹角小于危险区间右界-2π与期望方向的夹角 
                    if inc_angle(dangerous_ranges_organized[i][0], target_direction) <= inc_angle(dangerous_ranges_organized[i][1], target_direction-2*PI):
                        # 则最近的边为危险角度区间左界
                        nearest_side = correct(dangerous_ranges_organized[i][0]-0.01)
                    # 若危险区间左界与期望方向的夹角大于危险区间右界-2π与期望方向的夹角 
                    else:
                        # 则最近的边为危险角度区间右界
                        nearest_side = correct(dangerous_ranges_organized[i][1]+0.01)
                    break
        # 若期望方向在危险范围内
        if target_in_danger:
            # 期望方向改为最近边
            theta_wolf_desired = nearest_side
        else:
            theta_wolf_desired = target_direction
    # 若无危险角度范围
    else:
        dangerous_ranges_organized = []
    return vel_wolf_desired, theta_wolf_desired, dangerous_ranges_organized


def robot_avoid_collision(t: int,mark: int,vel_wolf_desired: float,theta_wolf_desired: float,my_t: int,wolves: list,mob_obss: list,sta_obss: list,irr_obss: list,m_irr_obss: list,border: object,D_DANGER: float,D_DANGER_W: float):
    """
    围捕机器人的避碰算法
    """

    wolves[mark].avoidCollisionFlag[t] = 0
    wolves[mark].avoidCollisionFlag2[t] = 0
    danger_w3 = []
    # 以下部分为围捕机器人之间紧急避障
    for i in range(WOLF_NUM):
        if i != mark:
            if 0<norm(wolves[mark].wolves_dif[i])<D_DANGER_W and 0<=inc_angle(peri_arctan(-wolves[mark].wolves_dif[i]),wolves[mark].ori)<PI/2:
                wolves[mark].danger_w[t,i] = 1
            else:
                wolves[mark].danger_w[t,i] = 0
            if 0<norm(wolves[mark].wolves_dif[i])<D_DANGER_W:
                wolves[mark].danger_w2[t,i] = 1
            else:
                wolves[mark].danger_w2[t,i] = 0
            if 0<norm(wolves[mark].wolves_dif[i])<1.0:
                danger_w3.append(i)
    danger_partner1 = np.where(wolves[mark].danger_w[t] == 1)[0]
    if t>1:
        danger_partner2 = np.where(wolves[mark].danger_w[t-2] == 1)[0]
    danger_partner3 = np.where(wolves[mark].danger_w2[t] == 1)[0]

    direction2 = np.zeros(2)
    # 将所有和障碍物的距离按类别分行存储在d_obs中
    d_obs = [[] for _ in range(6)]
    for item in wolves[mark].danger_m:
        d_obs[0].append(wolves[mark].d_m[item])
    for item in wolves[mark].danger_s:
        d_obs[1].append(wolves[mark].d_s[item])
    for item in wolves[mark].danger_ir:
        d_obs[2].append(wolves[mark].d_ir[item])
    for item in wolves[mark].danger_m_ir:
        d_obs[3].append(wolves[mark].d_m_ir[item])
    for item in wolves[mark].danger_border:
        d_obs[4].append(wolves[mark].border_d[item])
    for item in danger_w3:
        d_obs[5].append(norm(wolves[mark].wolves_dif[item]))
    # 用字典同时记录障碍物与个体的距离和障碍物的索引
    dict_d_obs = {}
    for i in range(len(d_obs)):
        for j in range(len(d_obs[i])):
            if i == 0:
                dict_d_obs[(i, wolves[mark].danger_m[j])] = d_obs[i][j]
            elif i == 1:
                dict_d_obs[(i, wolves[mark].danger_s[j])] = d_obs[i][j]
            elif i == 2:
                dict_d_obs[(i, wolves[mark].danger_ir[j])] = d_obs[i][j]
            elif i == 3:
                dict_d_obs[(i, wolves[mark].danger_m_ir[j])] = d_obs[i][j]
            elif i == 4:
                dict_d_obs[(i, wolves[mark].danger_border[j])] = d_obs[i][j]
            elif i == 5:
                dict_d_obs[(i, danger_w3[j])] = d_obs[i][j]
    # 按照障碍物的距离将其和对应的索引进行排序
    dict_obs = sorted(dict_d_obs.items(), key=lambda item: item[1])
    import matplotlib.pyplot as plt
    if len(dict_obs) != 0:
        for item in dict_obs:
            if 0 < item[1] < D_DANGER:
                if item[0][0] == 0:
                    wolves[mark].dangerObsFlag['mob_obss'][t, item[0][1]] = 1
                elif item[0][0] == 1:
                    wolves[mark].dangerObsFlag['sta_obss'][t, item[0][1]] = 1
                elif item[0][0] == 2:
                    wolves[mark].dangerObsFlag['irr_obss'][t, item[0][1]] = 1
                elif item[0][0] == 3:
                    wolves[mark].dangerObsFlag['m_irr_obss'][t, item[0][1]] = 1
    for i in range(S_OBS_NUM):
        if wolves[mark].dangerObsFlag['sta_obss'][t, i] == 1 or (wolves[mark].dangerObsFlag['sta_obss'][t-1, i] == 1 and t>0) or (wolves[mark].dangerObsFlag['sta_obss'][t-2, i] == 1 and t>1):
            wolves[mark].dangerObsFlag['obs'][t] = 1
            plt.plot(sta_obss[i].pos[0],sta_obss[i].pos[1],'ko')
            direction2 += 1/-(wolves[mark].wolf_to_s_obs[i]/norm(wolves[mark].wolf_to_s_obs[i])*wolves[mark].d_s[i])
    for i in range(M_OBS_NUM):
        if wolves[mark].dangerObsFlag['mob_obss'][t, i] == 1 or (wolves[mark].dangerObsFlag['mob_obss'][t-1, i] == 1 and t>0) or (wolves[mark].dangerObsFlag['mob_obss'][t-2, i] == 1 and t>1):
            wolves[mark].dangerObsFlag['obs'][t] = 1
            plt.plot(mob_obss[i].pos[0],mob_obss[i].pos[1],'ko')
            direction2 += 1/-(wolves[mark].wolf_to_m_obs[i]/norm(wolves[mark].wolf_to_m_obs[i])*wolves[mark].d_m[i])
    for i in range(IRR_OBS_NUM):
        if wolves[mark].dangerObsFlag['irr_obss'][t, i] == 1 or (wolves[mark].dangerObsFlag['irr_obss'][t-1, i] == 1 and t>0) or (wolves[mark].dangerObsFlag['irr_obss'][t-2, i] == 1 and t>1):
            wolves[mark].dangerObsFlag['obs'][t] = 1
            plt.plot(irr_obss[i].pos[0],irr_obss[i].pos[1],'ko')
            direction2 += 1/-(wolves[mark].wolf_to_irr_obs[i])
    for i in range(M_IRR_OBS_NUM):
        if wolves[mark].dangerObsFlag['m_irr_obss'][t, i] == 1 or (wolves[mark].dangerObsFlag['m_irr_obss'][t-1, i] == 1 and t>0) or (wolves[mark].dangerObsFlag['m_irr_obss'][t-2, i] == 1 and t>1):
            wolves[mark].dangerObsFlag['obs'][t] = 1
            plt.plot(m_irr_obss[item[0][1]].pos[0],m_irr_obss[item[0][1]].pos[1],'ko')
            direction2 += 1/-(wolves[mark].wolf_to_m_irr_obs[item[0][1]])

    if len(danger_partner1)!= 0:
        wolves[mark].avoidCollisionFlag[t] = 1
        dangerous_ranges_organized = []
        direction = np.zeros(2)
        if t>1:
            for index in danger_partner1:
                if index in danger_partner2:
                    wolves[mark].avoidCollisionFlag2[t] = 1
        if wolves[mark].avoidCollisionFlag2[t] == 1 or wolves[mark].avoidCollisionFlag2[t-1] == 1 or wolves[mark].avoidCollisionFlag2[t-2] == 1:
            for index in danger_partner3:
                direction += 1/wolves[mark].wolves_dif[index]
            direction += direction2
            theta_wolf_desired = peri_arctan(direction)
            if abs(wolves[mark].ori-theta_wolf_desired)>PI/2:
                vel_wolf_desired = 0
            else:
                vel_wolf_desired = wolves[mark].vel_max
        else:
            if wolves[mark].dangerObsFlag['obs'][t] == 1:
                theta_wolf_desired = peri_arctan(direction2)
                if abs(wolves[mark].ori-theta_wolf_desired)>PI/2:
                    vel_wolf_desired = 0
                else:
                    vel_wolf_desired = wolves[mark].vel_max
            else:
                vel_wolf_desired = 0
    elif len(danger_partner3)!= 0:
        direction = np.zeros(2)
        if wolves[mark].avoidCollisionFlag2[t] == 1 or wolves[mark].avoidCollisionFlag2[t-1] == 1 or wolves[mark].avoidCollisionFlag2[t-2] == 1:
            dangerous_ranges_organized = []
            for index in danger_partner3:
                direction += 1/wolves[mark].wolves_dif[index]
            direction += direction2
            theta_wolf_desired = peri_arctan(direction)
            if abs(wolves[mark].ori-theta_wolf_desired)>PI/2:
                vel_wolf_desired = 0
            else:
                vel_wolf_desired = wolves[mark].vel_max
        else:
            vel_wolf_desired, theta_wolf_desired, dangerous_ranges_organized = robot_avoid_obs(t, mark, vel_wolf_desired, theta_wolf_desired, my_t, wolves, mob_obss, sta_obss, irr_obss, m_irr_obss, border, D_DANGER, D_DANGER_W, dict_d_obs)
    elif wolves[mark].dangerObsFlag['obs'][t] == 1:
        dangerous_ranges_organized = []
        theta_wolf_desired = peri_arctan(direction2)
        if abs(wolves[mark].ori-theta_wolf_desired)>PI/2:
            vel_wolf_desired = 0
        else:
            vel_wolf_desired = wolves[mark].vel_max
    else:
        vel_wolf_desired, theta_wolf_desired, dangerous_ranges_organized = robot_avoid_obs(t, mark, vel_wolf_desired, theta_wolf_desired, my_t, wolves, mob_obss, sta_obss, irr_obss, m_irr_obss, border, D_DANGER, D_DANGER_W, dict_d_obs)
    return vel_wolf_desired, theta_wolf_desired, dangerous_ranges_organized
        
            