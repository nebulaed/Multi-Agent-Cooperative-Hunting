# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义了画图函数PlotAll，用于在每一步时将当前步的情况呈现在绘图窗口中。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import rcParams, patches
from utils.math_func import norm
from utils.params import WOLF_NUM, TARGET_NUM, PI
from matplotlib.font_manager import FontProperties  # 字体属性管理器


# 设置字体
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# 设置字体及其大小，修改字体时替换字体的路径即可
font1 = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=14)


def plot_all(data: dict, wolves: list, targets: list, sta_obss: list, mob_obss: list, irr_obss: list, m_irr_obss: list, rectangle_border: object, t: int, w_d_range: list, v_vector: np.ndarray, t_d_range: list, **kwargs) -> None:
    """
    画图函数，将仿真中每一步时的位置情况呈现在绘图窗口中

    输入：
        data: 存储数据的字典
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        rectangle_border: 边界对象
        t: 当前仿真步数(单位为step)
        w_d_range: 存放matplotlib扇形对象的list，用于画图
        v_vector: 存放围捕机器人的期望速度矢量，用于画图方便debug
        t_d_range: 存放matplotlib扇形对象的list，用于画图
    """
    ax = plt.gca()
    wolves_wedges, targets_wedges = [], []
    for i in range(WOLF_NUM):
        wolves[i].plot_robot()
        plt.text(wolves[i].pos[0], wolves[i].pos[1], i, fontproperties=font1)
        pos_wolvesx, pos_wolvesy = [], []
        for step in range(t):
            pos_wolvesx.append(data['pos_wolves'][step][i][0])
            pos_wolvesy.append(data['pos_wolves'][step][i][1])
        plt.plot(pos_wolvesx, pos_wolvesy, ls='--', color='b', lw=0.8)
        for j in range(len(w_d_range[i])):
            wedge = patches.Wedge(wolves[i].pos, wolves[i].R_VISION, w_d_range[i][j]
                                  [0]/PI*180, w_d_range[i][j][1]/PI*180, ec="red", lw=1, ls='-')
            wolves_wedges.append(wedge)

    for i in range(TARGET_NUM):
        targets[i].plot_target()
        pos_targetsx, pos_targetsy = [], []
        for step in range(t):
            pos_targetsx.append(data['pos_targets'][step][i][0])
            pos_targetsy.append(data['pos_targets'][step][i][1])
        plt.plot(pos_targetsx, pos_targetsy, ls='--', color='r', lw=0.8)
        for j in range(len(t_d_range[i])):
            wedge = patches.Wedge(targets[i].pos, targets[i].R_VISION, t_d_range[i]
                                  [j][0]/PI*180, t_d_range[i][j][1]/PI*180, ec="none")
            targets_wedges.append(wedge)

    # 画出固定障碍物和移动障碍物
    sta_cir, mob_cir, irr_cir, m_irr_cir = [], [], [], []
    for sta_obs in sta_obss:
        sta_cir.append(sta_obs.plot_obs())
    for mob_obs in mob_obss:
        mob_cir.append(mob_obs.plot_obs())
    for irr_obs in irr_obss:
        irr_dist = np.zeros(WOLF_NUM)
        for i in range(WOLF_NUM):
            irr_dist[i] = norm(wolves[i].pos-irr_obs.pos)
        irr_cir.append(irr_obs.plot_irr_obs())
    for m_irr_obs in m_irr_obss:
        m_irr_dist = np.zeros(WOLF_NUM)
        for i in range(WOLF_NUM):
            m_irr_dist[i] = norm(wolves[i].pos-m_irr_obs.pos)
        m_irr_cir.append(m_irr_obs.plot_irr_obs())
    debug_car = []
    for wolf in wolves:
        if wolf.dangerObsFlag['obs'][t] == 1:
            debug_car.append(wolf.plot_agent2())

    # 保持横纵坐标比例尺一致
    plt.xlim(rectangle_border.X_MIN - 0.5, rectangle_border.X_MAX + 0.5)
    plt.ylim(rectangle_border.Y_MIN - 0.5, rectangle_border.Y_MAX + 0.5)
    ax.set_aspect('equal', adjustable='box')
    if wolves_wedges != None:
        collection = PatchCollection(wolves_wedges, alpha=0.5, ls='-', lw=3, edgecolor='k')
        collection.set_facecolor('#AFEEEE')
        ax.add_collection(collection)
    if targets_wedges != None:
        collection = PatchCollection(targets_wedges, alpha=0.5)
        collection.set_facecolor('#FFD39B')
        ax.add_collection(collection)
    # 画出边界
    ax.add_patch(rectangle_border.plot_border())
    for item in sta_cir+mob_cir:
        ax.add_artist(item)
    for item in irr_cir+m_irr_cir+debug_car:
        ax.add_patch(item)

    plt.title(f'仿真步数t={t}', fontsize=14, fontfamily='SimSun')
    # 设置刻度标签字体
    plt.xticks(fontsize=14, fontfamily="Times New Roman")
    plt.yticks(fontsize=14, fontfamily="Times New Roman")
    # 设置x/y轴标签字体
    plt.xlabel(r'$x$'+'/m', fontsize=14, fontfamily="Times New Roman")
    plt.ylabel(r'$y$'+'/m', fontsize=14, fontfamily="Times New Roman")

    # 暂停时间
    plt.pause(0.001)
