# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义Record函数用来记录围捕机器人和目标的速度、角速度、能量消耗，并用PlotData函数和Axplot函数把数据变化曲线图画出来。


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
from typing import List
from model import Robot, Target
from utils.params import WOLF_NUM, TARGET_NUM


# 设置字体
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def record_data(wolves: List[Robot], targets: List[Target], t: int, interact: List, **kwargs):
    """
    记录个体的速度和角速度

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        t: 当前仿真的步数(单位为step)
        interact: 当前步拓扑矩阵

    输出：
        pos_targets: 当前步所有目标的全局坐标(单位为m)
        vel_targets: 当前步所有目标的线速度(单位为m/s)
        ang_vel_targets: 当前步所有目标的角速度(单位为rad/s)
        energy_targets: 当前步所有目标的能量消耗(单位为J)
        pos_wolves: 当前步所有围捕机器人的全局坐标(单位为m)
        vel_wolves: 当前步所有围捕机器人的线速度(单位为m/s)
        ang_vel_wolves: 当前步所有围捕机器人的角速度(单位为rad/s)
        energy_wolves: 当前步所有围捕机器人的能量消耗(单位为J)
        interact: 当前步拓扑矩阵
    """
    pos_targets = []
    vel_targets = []
    ang_vel_targets = []
    energy_targets = []
    pos_wolves = []
    vel_wolves = []
    ang_vel_wolves = []
    energy_wolves = []
    for target in targets:
        pos_targets.append(target.pos.tolist())
        vel_targets.append(target.vel)
        ang_vel_targets.append(target.ang_vel)
        energy_targets.append(target.energy)
    for wolf in wolves:
        pos_wolves.append(wolf.pos.tolist())
        vel_wolves.append(wolf.vel)
        ang_vel_wolves.append(wolf.ang_vel)
        energy_wolves.append(wolf.energy)
    return pos_targets, vel_targets, ang_vel_targets, energy_targets, pos_wolves, vel_wolves, ang_vel_wolves, energy_wolves, interact


def plot_data(var: str, vel_wolves: List, ang_vel_wolves: List, energy_wolves: List, vel_targets: List, ang_vel_targets: List, energy_targets: List, **kwargs) -> None:
    """
    为运动数据提供画图界面

    输入：
        var: 判断本次函数要画的是速度、角速度还是能量消耗曲线，只能是'v','w','E'
        vel_wolves: 围捕机器人当前步的速度(单位为m/s)
        ang_vel_wolves: 围捕机器人当前步的角速度(单位为rad/s)
        energy_wolves: 围捕机器人当前步的累计能量消耗(单位为J)
        vel_targets: 目标当前步的速度(单位为m/s)
        ang_vel_targets: 目标当前步的角速度(单位为rad/s)
        energy_targets: 目标当前步的累计能量消耗(单位为J)
    """
    # 打开绘图窗口
    plt.figure(figsize=(8, 6), constrained_layout=True)
    # 绘图窗口中轴上刻度的字体控制
    plt.xticks(fontsize=14, fontfamily="Times New Roman")
    plt.yticks(fontsize=14, fontfamily="Times New Roman")
    if var == 'v':
        # 画出目标的速度曲线
        plt.plot(np.arange(0, len(vel_targets), 1), vel_targets)
        # 绘图窗口中标题的内容和字体控制
        plt.title(u'目标的速度曲线', fontsize=14, fontfamily='SimSun')
        # 绘图窗口中y轴标签的内容和字体控制
        plt.ylabel(var+'/m·'+r'$s^{-1}$', fontsize=14, fontfamily="Times New Roman")
    elif var == 'w':
        # 画出目标的角速度曲线
        plt.plot(np.arange(0, len(ang_vel_targets), 1), ang_vel_targets)
        # 绘图窗口中标题的内容和字体控制
        plt.title(u'目标的角速度曲线', fontsize=14, fontfamily='SimSun')
        # 绘图窗口中y轴标签的内容和字体控制
        plt.ylabel(var+'/rad·'+r'$s^{-1}$', fontsize=14, fontfamily="Times New Roman")
    elif var == 'E':
        # 画出目标的能量消耗曲线
        plt.plot(np.arange(0, len(energy_targets), 1), energy_targets, color='r', linewidth=2.5)
        # 绘图窗口中标题的内容和字体控制
        plt.title(u'目标的能量消耗曲线', fontsize=14, fontfamily='SimSun')
        # 绘图窗口中y轴标签的内容和字体控制
        plt.ylabel(var+'/J', fontsize=14, fontfamily="Times New Roman")
    else:
        raise ValueError(f'Input error: var must be v or w. But now var=={var}.')
    # 绘图窗口中x轴标签的内容和字体控制
    plt.xlabel('t/step', fontsize=14, fontfamily="Times New Roman")
    plt.show()

    # 打开绘图窗口
    fig = plt.figure(figsize=(12, 8))

    # 以下代码主要为了实现将5张曲线图放在同一个绘图窗口中
    gs2 = gridspec.GridSpec(1, 3)

    ax3 = fig.add_subplot(gs2[0])
    ax4 = fig.add_subplot(gs2[1])
    ax5 = fig.add_subplot(gs2[2])

    plot_ax(ax3, 2, var, vel_wolves, ang_vel_wolves, energy_wolves)
    plot_ax(ax4, 3, var, vel_wolves, ang_vel_wolves, energy_wolves)
    plot_ax(ax5, 4, var, vel_wolves, ang_vel_wolves, energy_wolves)

    gs2.tight_layout(fig, rect=[0, 0, 1, 0.5])

    gs1 = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])

    plot_ax(ax1, 0, var, vel_wolves, ang_vel_wolves, energy_wolves)
    plot_ax(ax2, 1, var, vel_wolves, ang_vel_wolves, energy_wolves)

    gs1.tight_layout(fig, rect=[0, 0.5, 1, 1], h_pad=0, w_pad=2.0)

    left = max(gs1.left, gs2.left)
    right = min(gs1.right, gs2.right)
    gs1.update(left=left, right=right, top=gs1.top+0.015)
    gs2.update(left=left, right=right, bottom=gs2.bottom-0.015)

    plt.show()


def plot_ax(ax: object, n: int, var: str, vel_wolves: List, ang_vel_wolves: List, energy_wolves: List) -> None:
    """
    画出机器人的速度曲线

    输入:
        ax: 传入的matplotlib对象，用于画图
        n: 围捕机器人序号
        var: 判断本次函数要画的是速度、角速度还是能量消耗曲线，只能是'v','w','E'
        vel_wolves: 围捕机器人当前步的速度(单位为m/s)
        ang_vel_wolves: 围捕机器人当前步的角速度(单位为rad/s)
        energy_wolves: 围捕机器人当前步的累计能量消耗(单位为J)
    """
    vel_wolves_n, ang_vel_wolves_n, energy_wolves_n = [], [], []
    for i in range(len(vel_wolves)):
        vel_wolves_n.append(vel_wolves[i][n])
        ang_vel_wolves_n.append(ang_vel_wolves[i][n])
        energy_wolves_n.append(energy_wolves[i][n])
    # 设定坐标轴刻度字体
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # 画出狼的速度曲线
    if var == 'v':
        ax.plot(np.arange(0, len(vel_wolves_n), 1), vel_wolves_n)
        # 设定标题字体
        ax.set_title(f'狼{n+1}的速度曲线', fontsize=14, fontfamily='SimSun')
        ax.set_ylabel(var+'/m·'+r'$s^{-1}$', fontsize=14, fontfamily="Times New Roman")
    # 画出狼的角速度曲线
    elif var == 'w':
        ax.plot(np.arange(0, len(ang_vel_wolves_n), 1), ang_vel_wolves_n)
        # 设定标题字体
        ax.set_title(f'狼{n+1}的角速度曲线', fontsize=14, fontfamily='SimSun')
        ax.set_ylabel(var+'/rad·'+r'$s^{-1}$', fontsize=14, fontfamily="Times New Roman")
    # 画出狼的能量消耗曲线
    elif var == 'E':
        ax.plot(np.arange(0, len(energy_wolves_n), 1), energy_wolves_n, linewidth=2.5)
        # 设定标题字体
        ax.set_title(f'狼{n+1}的能量消耗曲线', fontsize=14, fontfamily='SimSun')
        ax.set_ylabel(var+'/J', fontsize=14, fontfamily="Times New Roman")
    else:
        raise ValueError(f'Input error: var must be v or w. But now var=={var}.')
    # 设定坐标轴标签字体
    ax.set_xlabel('t/step', fontsize=14, fontfamily="Times New Roman")
