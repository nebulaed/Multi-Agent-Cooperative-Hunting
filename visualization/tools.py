# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件主要为保存下来的仿真结果的绘图复现提供工具函数。

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import patches
from utils.math_func import cos, sin
from utils.params import PI


def draw_circle(ax, pos_x: float, pos_y: float, r: float, ec: str, ls: str = '-') -> None:
    """画圆

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param pos_x: 圆心x坐标(单位为m)
        @param pos_y: 圆心y坐标(单位为m)
        @param r: 圆的半径(单位为m)
        @param ec: 线条颜色
        @param ls: 线条线型，默认为实线
    """
    Circle = plt.Circle((pos_x, pos_y), r,
                        color='black', fill=False, linewidth=1.0, ls = ls, ec = ec)
    ax.add_patch(Circle)


def draw_robot(ax, pos_x: float, pos_y: float, theta: float, DISPLAYHEIGHT: float, DISPLAYBASE: float, R_VISION: float):
    """画机器人

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param pos_x: 机器人x坐标(单位为m)
        @param pos_y: 机器人y坐标(单位为m)
        @param theta: 机器人车头方向(单位为rad)
        @param DISPLAYHEIGHT: 机器人三角形的高(单位为m)
        @param DISPLAYBASE: 机器人三角形的底边长(单位为m)
        @param R_VISION: 机器人的观察范围(单位为m)
    """
    vertex_x0, vertex_x1, vertex_x2, vertex_y0, vertex_y1, vertex_y2 = compute_vertex(pos_x, pos_y, theta, DISPLAYHEIGHT, DISPLAYBASE)
    polyvertexs = [[vertex_x0,vertex_y0],[vertex_x1,vertex_y1],[vertex_x2,vertex_y2]]
    # 画等腰三角形，底边中点坐标为[pos_x,pos_y]
    p5 = plt.Line2D([pos_x, vertex_x0], [pos_y, vertex_y0], linewidth=1.2, color='b', label='r1')
    poly = plt.Polygon(polyvertexs,ec="k",fill=False,linewidth=1.0)
    ax.add_patch(poly)
    ax.add_line(p5)
    # 画出围捕机器人的观察范围
    draw_circle(ax, pos_x, pos_y, R_VISION, 'b', '--')


def draw_target(ax, pos_x: float, pos_y: float, theta: float, DISPLAYHEIGHT: float, DISPLAYBASE: float, R_VISION: float, R_ATTACKED: float):
    """画目标

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param pos_x: 目标x坐标(单位为m)
        @param pos_y: 目标y坐标(单位为m)
        @param theta: 目标车头方向(单位为rad)
        @param DISPLAYHEIGHT: 目标三角形的高(单位为m)
        @param DISPLAYBASE: 目标三角形的底边长
        @param R_VISION: 目标的观察范围
        @param R_ATTACKED: 目标的受攻击范围
    """
    vertex_x0, vertex_x1, vertex_x2, vertex_y0, vertex_y1, vertex_y2 = compute_vertex(pos_x, pos_y, theta, DISPLAYHEIGHT, DISPLAYBASE)
    polyvertexs = [[vertex_x0,vertex_y0],[vertex_x1,vertex_y1],[vertex_x2,vertex_y2]]
    # 画等腰三角形，底边中点坐标为[pos_x,pos_y]
    p5 = plt.Line2D([pos_x, vertex_x0], [pos_y, vertex_y0], linewidth=1.2, color='r', label='r1')
    poly = plt.Polygon(polyvertexs,ec="k",fill=False,linewidth=1.0)
    ax.add_patch(poly)
    ax.add_line(p5)
    # 画出目标的受攻击范围
    draw_circle(ax, pos_x, pos_y, R_ATTACKED, 'r')
    # 画出目标的观察范围
    draw_circle(ax, pos_x, pos_y, R_VISION, 'r', '--')


def draw_staobs(ax, pos_x: float, pos_y: float, R: float):
    """画固定圆形障碍物

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param pos_x: 固定圆形障碍物圆心x坐标(单位为m)
        @param pos_y: 固定圆形障碍物圆心y坐标(单位为m)
        @param R: 固定圆形障碍物半径R(单位为m)
    """
    # 黑色线条，且有黑色斜纹填充的matplotlib圆对象，以pos_x，pos_y为圆心，以__R为半径
    cir = plt.Circle((pos_x, pos_y), R,
                        color='black', fill=False, hatch='//', linewidth=1.5)
    ax.add_patch(cir)


def draw_mobobs(ax, pos_x: float, pos_y: float, R: float):
    """画移动圆形障碍物

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param pos_x: 移动圆形障碍物圆心x坐标(单位为m)
        @param pos_y: 移动圆形障碍物圆心y坐标(单位为m)
        @param R: 移动圆形障碍物半径R(单位为m)
    """
    # 绿色线条，且有绿色斜纹填充的matplotlib圆对象，以pos_x，pos_y为圆心，以__R为半径
    cir = plt.Circle((pos_x, pos_y), R,
                        color='green', fill=False, hatch='\\\\', linewidth=1.5)
    ax.add_patch(cir)


def draw_irrobs(ax, poly: np.ndarray):
    """画旋转不规则障碍物

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param poly: 不规则障碍物多边形顶点坐标数组
    """
    poly = plt.Polygon(poly, ec="k", fill=False, hatch='//', linewidth=1.5)
    ax.add_patch(poly)


def draw_mobirrobs(ax, poly: np.ndarray):
    """画移动旋转不规则障碍物

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param poly: 移动不规则障碍物多边形顶点坐标数组
    """
    poly = plt.Polygon(poly, ec="g", fill=False, hatch='\\\\', linewidth=1.5)
    ax.add_patch(poly)


def draw_border(ax, BORDER: np.ndarray):
    """输出matplotlib.patches.Rectangle对象，用于在地图中画出边界

    输入：
        @param ax: plt.gca()获得的当前figure的Axes对象
        @param BORDER: 长度为4的np.ndarray，BORDER[0]为x_min(单位为m)，BORDER[1]为y_min(单位为m)，BORDER[2]为x_max(单位为m)，BORDER[3]为y_max(单位为m)。
              矩形边界的四个顶点分别为[x_min,y_min],[x_max,y_min],[x_min,y_max],[x_max,y_max]
    """
    border = patches.Rectangle((BORDER[0], BORDER[1]), BORDER[2] - BORDER[0], BORDER[3]-BORDER[1], fill=False, linewidth=5)
    ax.add_patch(border)


@jit(nopython = True)
def compute_vertex(pos_x: float, pos_y: float, theta: float, DISPLAYHEIGHT: float, DISPLAYBASE: float):
    """计算机器人三角形三个顶点坐标的数组

    输入：
        @param pos_x: 机器人x坐标(单位为m)
        @param pos_y: 机器人y坐标(单位为m)
        @param theta: 机器人车头方向(单位为rad)
        @param DISPLAYHEIGHT: 小车绘图呈现的三角形高长度(单位为m)
        @param DISPLAYBASE: 小车绘图呈现的三角形底边长度(单位为m)

    输出：
        @return vertex_x0: 机器人三角形顶点0的x坐标(单位为m)
        @return vertex_x1: 机器人三角形顶点1的x坐标(单位为m)
        @return vertex_x2: 机器人三角形顶点2的x坐标(单位为m)
        @return vertex_y0: 机器人三角形顶点0的y坐标(单位为m)
        @return vertex_y1: 机器人三角形顶点1的y坐标(单位为m)
        @return vertex_y2: 机器人三角形顶点2的y坐标(单位为m)
    """

    # 三角形画图顶点vertex_0,vertex_1,vertex_2(为画图效果比实际小车要大)
    vertex_x0 = pos_x+DISPLAYHEIGHT*cos(theta)
    vertex_y0 = pos_y+DISPLAYHEIGHT*sin(theta)
    Q1 = theta-PI/2
    Q2 = theta+PI/2
    vertex_x1 = pos_x+DISPLAYBASE/2*cos(Q1)
    vertex_y1 = pos_y+DISPLAYBASE/2*sin(Q1)
    vertex_x2 = pos_x+DISPLAYBASE/2*cos(Q2)
    vertex_y2 = pos_y+DISPLAYBASE/2*sin(Q2)
    return vertex_x0, vertex_x1, vertex_x2, vertex_y0, vertex_y1, vertex_y2
