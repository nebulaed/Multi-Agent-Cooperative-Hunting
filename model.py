# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义的Agent表示基本unicycle模型，Robot表示围捕机器人，Target表示目标，后两者是Agent的子类。此外本文件还定义了固定障碍物StaObs，移动障碍物MobObs，不规则障碍物IrregularObs，移动不规则障碍物MobIrregularObs，边界Border。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import List
from utils.math_func import correct, peri_arctan, sin, cos, sqrt, rotate_update
from utils.params import WOLF_NUM, TARGET_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM, PI, TOTSTEP, TS
# from utils.collision_detection import two_triangle_test, circle_triangle_test, two_polygon_test


class Agent(object):
    """
    利用输入的初始位置和初始车头方向建立基本unicycle模型，有以下函数：
        update_vertex: 更新小车三角形的顶点位置
        move: 移动
        plot_agent: 在matplotlib绘图窗口中绘出车体
        plot_circle: 在matplotlib绘图窗口中以车体为圆心画圆
        check_feasibility: 检测按照给定速度和角速度运动当前小车会不会发生碰撞
        pos: 输出位置(单位为m)
        ori: 输出车头方向∈[0,2π)
        vel: 输出车的速度(单位为m/s)
        ang_vel: 输出车的角速度(单位为rad/s)
    """
    def __init__(self, f_in: List[float], DISPLAYBASE: float, DISPLAYHEIGHT: float, REALBASE: float, REALHEIGHT: float, vel_max: float, ang_vel_max: float, DIS_AVOID_BORDER: float) -> None:
        """
        输入：
            @param f_in: 长度为3的list，f_in[0]∈[0,2π)为初始车头方向，f_in[1]为初始位置pos_x(单位为m)，f_in[2]为初始位置pos_y(单位为m)
            @param DISPLAYBASE: 小车绘图呈现的三角形底边长度(单位为m)
            @param DISPLAYHEIGHT: 小车绘图呈现的三角形高长度(单位为m)
            @param REALBASE: 小车实际上的三角形底边长度，即轮间距(单位为m)
            @param REALHEIGHT: 小车实际上的三角形高的长度(单位为m)
            @param vel_max: 小车的最大线速度(单位为m/s)
            @param ang_vel_max: 小车的最大角速度(单位为rad/s)
            @param DIS_AVOID_BORDER: 小车对墙的避障距离(单位为m)
       """
        # 位置pos_x, pos_y
        self.__pos_x = f_in[1]
        self.__pos_y = f_in[2]
        # 航向θ∈[0,2π)
        self.__theta = correct(f_in[0])

        # 三角形小车的显示底边长度(单位为m)
        self.DISPLAYBASE = DISPLAYBASE
        # 三角形小车的显示高长度(单位为m)
        self.DISPLAYHEIGHT = DISPLAYHEIGHT
        # 三角形小车的实际底边长度(单位为m)
        self.REALBASE = REALBASE
        # 三角形小车的实际高长度(单位为m)
        self.REALHEIGHT = REALHEIGHT
        # 更新三角形画图顶点和小车实际顶点
        self.update_vertex()
        # 线速度v最大值(单位为m/s)
        self.vel_max = vel_max
        # 角速度ω最大值(单位为rad/s)
        self.ang_vel_max = ang_vel_max
        # 线速度v(单位为m/s)和角速度ω(单位为rad/s)
        self.__vel = 0
        self.__ang_vel = 0
        # 与边界距离(单位为m)
        self.border_d = np.zeros(4)
        # 边界最近点
        self.nearest = np.zeros(2)
        # 累计能量消耗(单位为J)
        self.energy = 0
        # 仿真过程中每一步是否距离障碍物过近，是则self.danger[t]为1，否则为0
        self.danger = [0 for _ in range(TOTSTEP)]
        # 仿真过程中每一步是否有其他机器人距离过近且在车头方向180°扇形内，是则self.danger_w为1，否则为0
        self.danger_w = np.zeros((TOTSTEP, WOLF_NUM), dtype=int)
        # 仿真过程中每一步是否有其他机器人距离过近，是则self.danger_w为1，否则为0
        self.danger_w2 = np.zeros((TOTSTEP, WOLF_NUM), dtype=int)
        # 避墙距离(单位为m)
        self.DIS_AVOID_BORDER = DIS_AVOID_BORDER
        # 避碰标签
        self.avoidCollisionFlag = [0 for _ in range(TOTSTEP)]
        # 避碰标签2
        self.avoidCollisionFlag2 = [0 for _ in range(TOTSTEP)]
        self.dangerObsFlag = {"sta_obss": np.zeros((TOTSTEP, S_OBS_NUM), dtype=int),
                              "mob_obss": np.zeros((TOTSTEP, M_OBS_NUM), dtype=int),
                              "irr_obss": np.zeros((TOTSTEP, IRR_OBS_NUM), dtype=int),
                              "m_irr_obss": np.zeros((TOTSTEP, M_IRR_OBS_NUM), dtype=int),
                              "obs": np.zeros(TOTSTEP, dtype=int)}

        # 初始化该个体到移动障碍物的距离(单位为m)
        self.d_m = np.zeros(M_OBS_NUM)
        # 初始化该个体到固定障碍物的距离(单位为m)
        self.d_s = np.zeros(S_OBS_NUM)
        # 初始化该个体到不规则障碍物的距离(单位为m)
        self.d_ir = np.zeros(IRR_OBS_NUM)
        # 初始化该个体到移动不规则障碍物的距离(单位为m)
        self.d_m_ir = np.zeros(M_IRR_OBS_NUM)
        # 初始化该个体避障距离内的移动障碍物列表
        self.danger_m = []
        # 初始化该个体避障距离内的固定障碍物列表
        self.danger_s = []
        # 初始化该个体避障距离内的不规则障碍物列表
        self.danger_ir = []
        # 初始化该个体避障距离内的移动不规则障碍物列表
        self.danger_m_ir = []
        # 初始化该个体避墙距离内的边界列表
        self.danger_border = []

    def update_vertex(self) -> None:
        # 三角形画图顶点vertex_0,vertex_1,vertex_2(为画图效果比实际小车要大)
        self.vertex_x0 = self.__pos_x+self.DISPLAYHEIGHT*cos(self.__theta)
        self.vertex_y0 = self.__pos_y+self.DISPLAYHEIGHT*sin(self.__theta)
        Q1 = self.__theta-PI/2
        Q2 = self.__theta+PI/2
        self.vertex_x1 = self.__pos_x+self.DISPLAYBASE/2*cos(Q1)
        self.vertex_y1 = self.__pos_y+self.DISPLAYBASE/2*sin(Q1)
        self.vertex_x2 = self.__pos_x+self.DISPLAYBASE/2*cos(Q2)
        self.vertex_y2 = self.__pos_y+self.DISPLAYBASE/2*sin(Q2)
        # 小车实际顶点real_x0,real_x1,real_x2
        self.real_x0 = self.__pos_x+self.REALHEIGHT*cos(self.__theta)
        self.real_y0 = self.__pos_y+self.REALHEIGHT*sin(self.__theta)
        self.real_x1 = self.__pos_x+self.REALBASE/2*cos(Q1)
        self.real_y1 = self.__pos_y+self.REALBASE/2*sin(Q1)
        self.real_x2 = self.__pos_x+self.REALBASE/2*cos(Q2)
        self.real_y2 = self.__pos_y+self.REALBASE/2*sin(Q2)

    def move(self, vel: float, ang_vel: float) -> None:
        """
        unicycle模型遵循的运动学方程:
        x[k+1] = x[k] + v[k]*cosθ[k]*TS
        y[k+1] = y[k] + v[k]*sinθ[k]*TS
        θ[k+1] = θ[k] + ω[k]*TS

        输入：
            @param vel: 线速度(单位为m/s)
            @param ang_vel: 角速度(单位为rad/s)
        """
        self.__pos_x += vel*cos(self.__theta)*TS
        self.__pos_y += vel*sin(self.__theta)*TS
        self.__theta = correct(self.__theta+ang_vel*TS)
        self.__vel = vel
        self.__ang_vel = ang_vel
        # 计算累计能量消耗，默认质量为1kg，速度为m/s，因此能量单位为J
        self.energy += 1/2*1*(self.vel**2)
        self.update_vertex()

    def plot_agent(self, ax) -> None:
        """
        在matplotlib绘图窗口中画出当前agent的车体

        输入：
            @param ax: 当前figure的Axes对象
        """

        # 画等腰三角形，底边中点坐标为[self.__pos_x,self.__pos_y]
        polyvertexs = [[self.vertex_x0,self.vertex_y0],[self.vertex_x1,self.vertex_y1],[self.vertex_x2,self.vertex_y2]]
        poly = plt.Polygon(polyvertexs,ec="k",fill=False,linewidth=1.0, label = 'r1')
        ax.add_patch(poly)
    
    # def plot_agent2(self, ax) -> None:
    #     """在matplotlib绘图窗口中画出车体"""
    #     polyvertexs = [[self.vertex_x0,self.vertex_y0],[self.vertex_x1,self.vertex_y1],[self.vertex_x2,self.vertex_y2]]
    #     # 黑色条纹填充
    #     poly = plt.Polygon(polyvertexs,ec="b",fill=True,facecolor='b',linewidth=1.5)
    #     ax.add_patch(poly)

    def plot_circle(self, ax, r: float, ec: str, ls: str = '-') -> None:
        """画圆

        输入：
            @param ax: plt.gca()获得的当前figure的Axes对象
            @param r: 圆的半径(单位为m)
            @param ec: 线条颜色
            @param ls: 线型，默认为实线
        """
        Circle = plt.Circle((self.__pos_x, self.__pos_y), r, ec=ec, ls = ls, fill=False, linewidth=1.0)
        ax.add_patch(Circle)

    def check_feasibility(self, vel: float, ang_vel: float, border: object, wolves: List, sta_obss: List, mob_obss: List, irr_obss: List, m_irr_obss: List, mark: int) -> bool:
        '''
        若移动后的新位置不超出边界范围且不与其他小车碰撞，按照输入速度和角速度移动

        输入：
            @param vel: 线速度(单位为m/s)
            @param ang_vel: 角速度(单位为rad/s)
            @param border: 边界
            @param wolves: 存放所有围捕机器人对象的list
            @param sta_obss: 存放所有固定障碍物对象的list
            @param mob_obss: 存放所有移动障碍物对象的list
            @param irr_obss: 存放所有不规则障碍物对象的list
            @param m_irr_obss: 存放所有移动不规则障碍物对象的list
            @param mark: 当前机器人序号

        输出：
            @return: True表示新位置不超出边界且不与其他小车碰撞，False表示新位置超出边界或与其他小车碰撞，不可行
        '''

        # 按照给定速度和角速度到达的新位置
        new_pos_x = self.__pos_x+vel*cos(self.__theta)*TS
        new_pos_y = self.__pos_y+vel*sin(self.__theta)*TS
        # 若新位置未超出边界
        if border.X_MIN < new_pos_x < border.X_MAX and border.Y_MIN < new_pos_y < border.Y_MAX:
            # # 新车头方向
            # new_theta = correct(self.__theta+ang_vel*TS)
            # # 根据机器人实际大小算出机器人三角形各顶点的位置
            # tri1 = np.array([[new_pos_x+self.REALHEIGHT*cos(new_theta), new_pos_y+self.REALHEIGHT*sin(new_theta)],
            #         [new_pos_x+self.REALBASE/2*cos(new_theta-PI/2), new_pos_y+self.REALBASE/2*sin(new_theta-PI/2)],
            #         [new_pos_x+self.REALBASE/2*cos(new_theta+PI/2), new_pos_y+self.REALBASE/2*sin(new_theta+PI/2)]])
            # for i in range(len(wolves)):
            #     if i != mark:
            #         '''
            #         # 另外一机器人的领域中心
            #         field_center = wolves[i].pos+0.2*np.array([cos(wolves[i].ori),sin(wolves[i].ori)])
            #         theta = np.linspace(0, 6.28, 129)
            #         Circle1 = field_center[0]+0.2*cos(theta)
            #         Circle2 = field_center[1]+0.2*sin(theta)
            #         plt.plot(Circle1, Circle2, 'k-', linewidth=1.0)
            #         # 若当前机器人三角形与另外一机器人的领域圆交叠，则判定不可行
            #         if circle_triangle_test(field_center, 0.35, tri1):
            #             return False
            #         '''
            #         # 另一机器人的显示三角形各顶点位置
            #         tri2 = np.array([[wolves[i].real_x0, wolves[i].real_y0],
            #                 [wolves[i].real_x1, wolves[i].real_y1],
            #                 [wolves[i].real_x2, wolves[i].real_y2]])
            #         # 若当前机器人三角形与另外一机器人的显示三角形交叠，则判定不可行
            #         if two_triangle_test(tri1, tri2):
            #             return False
            # for sta_obs in sta_obss:
            #     if circle_triangle_test(sta_obs.pos, sta_obs.R, tri1):
            #         return False
            # for mob_obs in mob_obss:
            #     if circle_triangle_test(mob_obs.pos, mob_obs.R, tri1):
            #         return False
            # poly1 = np.array([[[tri1[0][0],tri1[0][1]]],
            #                     [[tri1[1][0],tri1[1][1]]],
            #                     [[tri1[2][0],tri1[2][1]]]]).astype(np.float32)
            # for irr_obs in irr_obss:
            #     poly2 = np.zeros(((irr_obs.samples_num, 1, 2))).astype(np.float32)
            #     for k in range(irr_obs.samples_num):
            #         poly2[k, 0] = np.array([irr_obs.vertex_x[irr_obs.pose_order[k]],
            #                             irr_obs.vertex_y[irr_obs.pose_order[k]]])
            #     if two_polygon_test(poly1,poly2):
            #         return False
            # for m_irr_obss in m_irr_obss:
            #     poly2 = np.zeros(((m_irr_obss.samples_num, 1, 2))).astype(np.float32)
            #     for k in range(m_irr_obss.samples_num):
            #         poly2[k, 0] = np.array([m_irr_obss.vertex_x[m_irr_obss.pose_order[k]],
            #                             m_irr_obss.vertex_y[m_irr_obss.pose_order[k]]])
            #     if two_polygon_test(poly1,poly2):
            #         return False
            return True
        # 若新位置超出边界，则判定不可行
        else:
            return False

    @property
    def pos(self) -> np.ndarray:
        """输出位置的接口

        输出：
            @return np.array([self.__pos_x,self.__pos_y]): 两个元素的numpy.ndarray数组，其中第一个为x坐标，第二个为y坐标，单位均为m
        """
        return np.array([self.__pos_x, self.__pos_y])

    @property
    def ori(self) -> float:
        """输出车头方向的接口

        输出：
            @return self.__theta: 车头方向∈[0,2π)
        """
        return self.__theta

    @property
    def vel(self) -> float:
        """输出线速度的接口

        输出：
            @return self.__vel: 线速度(单位为m/s)
        """
        return self.__vel

    @property
    def ang_vel(self) -> float:
        """输出角速度的接口

        输出：
            @return self.__ang_vel: 角速度(单位为rad/s)
        """
        return self.__ang_vel


class Robot(Agent):
    """
    以Agent为父类，增加一些属于围捕机器人的属性和功能，有以下函数：
        plot_robot: 调用父类Agent的plotagent函数并用蓝色线画出等腰三角形的高，调用父类Agent的plotcircle函数画出围捕机器人的观察范围
        find_near: 对围捕机器人同伴按由近到远进行排序
    """
    def __init__(self, f_in: List[float], DISPLAYBASE: float, DISPLAYHEIGHT: float, REALBASE: float, REALHEIGHT: float, vel_max: float, ang_vel_max: float, DIS_AVOID_BORDER: float, R_VISION: float, AVOID_DIST: float) -> None:
        """
        输入：
            @param f_in: 长度为3的list，f_in[0]∈[0,2π)为初始车头方向，f_in[1]为初始位置pos_x(单位为m)，f_in[2]为初始位置pos_y(单位为m)
            @param DISPLAYBASE: 小车绘图呈现的三角形底边长度(单位为m)
            @param DISPLAYHEIGHT: 小车绘图呈现的三角形高长度(单位为m)
            @param REALBASE: 小车实际上的三角形底边长度，即轮间距(单位为m)
            @param REALHEIGHT: 小车实际上的三角形高的长度(单位为m)
            @param vel_max: 小车的最大线速度(单位为m/s)
            @param ang_vel_max: 小车的最大角速度(单位为rad/s)
            @param DIS_AVOID_BORDER: 小车对墙的避障距离(单位为m)
            @param R_VISION: 小车的观察距离(单位为m)
            @param AVOID_DIST: 小车的避障距离(单位为m)
       """
        # 将父类Agent的__init__函数包含进来
        super(Robot, self).__init__(f_in, DISPLAYBASE, DISPLAYHEIGHT, REALBASE, REALHEIGHT, vel_max, ang_vel_max, DIS_AVOID_BORDER)
        # 机器人的车身颜色: 蓝色
        self.__color = 'b'
        # 机器人的观察范围(单位为m)
        self.R_VISION = R_VISION
        # 机器人的避障距离(单位为m)
        self.AVOID_DIST = AVOID_DIST
        # 机器人是否检测到目标
        self.detection = False

        # 初始化围捕机器人到固定障碍物的向量(单位为m)
        self.wolf_to_s_obs = np.zeros((S_OBS_NUM, 2))
        # 初始化围捕机器人到移动障碍物的向量(单位为m)
        self.wolf_to_m_obs = np.zeros((M_OBS_NUM, 2))
        # 初始化围捕机器人到不规则障碍物的向量(单位为m)
        self.wolf_to_irr_obs = np.zeros((IRR_OBS_NUM, 2))
        # 初始化围捕机器人到移动不规则障碍物的向量(单位为m)
        self.wolf_to_m_irr_obs = np.zeros((M_IRR_OBS_NUM, 2))
        # 初始化围捕机器人到目标的向量(单位为m)
        self.wolf_to_target = np.zeros((TARGET_NUM, 2))
        # 初始化围捕机器人间位置向量差(单位为m)
        self.wolves_dif = np.zeros((WOLF_NUM, 2))
        # 围捕机器人的同类由近到远排序的索引list
        self.neighbor = np.zeros(WOLF_NUM-1)

    def plot_robot(self, ax) -> None:
        """在matplotlib绘图窗口中画出围捕机器人的车体

        输入：
            @param ax: 当前figure的Axes对象
        """
        self.plot_agent(ax)
        p5 = plt.Line2D([self.pos[0], self.vertex_x0], [self.pos[1], self.vertex_y0], linewidth=1.2, color=self.__color, label='r1')
        ax.add_line(p5)
        # 画出围捕机器人的观察范围
        self.plot_circle(ax, self.R_VISION, 'b', '--')

    def find_near(self) -> None:
        """
        将同类围捕机器人按照由近及远的顺序进行排序
        """

        n = np.arange(1, WOLF_NUM+1, 1)
        m = np.zeros(WOLF_NUM)
        for j in range(WOLF_NUM):
            m[j] = np.linalg.norm(self.wolves_dif[j])
        for x in range(WOLF_NUM):
            for y in range(x+1, WOLF_NUM):
                if m[x] > m[y]:
                    m[x], m[y] = m[y], m[x]
                    n[x], n[y] = n[y], n[x]
        for k in range(WOLF_NUM-1):
            self.neighbor[k] = n[k+1]


class Target(Agent):
    """
    以Agent为父类，增加一些属于目标的属性和功能，有以下函数：
        plot_target: 调用父类Agent的plotagent函数并用红色线画出等腰三角形的高，调用父类Agent的plotcircle函数画出目标的观察范围和受攻击范围
    """
    def __init__(self, f_in: List[float], DISPLAYBASE: float, DISPLAYHEIGHT: float, REALBASE: float, REALHEIGHT: float, vel_max: float, ang_vel_max: float, DIS_AVOID_BORDER: float, R_ATTACKED: float, R_VISION: float, AVOID_DIST: float) -> None:
        """
        输入：
            @param f_in: 长度为3的list，f_in[0]∈[0,2π)为初始车头方向，f_in[1]为初始位置pos_x(单位为m)，f_in[2]为初始位置pos_y(单位为m)
            @param DISPLAYBASE: 小车绘图呈现的三角形底边长度(单位为m)
            @param DISPLAYHEIGHT: 小车绘图呈现的三角形高长度(单位为m)
            @param REALBASE: 小车实际上的三角形底边长度，即轮间距(单位为m)
            @param REALHEIGHT: 小车实际上的三角形高的长度(单位为m)
            @param vel_max: 目标小车的最大线速度(单位为m/s)
            @param ang_vel_max: 目标小车的最大角速度(单位为rad/s)
            @param DIS_AVOID_BORDER: 目标小车对墙的避障距离(单位为m)
            @param R_ATTACKED: 目标的受攻击距离(单位为m)
            @param R_VISION: 目标的观察距离(单位为m)
            @param AVOID_DIST: 目标的避障距离(单位为m)
        """
        # 将父类Agent的__init__函数包含进来
        super(Target, self).__init__(f_in, DISPLAYBASE, DISPLAYHEIGHT, REALBASE, REALHEIGHT, vel_max, ang_vel_max, DIS_AVOID_BORDER)
        # 目标的车身颜色: 红色
        self.__color = 'r'
        # 目标受攻击的距离(单位为m)
        self.R_ATTACKED = R_ATTACKED
        # 目标的观察范围(单位为m)
        self.R_VISION = R_VISION
        # 目标的避障距离(单位为m)
        self.AVOID_DIST = AVOID_DIST
        # 目标首次观察到个体的时间，初始化为0
        self.t_ob = 0
        # 目标受个体攻击数，初始化为0
        self.attacked = 0
        # 目标是否死亡，True表示死亡，False表示未死亡
        self.death = False
        # 目标死亡的步数，初始化为0
        self.t_death = 0
        # 目标随机偏转的角度(单位为rad)
        self.deflection = np.random.rand(TOTSTEP//20+1)
        # 目标到狼的向量(单位为m)
        self.target_to_wolf = np.zeros((WOLF_NUM, 2))
        # 目标到固定障碍物的向量(单位为m)
        self.target_to_s_obs = np.zeros((S_OBS_NUM, 2))
        # 目标到移动障碍物的向量(单位为m)
        self.target_to_m_obs = np.zeros((M_OBS_NUM, 2))

    def plot_target(self, ax) -> None:
        """在matplotlib绘图窗口中画出目标

        输入：
            @param ax: 当前figure的Axes对象
        """
        self.plot_agent(ax)
        p5 = plt.Line2D([self.pos[0], self.vertex_x0], [self.pos[1], self.vertex_y0], linewidth=1.1, color=self.__color)
        ax.add_line(p5)
        # 画出目标的受攻击范围
        self.plot_circle(ax, self.R_ATTACKED, 'r')
        # 画出目标的观察范围
        self.plot_circle(ax, self.R_VISION, 'r', '--')


class Obs(object):
    """
    用输入的初始位置pos_x, pos_y和半径__R建立基本圆形障碍物模型，有以下函数：
        plot_obs: 输出matplotlib圆对象，用于画图
        pos: 输出障碍物圆心坐标
        R: 输出障碍物的半径
    """

    def __init__(self, f_in: List[float]) -> None:
        """
        输入：
            @param f_in: 长度为3的list，f_in[0]为障碍物圆半径(单位为m)，f_in[1]为障碍物圆心初始位置pos_x(单位为m)，f_in[2]为初始位置pos_y(单位为m)
        """

        # 障碍物的位置pos_x, pos_y和半径__R(单位为m)
        self._pos_x = f_in[1]
        self._pos_y = f_in[2]
        self.__R = f_in[0]

    def plot_obs(self, ax) -> None:
        """
        画出障碍物

        输入：
            @param ax: 当前figure的Axes对象
        """

        # 黑色线条，且有黑色斜纹填充的matplotlib圆对象，以pos_x，pos_y为圆心，以__R为半径
        cir = plt.Circle((self._pos_x, self._pos_y), self.__R,
                         color='black', fill=False, hatch='//', linewidth=1.5)
        ax.add_patch(cir)

    @property
    def pos(self) -> np.ndarray:
        """输出障碍物坐标的接口

        输出：
            @return np.array([self._pos_x,self._pos_y]): 两个元素的numpy.ndarray数组，其中第一个为障碍物圆心x坐标，第二个为障碍物圆心y坐标，单位均为m
        """
        return np.array([self._pos_x, self._pos_y])

    @property
    def R(self) -> float:
        """输出障碍物半径的接口

        输出：
            @return self.__R: 障碍物半径(单位为m)
        """
        return self.__R


class StaObs(Obs):
    """
    固定障碍物，完全继承自Obs类
    """
    pass


class MobObs(Obs):
    """
    以Obs为类，增加属于移动障碍物的属性和功能，有以下函数：
        plot_obs: 输出matplotlib圆对象，用于画图
        move: 障碍物的移动
    """

    def plot_obs(self, ax) -> None:
        """
        画出移动障碍物

        输入：
            @param ax: 当前figure的Axes对象
        """

        # 绿色线条，且有绿色斜纹填充的matplotlib圆对象，以pos_x，pos_y为圆心，以__R为半径
        cir = plt.Circle((self.pos[0], self.pos[1]), self.R,
                         color='green', fill=False, hatch='\\\\', linewidth=1.5)
        ax.add_patch(cir)

    def move(self, v_in: List[float]) -> None:
        """
        移动障碍物遵循质点运动规则：
        x[k+1] = x[k] + v_x[k]*TS
        y[k+1] = y[k] + v_y[k]*TS

        输入：
            @param v_in: 长度为2的list，v_in[0]为全局坐标系中速度的x轴方向分量(单位为m/s)，v_in[1]为全局坐标系中速度的y轴方向分量(单位为m/s)
        """
        self._pos_x += v_in[0]*TS
        self._pos_y += v_in[1]*TS


class IrregularObs(object):
    """
    用输入的生成点位置pos_x, pos_y和半径__R建立不规则障碍物，基本原理为以生成点pos_x,pos_y为圆心，R为半径画圆，在该圆内随机生成7到16个顶点，将这些顶点按[0,2π)的角度顺序连接起来，形成一个多边形不规则障碍物，有以下函数：
        plot_obs: 画出不规则障碍物的顶点出生的范围(实际未用到)
        update: 障碍物的旋转，同时更新不规则障碍物各边的构成点
        plot_irr_obs: 输出matplotlib多边形对象，用于在绘图窗口中画出不规则障碍物
        pos: 输出不规则障碍物顶点的生成圆圆心
        R: 输出不规则障碍物顶点生成半径
    """

    def __init__(self, f_in: List[float]) -> None:
        """
        输入：
            @param f_in: 长度为3的list，f_in[0]为障碍物顶点生成圆半径(单位为m)，f_in[1]为障碍物顶点生成圆圆心初始位置pos_x(单位为m)，f_in[2]为障碍物顶点生成圆圆心初始位置pos_y(单位为m)
        """
        # 不规则障碍物顶点的生成圆圆心pos_x, pos_y
        self._pos_x = f_in[1]
        self._pos_y = f_in[2]
        # 不规则障碍物的顶点生成半径R(单位为m)
        self.__R = f_in[0]
        # 圆内顶点数量: 7到16的随机整数
        self.samples_num = np.random.randint(7, 16)
        # 圆内顶点相对于圆心的方向，为保证第一二三四象限均有点，故在[0,π/2),[π/2,π),[π,3π/2),[3π/2,2π)都随机产生一些
        self.t = np.concatenate((np.random.random(size=round(self.samples_num/4))*PI/2, np.random.random(size=round(self.samples_num/4))*PI/2+PI/2, np.random.random(size=round(self.samples_num/4))*PI/2+PI, np.random.random(size=self.samples_num-round(self.samples_num/4)*3)*PI+3*PI/2), axis=0)
        # 修正角度
        for item in self.t:
            item = correct(item)
        # 0.07到1间的随机数，避免生成的顶点离生成圆圆心太近
        self.r_num = np.random.random(self.samples_num)*0.93+0.07
        # 圆内顶点的x坐标
        self.vertex_x = np.zeros(self.samples_num)
        # 圆内顶点的y坐标
        self.vertex_y = np.zeros(self.samples_num)
        # 不规则障碍物顶点各边的构成点
        self.elements = np.zeros(self.samples_num * 5)
        # 不规则障碍物的多边形
        self.poly = []
        # 单位圆内对应的x,y
        unit_x, unit_y = cos(self.t), sin(self.t)
        # 由于与生成圆圆心的距离为polar_r∈[sqrt(0.07),sqrt(1)]即[0.26m,1m]，方向为self.t[i]，可计算出顶点i为(vertex_x[i],vertex_y[i])
        for i in range(self.samples_num):
            polar_r = sqrt(self.r_num[i])*(self.R)
            self.vertex_x[i] = unit_x[i]*polar_r+self._pos_x
            self.vertex_y[i] = unit_y[i]*polar_r+self._pos_y
        # 将这些顶点按照出生时的角度顺序排好序，得到顶点索引顺序list: self.pose_order
        vertex_pose = np.zeros((self.samples_num, 2))
        angle = np.zeros(self.samples_num)
        for i in range(self.samples_num):
            vertex_pose[i] = np.array([self.vertex_x[i], self.vertex_y[i]])
            angle[i] = peri_arctan(vertex_pose[i]-self.pos)
        self.pose_order = np.argsort(angle)

    def plot_obs(self, ax) -> None:
        """
        输出matplotlib圆对象，用于画不规则障碍物的出生圆

        输入：
            @param ax: 当前figure的Axes对象
        """
        cir = plt.Circle((self._pos_x, self._pos_y), self.R,
                         color='black', fill=False, linewidth=1.5, linestyle='--')
        ax.add_patch(cir)

    def update(self) -> None:
        """障碍物的旋转和构成点的更新"""

        self.poly = []
        # 障碍物的旋转幅度为(-π/16,π/16)的均匀随机分布
        variation = np.random.uniform(-PI/16, PI/16)
        self.elements = rotate_update(self.t, self.samples_num, self.r_num, self.R, self.vertex_x, self.vertex_y, self._pos_x, self._pos_y, self.pose_order, variation)
        for i in range(self.samples_num):
            self.poly.append((self.vertex_x[self.pose_order[i]], self.vertex_y[self.pose_order[i]]))

    def plot_irr_obs(self, ax) -> None:
        """
        输出matplotlib.Polygon对象，用于在绘图窗口中画出不规则障碍物

        输入：
            @param ax: 当前figure的Axes对象
        """
        
        # 标出不规则障碍物每条边的构成点
        # for item in self.elements:
        #     plt.plot(item[0],item[1],'k.')
        # 黑色条纹填充
        poly = plt.Polygon(self.poly, ec="k", fill=False,
                           hatch='//', linewidth=1.5)
        ax.add_patch(poly)

    @property
    def pos(self) -> np.ndarray:
        """输出障碍物出生圆圆心的接口

        输出：
            @return np.array([self._pos_x,self._pos_y]): 两个元素的numpy.ndarray数组，其中第一个为出生圆圆心x坐标，第二个为出生圆圆心y坐标，单位均为m
        """
        return np.array([self._pos_x, self._pos_y])

    @property
    def R(self) -> float:
        """输出障碍物出生圆半径的接口

        输出：
            @return self.__R: 障碍物出生圆半径(单位为m)
        """
        return self.__R


class MobIrregularObs(IrregularObs):
    """
    继承自IrregularObs类，用输入的生成点位置pos_x, pos_y和半径__R建立移动不规则障碍物，增加或改动以下函数：
        update: 障碍物的移动和旋转，同时更新不规则障碍物各边的构成点
        plot_irr_obs: 输出matplotlib多边形对象，用于在绘图窗口中画出移动不规则障碍物
    """

    def update(self, v_in: List[float]) -> None:
        """
        障碍物的移动、旋转和各边构成点的更新。
        移动不规则障碍物的移动遵循质点运动规则：
        x[k+1] = x[k] + v_x[k]*TS
        y[k+1] = y[k] + v_y[k]*TS

        输入：
            @param v_in: 长度为2的list，v_in[0]为全局坐标系中速度的x轴方向分量v_x(单位为m/s)，v_in[1]为全局坐标系中速度的y轴方向分量v_y(单位为m/s)
        """

        self.poly = []
        # 障碍物的旋转幅度为(-π/16,π/16)的均匀随机分布
        variation = np.random.uniform(-PI/16, PI/16)
        self.elements = rotate_update(self.t, self.samples_num, self.r_num, self.R, self.vertex_x, self.vertex_y, self._pos_x, self._pos_y, self.pose_order, variation)
        # 障碍物的移动
        self._pos_x += v_in[0]*TS
        self._pos_y += v_in[1]*TS
        self.poly = []
        for i in range(self.samples_num):
            self.poly.append((self.vertex_x[self.pose_order[i]], self.vertex_y[self.pose_order[i]]))

    def plot_irr_obs(self, ax) -> None:
        """
        输出matplotlib.Polygon对象，用于在绘图窗口中画出不规则障碍物

        输入：
            @param ax: 当前figure的Axes对象
        """
        # 标出不规则障碍物每条边的构成点
        # for item in self.elements:
        #     plt.plot(item[0],item[1],'k.')
        # 绿色条纹填充
        poly = plt.Polygon(self.poly, ec="g", fill=False,
                           hatch='\\\\', linewidth=1.5)
        ax.add_patch(poly)


class Border(object):
    """
    地图边界，有以下函数：
        plot_border: 输出matplotlib对象，用于后续在绘图窗口中画出地图边界
        X_MIN: 输出矩形边界的左边界的x轴坐标
        Y_MIN: 输出矩形边界的下边界的y轴坐标
        X_MAX: 输出矩形边界的右边界的x轴坐标
        Y_MAX: 输出矩形边界的上边界的y轴坐标
    """

    def __init__(self, f_in: List[float]) -> None:
        """
        输入：
            @param f_in: 长度为4的list，f_in[0]为x_min(单位为m)，f_in[1]为y_min(单位为m)，f_in[2]为x_max(单位为m)，f_in[3]为y_max(单位为m)。
                  矩形边界的四个顶点分别为[x_min,y_min],[x_max,y_min],[x_min,y_max],[x_max,y_max]
        """
        self.__X_MIN = f_in[0]
        self.__Y_MIN = f_in[1]
        self.__X_MAX = f_in[2]
        self.__Y_MAX = f_in[3]

    def plot_border(self, ax) -> None:
        """
        输出matplotlib.patches.Rectangle对象，用于在地图中画出边界

        输入：
            @param ax: 当前figure的Axes对象
        """
        border = patches.Rectangle((self.__X_MIN, self.__Y_MIN), self.__X_MAX -
                                   self.__X_MIN, self.__Y_MAX-self.__Y_MIN, fill=False, linewidth=5)
        ax.add_patch(border)

    @property
    def X_MIN(self) -> float:
        return self.__X_MIN

    @property
    def Y_MIN(self) -> float:
        return self.__Y_MIN

    @property
    def X_MAX(self) -> float:
        return self.__X_MAX

    @property
    def Y_MAX(self) -> float:
        return self.__Y_MAX

