# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义一个函数用于判断围捕过程中围捕机器人是否撞上障碍物。


import numpy as np
from cv2 import pointPolygonTest
from utils.math_func import correct, peri_arctan, arcsin, norm, sin, cos, exp, inc_angle
from utils.init import ParamsTable
from utils.collision_detection import two_triangle_test,circle_triangle_test,two_polygon_test


def judge_fail(wolves: list, sta_obss: list, mob_obss: list, irr_obss: list, m_irr_obss: list, **kwargs) -> bool:
    """判断围捕过程中围捕机器人是否撞上障碍物，是输出True，否输出False

    输入：
        wolves: 存放所有围捕机器人对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list

    输出：
        0表示未撞上障碍物或其他机器人，1表示机器人已撞上障碍物，2表示有机器人撞上了其他机器人
    """
    WOLF_NUM, TARGET_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM = ParamsTable.WOLF_NUM, ParamsTable.TARGET_NUM, ParamsTable.S_OBS_NUM, ParamsTable.M_OBS_NUM, ParamsTable.IRR_OBS_NUM, ParamsTable.M_IRR_OBS_NUM
    for i in range(WOLF_NUM):
        tri1 = [[wolves[i].real_x0, wolves[i].real_y0],
                [wolves[i].real_x1, wolves[i].real_y1],
                [wolves[i].real_x2, wolves[i].real_y2]]
        # 检查围捕机器人是否互撞
        for j in range(WOLF_NUM):
            if j != i:
                tri2 = [[wolves[j].real_x0, wolves[j].real_y0],
                        [wolves[j].real_x1, wolves[j].real_y1],
                        [wolves[j].real_x2, wolves[j].real_y2]]
                if two_triangle_test(tri1, tri2):
                    return 2
        # 检查围捕机器人是否撞上固定障碍物
        for j in range(S_OBS_NUM):
            if circle_triangle_test(sta_obss[j].pos, sta_obss[j].R, tri1):
                return 1
        # 检查围捕机器人是否撞上移动障碍物
        for j in range(M_OBS_NUM):
            if circle_triangle_test(mob_obss[j].pos, mob_obss[j].R, tri1):
                return 1
        # 检查围捕机器人是否撞上不规则障碍物
        poly1 = np.array([[[tri1[0][0],tri1[0][1]]],
                            [[tri1[1][0],tri1[1][1]]],
                            [[tri1[2][0],tri1[2][1]]]]).astype(np.float32)
        for j in range(IRR_OBS_NUM):
            poly2 = np.zeros(((irr_obss[j].samples_num, 1, 2))).astype(np.float32)
            for k in range(irr_obss[j].samples_num):
                poly2[k, 0] = np.array([irr_obss[j].vertex_x[irr_obss[j].pose_order[k]],
                                      irr_obss[j].vertex_y[irr_obss[j].pose_order[k]]])
            if two_polygon_test(poly1,poly2):
                return 1
        # 检查围捕机器人是否撞上移动不规则障碍物
        for j in range(M_IRR_OBS_NUM):
            poly2 = np.zeros(((m_irr_obss[j].samples_num, 1, 2))).astype(np.float32)
            for k in range(m_irr_obss[j].samples_num):
                poly2[k, 0] = np.array([m_irr_obss[j].vertex_x[m_irr_obss[j].pose_order[k]],
                                      m_irr_obss[j].vertex_y[m_irr_obss[j].pose_order[k]]])
            if two_polygon_test(poly1,poly2):
                return 1
    return 0
