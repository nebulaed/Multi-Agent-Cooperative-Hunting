# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义的AllUpdate函数内容为障碍物的移动和旋转。


import numpy as np
from utils.init import ParamsTable
from utils.relative_pos import vector_count


def all_update(wolves: list, targets: list, sta_obss: list, mob_obss: list, irr_obss: list, m_irr_obss: list, rectangle_border: object, **kwargs) -> None:
    """
    障碍物的移动和旋转

    输入：
        wolves: 存放所有围捕机器人对象的list
        targets: 存放所有目标对象的list
        sta_obss: 存放所有固定障碍物对象的list
        mob_obss: 存放所有移动障碍物对象的list
        irr_obss: 存放所有不规则障碍物对象的list
        m_irr_obss: 存放所有移动不规则障碍物对象的list
        rectangle_border: 边界对象
    """
    M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM = ParamsTable.M_OBS_NUM, ParamsTable.IRR_OBS_NUM, ParamsTable.M_IRR_OBS_NUM
    # 障碍物在全局坐标系中的速度x轴分量最大值和y轴分量最大值
    VEL_OBS = 0.5
    # 移动障碍物移动
    for i in range(M_OBS_NUM):
        mob_obss[i].move([np.random.normal(0, VEL_OBS), np.random.normal(0, VEL_OBS)])
    # 不规则障碍物旋转
    for i in range(IRR_OBS_NUM):
        irr_obss[i].update()
    # 移动不规则障碍物移动和旋转
    for i in range(M_IRR_OBS_NUM):
        m_irr_obss[i].update([np.random.normal(0, VEL_OBS), np.random.normal(0, VEL_OBS)])
    # 更新向量差
    vector_count(wolves, targets, sta_obss, mob_obss, irr_obss, m_irr_obss, rectangle_border)
