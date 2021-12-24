# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义的AllUpdate函数内容为障碍物的移动和旋转。


import numpy as np
from typing import List
from model import Robot, Target, StaObs, MobObs, IrregularObs, MobIrregularObs, Border
from utils.relative_pos import vector_count


def all_update(wolves: List[Robot], targets: List[Target], sta_obss: List[StaObs], mob_obss: List[MobObs], irr_obss: List[IrregularObs], m_irr_obss: List[MobIrregularObs], rectangle_border: Border, VEL_OBS: float, **kwargs) -> None:
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
    
    # 移动障碍物移动
    for mob_obs in mob_obss:
        mob_obs.move([np.random.normal(0, VEL_OBS), np.random.normal(0, VEL_OBS)])
    # 不规则障碍物旋转
    for irr_obs in irr_obss:
        irr_obs.update()
    # 移动不规则障碍物移动和旋转
    for m_irr_obs in m_irr_obss:
        m_irr_obs.update([np.random.normal(0, VEL_OBS), np.random.normal(0, VEL_OBS)])

    vector_count(wolves, targets, sta_obss, mob_obss, irr_obss, m_irr_obss, rectangle_border)