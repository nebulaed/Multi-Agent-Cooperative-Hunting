# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件利用read_yml中的Params类从params.yml中读取外部参数，如围捕机器人数量、目标数量等等

from utils.read_yml import Params
from numpy import pi

# 从外部读取初始化参数
ParamsTable = Params('params.yml')
WOLF_NUM, TARGET_NUM, S_OBS_NUM, M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM, PI, TOTSTEP, TS, BORDER, INIT_D = ParamsTable.WOLF_NUM, ParamsTable.TARGET_NUM, ParamsTable.S_OBS_NUM, ParamsTable.M_OBS_NUM, ParamsTable.IRR_OBS_NUM, ParamsTable.M_IRR_OBS_NUM, pi, ParamsTable.TOTSTEP, ParamsTable.TS, ParamsTable.border, ParamsTable.INIT_D