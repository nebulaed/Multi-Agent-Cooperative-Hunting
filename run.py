# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件内容为整个仿真过程的主函数，包括初始化、仿真循环语句到数据统计绘图。
# 当前代码实现步骤：
# step1: 初始化围捕机器人群wolves，目标targets，固定障碍物sta_obss，移动障碍物mob_obss，不规则障碍物irr_obss，移动不规则障碍物m_irr_obss，边界rectangle_border。
# step2: 初始化存放围捕机器人和目标的速度、角速度、能量消耗的字典。
# step3: 打开matplotlib绘图窗口。
# step4: 初始化仿真过程中传递数据的字典。
# step5: 执行TOTSTEP次循环，每次循环中
#       1. 清除当前绘图窗口中所有对象。
#       2. 障碍物的移动和旋转，然后计算不规则障碍物和移动不规则障碍物构成边的点。
#       3. 通过围捕算法计算围捕机器人的速度和角速度。
#       4. 计算目标的速度和角速度。
#       5. 围捕机器人和目标根据3中算法给出的速度和角速度移动。
#       6. 画图。
#       7. 判断围捕机器人是否撞上障碍物，若是则跳出循环，判定这次围捕失败，否则继续。
#       8. 记录围捕机器人和目标的速度、角速度、能量消耗，用于后续数据绘图。
#       9. 清除不规则障碍物和移动不规则障碍物构成边的点。
#       10. 判断目标是否失去移动能力，是则判定这次围捕成功，结束循环，否则继续。
# step6: 结束绘图并hold住窗口。
# step7: 利用保存的数据画速度、角速度、能量消耗变化图。
# step8: 如果这次围捕成功，在终端输出机器人的总能量消耗和完成围捕消耗步数。

import argparse
import numpy as np
np.random.seed(100)
import os
import matplotlib.pyplot as plt
import time
from utils.init import init
from utils.draw_data import plot_data, record_data
from utils.draw import plot_all
from utils.updateobs import all_update
from utils.robots_control import robots_movement_strategy
from utils.targets_control import target_go
from utils.move import all_move
from utils.determine import judge_fail
from utils.params import WOLF_NUM, TARGET_NUM, TOTSTEP
from matplotlib.animation import FFMpegWriter
# 导入FFMPEG, 用于制作动画
plt.rcParams['animation.ffmpeg_path'] = 'D:\\Software\\Anaconda3\\Library\\bin\\ffmpeg.exe'


def str2bool(s: str) -> bool:
    """
    将字符串变量'True'或'False'转换为bool变量True或False

    输入：
        s: 字符串变量，'True'或'False'

    输出：
        bool变量
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args():
    """用python执行文件时给定不同的命令产生不同的效果"""

    parser = argparse.ArgumentParser('Hunting Escape Model - XuBozhe')
    parser.add_argument('--display', type=str2bool, default=True,
                        help='whether display the figure')
    parser.add_argument('--output', type=str2bool, default=False,
                        help='whether save the data of robots and targets')
    parser.add_argument('--record', type=str2bool, default=False,
                        help='whether to record video')
    parser.add_argument('--showdata', type=str2bool, default=False,
                        help='whether to display the data curve')

    args = parser.parse_args()
    return args

# 为方便重复调用，将画图前的迭代更新抽象为一个函数
def before_plot(parameter, t):
    # 初始化当前步交互矩阵
    parameter['interact'] = [[0]*(WOLF_NUM+TARGET_NUM)] * (WOLF_NUM+TARGET_NUM)
    # 将当前仿真步数传入参数字典
    parameter['t'] = t
    # 障碍物的旋转和移动，不规则障碍物各边构成点的更新
    all_update(**parameter)
    # 通过围捕机器人的围捕算法计算出所有机器人的速度和角速度
    parameter['old_track_target'], parameter['global_my_t'], parameter['vel_wolves'],parameter['ang_vel_wolves'], parameter['w_d_range'], parameter['v_vector'], parameter['old_attract'] = robots_movement_strategy(**parameter)
    # 计算出目标的速度和角速度
    parameter['vel_targets'], parameter['ang_vel_targets'], parameter['t_d_range'] = target_go(**parameter)
    # 所有围捕机器人和目标根据算法给出的速度和角速度移动
    all_move(**parameter)


# 为方便重复调用，将画图后的迭代更新抽象为一个函数
def after_plot(parameter, data, targets, irr_obss, m_irr_obss):
    # 记录围捕机器人、目标的速度、角速度、能量消耗，用于后续数据绘图
    pos_targets_t, vel_targets_t, ang_vel_targets_t, energy_targets_t, pos_wolves_t, vel_wolves_t, ang_vel_wolves_t, energy_wolves_t, interact_t = record_data(**parameter)
    data['pos_targets'].append(pos_targets_t)
    data['vel_targets'].append(vel_targets_t)
    data['ang_vel_targets'].append(ang_vel_targets_t)
    data['energy_targets'].append(energy_targets_t)
    data['pos_wolves'].append(pos_wolves_t)
    data['vel_wolves'].append(vel_wolves_t)
    data['ang_vel_wolves'].append(ang_vel_wolves_t)
    data['energy_wolves'].append(energy_wolves_t)
    data['interact'].append(interact_t)
    # 判断围捕机器人是否撞上障碍物
    judge_f = judge_fail(**parameter)
    # 若是则跳出循环，判定这次围捕失败，否则继续
    if judge_f == 1:
        return 0, [0, 0, 0], []
    elif judge_f == 2:
        return 0, [3, 0, 0], []
    # 清除不规则障碍物和移动不规则障碍物各边构成点
    # for irr_obs in irr_obss:
    #     irr_obs.elements = []
    # for m_irr_obs in m_irr_obss:
    #     m_irr_obs.elements = []
    # 判断目标是否失去移动能力，是则判定这次围捕成功，结束循环，否则继续。
    all_death = []
    for target in targets:
        all_death.append(target.death)
    if all(all_death):
        return 1, [], all_death

    return 2, [], all_death

# 若该文件存在，则删除此文件，然后保存
def rewrite(path, data):
    if opt.output:
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'w') as f:
            f.write(str(data))

def main(opt):
    """
    群机器人围捕仿真主函数

    输入：
        opt: 命令语句

    输出：
        a: 0表示因与障碍物碰撞，围捕失败，1表示围捕成功，2表示因未在规定时间内完成围捕而失败，3表示因围捕机器人之间碰撞而导致围捕失败
        b: 所有机器人总累计消耗能量(单位为J)
        c: 围捕任务完成所需时间(单位为step)
    """

    # 初始化仿真过程中传递参数的字典
    parameter = {# 以下这部分是优化后选取的算法参数
                 'VARSIGMA': 0.3869405500381927,
                 'D_DANGER': 0.28489136293795562,
                 'D_DANGER_W': 0.50,
                 'ALPHA': 5.050359027046031,
                 'BETA': 2.32753414,
                 'TAU_1': 2.114010945835207,
                 'TAU_2': -0.6788510831700431,
                 'TAU_3': 1.0300341638172719,
                 # 以下这部分是默认设定的一些仿真参数
                 'RADIUS': 0.65,    # 围捕半径
                 'R_VISION': 3.0,   # 机器人的观察范围
                 'D_AVOID': 1.6479672158640435, # 机器人的避障距离
                 'DISPLAYBASE': 0.35,       # 三角形小车的显示底边长度(单位为m)
                 'DISPLAYHEIGHT': 0.28,     # 三角形小车的显示高长度(单位为m)
                 'REALBASE': 0.2,           # 三角形小车的实际底边长度(单位为m)
                 'REALHEIGHT': 0.2,         # 三角形小车的实际高长度(单位为m)
                 'vel_max': 1.5,            # 线速度v最大值(单位为m/s)
                 'ang_vel_max': 7.0,        # 角速度ω最大值(单位为rad/s) 
                 'DIS_AVOID_BORDER': 1.4,   # 避墙距离(单位为m) 
                 'R_ATTACKED': 0.8,         # 目标受攻击的距离(单位为m)
                 'VEL_OBS': 0.5,            # 障碍物在全局坐标系中的速度x轴分量最大值和y轴分量最大值
                 'EXPANSION1': [1.6, 2.0, 2.0, 2.8, 1.0],   # 障碍物少的情况下目标的危险角度区间的扩展系数
                 'EXPANSION2': [1.3, 1.6, 1.6, 2.0, 1.0],   # 障碍物多的情况下目标的危险角度区间的扩展系数
                 'EXPANSION3': [1.6, 2.4, 2.4, 3.2],        # 障碍物少的情况下机器人的危险角度区间的扩展系数
                 'EXPANSION4': [1.3, 1.8, 1.8, 2.2],        # 障碍物多的情况下机器人的危险角度区间的扩展系数
                 'ASSIGN_CYCLE': 5,                         # 机器人重新分配目标的仿真步数周期，即每5步重新分配目标
                 # 以下这些是为了方便传参放进来的一些不断更新的数据
                 'wolves': [], 'targets': [], 'sta_obss': [], 'mob_obss': [], 'irr_obss': [], 'm_irr_obss': [], 'rectangle_border': [],
                 'global_my_t': [],
                 'old_attract': np.zeros((WOLF_NUM, 2)),
                 'old_track_target': np.zeros((WOLF_NUM, 2))}
    
    # 初始化围捕机器人wolves，目标targets，固定障碍物sta_obss，移动障碍物mob_obss，不规则障碍物irr_obss，移动不规则障碍物m_irr_obss，地图边界rectangle_border
    wolves, targets, sta_obss, mob_obss, irr_obss, m_irr_obss, rectangle_border = init(**parameter)
    parameter['wolves'], parameter['targets'], parameter['sta_obss'], parameter['mob_obss'], parameter['irr_obss'], parameter['m_irr_obss'], parameter['rectangle_border'] = wolves, targets, sta_obss, mob_obss, irr_obss, m_irr_obss, rectangle_border

    # 初始化存放围捕机器人和目标的速度、角速度、能量消耗的字典。
    data = {'pos_targets': [], 'vel_targets': [], 'ang_vel_targets': [], 'energy_targets': [],
            'pos_wolves': [], 'vel_wolves': [], 'ang_vel_wolves': [], 'energy_wolves': [], 'interact': []}

    if opt.display:
        # 打开matplotlib绘图窗口
        figure = plt.figure(figsize=(12, 10), constrained_layout=True)
        plt.ion()
        if opt.record:
            metadata = dict(title='hunting', artist='Bozhe Xu',
                            comment='wolves_hunt')
            writer = FFMpegWriter(fps=10, metadata=metadata)    # 输出视频帧率为10
            mp4name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            t1 = time.time()
            with writer.saving(figure, mp4name+'.mp4', 100):    # 图像大小, 文件名, dpi
                # 仿真循环语句
                for t in range(TOTSTEP):
                    # 清除当前绘图窗口中的所有对象
                    plt.cla()
                    before_plot(parameter, t)
                    # 绘图
                    plot_all(data, **parameter)
                    writer.grab_frame()
                    branch, ret, all_death = after_plot(parameter, data, targets, irr_obss, m_irr_obss)
                    if branch == 0: return ret[0], ret[1], ret[2]
                    elif branch == 1: break
        else:
            t1 = time.time()
            # 仿真循环语句
            for t in range(TOTSTEP):
                # 清除当前绘图窗口中的所有对象
                plt.cla()
                before_plot(parameter, t)
                # 绘图
                plot_all(data, **parameter)
                branch, ret, all_death = after_plot(parameter, data, targets, irr_obss, m_irr_obss)
                if branch == 0: return ret[0], ret[1], ret[2]
                elif branch == 1: break
        t2 = time.time()
        tact_time = t2-t1
        print(f'{tact_time} seconds, {t / tact_time} FPS')

        # 停止绘图并hold住窗口
        plt.ioff()
        plt.show()
    else:
        t1 = time.time()
        # 仿真循环语句
        for t in range(TOTSTEP):
            before_plot(parameter, t)
            branch, ret, all_death = after_plot(parameter, data, targets, irr_obss, m_irr_obss)
            if branch == 0: return ret[0], ret[1], ret[2]
            elif branch == 1: break
        t2 = time.time()
        tact_time = t2-t1
        print(f'{tact_time} seconds, {t / tact_time} FPS')

    # 若该文件存在，则删除此文件，然后保存仿真过程中围捕机器人的坐标、速度、角速度、累计能量消耗。
    if opt.output:
        rewrite('output/wolves_pos.txt', data['pos_wolves'])
        rewrite('output/wolves_vel.txt', data['vel_wolves'])
        rewrite('output/wolves_ang_vel.txt', data['ang_vel_wolves'])
        rewrite('output/wolves_energy.txt', data['energy_wolves'])

    # 画出围捕机器人和目标速度、角速度、能量消耗的变化曲线
    if opt.showdata:
        plot_data('v', **data)
        plot_data('w', **data)
        plot_data('E', **data)

    # 若这次围捕成功，在终端输出机器人的总能量消耗和完成围捕消耗步数
    all_t_death = []
    for target in targets:
        all_t_death.append(target.t_death)
    if all(all_death):
        E_sum = 0
        for i in range(WOLF_NUM):
            E_sum += data['energy_wolves'][t][i]
        print('机器人的总能量消耗:', E_sum)
        print('机器人围捕任务完成时间:', max(all_t_death))
        return 1, E_sum, max(all_t_death)
    else:
        return 2, 0, 0


if __name__ == '__main__':
    opt = get_args()
    main(opt)
