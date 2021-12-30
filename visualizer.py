# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件内容为整个复现过程的主函数，包括初始化、画图循环语句到数据统计绘图。

import argparse
import numpy as np
import time
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
from matplotlib.font_manager import FontProperties  # 字体属性管理器
from matplotlib.animation import FFMpegWriter
from visualization.tools import draw_robot, draw_target, draw_staobs, draw_mobobs, draw_irrobs, draw_mobirrobs, draw_border
from utils.draw_data import plot_data
# 导入FFMPEG, 用于制作动画
plt.rcParams['animation.ffmpeg_path'] = 'D:\\Software\\Anaconda3\\Library\\bin\\ffmpeg.exe'

# 设置字体
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# 设置字体及其大小，修改字体时替换字体的路径即可
font1 = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=14)


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
    parser.add_argument('--record', type=str2bool, default=False,
                        help='whether to record video')
    parser.add_argument('--showdata', type=str2bool, default=True,
                        help='whether to display the data curve')

    args = parser.parse_args()
    return args


def plot_all(t: int, pos_wolves: np.ndarray, ori_wolves: np.ndarray, pos_targets: np.ndarray, ori_targets: np.ndarray, sta_obs_params: np.ndarray, mob_obs_params: np.ndarray, irr_obs_params: np.ndarray, m_irr_obs_params: np.ndarray, BORDER: np.ndarray, DISPLAYHEIGHT: float, DISPLAYBASE: float, R_VISION: float, R_ATTACKED: float, wolves_path: np.ndarray, targets_path: np.ndarray, **kwargs):

    WOLF_NUM, TARGET_NUM, S_OBS_NUM = len(pos_wolves[0]), len(pos_targets[0]), len(sta_obs_params)
    M_OBS_NUM, IRR_OBS_NUM, M_IRR_OBS_NUM= len(mob_obs_params[0]), len(irr_obs_params[0]), len(m_irr_obs_params[0])
    ax = plt.gca()
    # wolves_wedges, targets_wedges = [], []
    for i in range(WOLF_NUM):
        pos_x, pos_y = pos_wolves[t][i][0], pos_wolves[t][i][1]
        draw_robot(ax, pos_x, pos_y, ori_wolves[t][i], DISPLAYHEIGHT, DISPLAYBASE, R_VISION)
        plt.text(pos_x, pos_y, i, fontproperties=font1)
        path = plt.Line2D(wolves_path[i, 0, :t+1], wolves_path[i, 1, :t+1], ls='--', color='b', lw=0.8)
        ax.add_line(path)

    for i in range(TARGET_NUM):
        draw_target(ax, pos_targets[t][i][0], pos_targets[t][i][1], ori_targets[t][i], DISPLAYHEIGHT, DISPLAYBASE, R_VISION, R_ATTACKED)
        path = plt.Line2D(targets_path[i, 0, :t+1], targets_path[i, 1, :t+1], ls='--', color='r', lw=0.8)
        ax.add_line(path)


    # 保持横纵坐标比例尺一致
    plt.xlim(BORDER[0] - 0.5, BORDER[2] + 0.5)
    plt.ylim(BORDER[1] - 0.5, BORDER[3] + 0.5)
    ax.set_aspect('equal', adjustable='box')
    # 画出边界
    draw_border(ax, BORDER)
    for i in range(S_OBS_NUM):
        draw_staobs(ax, sta_obs_params[i][1], sta_obs_params[i][2], sta_obs_params[i][0])
    for i in range(M_OBS_NUM):
        draw_mobobs(ax, mob_obs_params[t][i][1], mob_obs_params[t][i][2], mob_obs_params[t][i][0])
    for i in range(IRR_OBS_NUM):
        draw_irrobs(ax, irr_obs_params[t][i])
    for i in range(M_IRR_OBS_NUM):
        draw_mobirrobs(ax, m_irr_obs_params[t][i])

    plt.title(f'仿真步数t={t}', fontsize=14, fontfamily='SimSun')
    # 设置刻度标签字体
    plt.xticks(fontsize=14, fontfamily="Times New Roman")
    plt.yticks(fontsize=14, fontfamily="Times New Roman")
    # 设置x/y轴标签字体
    plt.xlabel(r'$x$'+'/m', fontsize=14, fontfamily="Times New Roman")
    plt.ylabel(r'$y$'+'/m', fontsize=14, fontfamily="Times New Roman")

    # 暂停时间
    plt.pause(0.001)


def init(TOTSTEP: int, pos_wolves: np.ndarray, pos_targets: np.ndarray, **kwargs) -> Dict:
    draw_data = {}
    draw_data['wolves_path'] = np.array([np.array([np.array([pos_wolves[t][i][0] for t in range(TOTSTEP)]) , np.array([pos_wolves[t][i][1] for t in range(TOTSTEP)])]) for i in range(len(pos_wolves[0]))])
    draw_data['targets_path'] = np.array([np.array([np.array([pos_targets[t][i][0] for t in range(TOTSTEP)]) , np.array([pos_targets[t][i][1] for t in range(TOTSTEP)])]) for i in range(len(pos_targets[0]))])
    return draw_data


def main(opt):

    data = dict(np.load('output/2021_12_30_10_35_07.npz', allow_pickle=True))

    TOTSTEP = data['ori_targets'].size
    draw_data = init(TOTSTEP, **data)
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
                plot_all(t, **data, **draw_data)
                writer.grab_frame()
    else:
        t1 = time.time()
        # 仿真循环语句
        for t in range(TOTSTEP):
            # 清除当前绘图窗口中的所有对象
            plt.cla()
            # 绘图
            plot_all(t, **data, **draw_data)
    t2 = time.time()
    tact_time = t2-t1
    print(f'{tact_time} seconds, {t / tact_time} FPS')

    # 停止绘图并hold住窗口
    plt.ioff()
    plt.show()
    
    if opt.showdata:
        plot_data('v', **data)
        plot_data('w', **data)
        plot_data('E', **data)

if __name__ == '__main__':
    main(get_args())
