# Multi-Agent Cooperative Hunting ![](https://img.shields.io/badge/python-%3E%3D3.7-blue)
这是我在2020-2021年用python语言设计的群机器人的协同围捕的仿真工程。

## I. 描述

- 项目简介：本项目考虑的围捕任务是在有边界的场地中，有<img src="https://latex.codecogs.com/svg.image?n" title="n" />个机器人、<img src="https://latex.codecogs.com/svg.image?m_1" title="m_1" />个固定障碍物、<img src="https://latex.codecogs.com/svg.image?m_2" title="m_2" />个固定障碍物、<img src="https://latex.codecogs.com/svg.image?m_3" title="m_3" />个可旋转不规则障碍物、<img src="https://latex.codecogs.com/svg.image?m_4" title="m_4" />个移动可旋转不规则障碍物、1个目标，机器人群需要在该场地中避开障碍物并完成对目标的围捕，最终对目标形成以目标为圆心，以给定值<img src="https://latex.codecogs.com/svg.image?r_h" title="r_h" />为半径的圆形包围圈。
- 关键思路：机器人和目标均采用unicycle模型作为运动学模型，本项目主要设计的是机器人的控制输入<img src="https://latex.codecogs.com/svg.image?v" title="v" />、<img src="https://latex.codecogs.com/svg.image?\omega" title="\omega" />。
- 描述结果：机器人群需要在该场地中避开障碍物并完成对目标的围捕，最终对目标形成以目标为圆心，以给定值<img src="https://latex.codecogs.com/svg.image?r_h" title="r_h" />为半径的圆形包围圈。全过程使用matplotlib画图逐帧展示，同时也能输出视频和机器人、目标的位置变化。

## II. 使用方法

1. 环境、软件、特殊的依赖及其版本详细说明

- 环境： `Windows 10` 
- 软件：`Python 3.7`或更新版
- python依赖包：`numpy`、`yaml`、`matplotlib`、`numba`、`opencv-python`

python依赖包可以通过以下方式安装：

在`cmd`中输入

```bash
pip install XXX
```

XXX为`numpy`、`yaml`、`matplotlib`、`numba`、`opencv-python`等包。

2. 本项目只需将文件夹下载，在`cmd`中输入

```bash
python3 run.py
```

即可运行。

3. 本项目成功运行的结果：

<img src="Figure_1.png" alt="Figure_1" style="zoom: 50%;" />

出现以上界面并且画面正常刷新，即说明项目成功运行。

预期的图片在`./example/Figure_1.png`中，预期视频在`./output/2021_04_22.mp4`，输出数据文件的保存方式为txt，存储路径为`./output`。

### III. 注意事项

1. 要关闭绘图呈现功能，请将在`cmd`中的执行语句改为：

```bash
python3 run.py --display False
```

2. 要保存仿真过程各机器人和目标位置变化及其他相关数据，请将在`cmd`中的执行语句改为：

```bash
python3 run.py --output True
```

3. 要保存仿真过程的视频，请先安装FFmpeg，并将FFmpeg安装路径写到`run.py`第47行的`plt.rcParams['animation.ffmpeg_path'] =`后面，再将在`cmd`中的执行语句改为：

```bash
python3 run.py --record True
```

4. 要观察仿真过程中的机器人、目标的速度、角速度、能量消耗等变化曲线，请将在`cmd`中的执行语句改为：

```bash
python3 run.py --showdata True
```
