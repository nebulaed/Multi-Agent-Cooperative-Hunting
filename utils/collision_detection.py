# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Hunting-Escape Model
# Written by 许博喆
# --------------------------------------------------------
# 本文件定义一个函数用于判断两条线段是否有交点，一个函数用于判断两个三角形是否相交，三个函数用于判断圆和三角形是否相交。

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from cv2 import pointPolygonTest
from utils.math_func import norm


@jit(nopython=True)
def two_line_segment_test(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float) -> bool:
    """
    判断输入两根线段是否有交点。

    输入：
        @param x1: 输入线段1的起点x坐标
        @param y1: 输入线段1的起点y坐标
        @param x2: 输入线段1的终点x坐标
        @param y2: 输入线段1的终点y坐标
        @param x3: 输入线段2的起点x坐标
        @param y3: 输入线段2的起点y坐标
        @param x4: 输入线段2的终点x坐标
        @param y4: 输入线段2的终点y坐标

    输出：
        @return: True表示两条线段有交点，False表示无交点
    """
    if ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)) == 0:
        return False  # ,[0,0]
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / \
        ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / \
        ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    if (x1 <= px <= x2 or x2 <= px <= x1) and (x3 <= px <= x4 or x4 <= px <= x3) and (y1 <= py <= y2 or y2 <= py <= y1) and (y3 <= py <= y4 or y4 <= py <= y3):
        return True  # ,[px, py]
    else:
        return False  # ,[0,0]


def two_triangle_test(tri1: np.ndarray, tri2: np.ndarray) -> bool:
    """
    判断输入的两个三角形是否有交集。

    输入：
        @param tri1: 形如[顶点1:[x,y],顶点2:[x,y],顶点3:[x,y]]的list
        @param tri2: 形如[顶点1:[x,y],顶点2:[x,y],顶点3:[x,y]]的list

    输出：
        @return: True表示有交集，False表示无交集
    """
    # 三角形1的边[边1:[起点:[x,y],终点:[x,y]], 边2:...]
    tri1_sides = [[[tri1[0][0], tri1[0][1]], [tri1[1][0], tri1[1][1]]],
                  [[tri1[1][0], tri1[1][1]], [tri1[2][0], tri1[2][1]]],
                  [[tri1[2][0], tri1[2][1]], [tri1[0][0], tri1[0][1]]]]
    # 三角形2的边[边1:[起点:[x,y],终点:[x,y]], 边2:...]
    tri2_sides = [[[tri2[0][0], tri2[0][1]], [tri2[1][0], tri2[1][1]]],
                  [[tri2[1][0], tri2[1][1]], [tri2[2][0], tri2[2][1]]],
                  [[tri2[2][0], tri2[2][1]], [tri2[0][0], tri2[0][1]]]]
    # 检查三角形1和三角形2的边是否相交
    for i in range(len(tri1_sides)):
        for j in range(len(tri2_sides)):
            if two_line_segment_test(tri1_sides[i][0][0], tri1_sides[i][0][1], tri1_sides[i][1][0], tri1_sides[i][1][1], tri2_sides[j][0][0], tri2_sides[j][0][1], tri2_sides[j][1][0], tri2_sides[j][1][1]):
                return True
    # 将三角形1转换为opencv能识别的多边形格式
    poly = np.zeros(((len(tri1), 1, 2))).astype(np.float32)
    for i in range(len(tri1)):
        poly[i, 0] = np.array([tri1[i][0], tri1[i][1]])
    # 利用opencv中的pointPolygonTest函数判断点是否在多边形外，1表示在多边形内，0表示在多边形边上，-1表示在多边形外
    for i in range(len(tri2)):
        retval = pointPolygonTest(poly, (np.float32(tri2[i][0]), np.float32(tri2[i][1])), measureDist=False)
        # 若三角形2的顶点在三角形1内或边上，说明两个小车相撞
        if retval == 1 or retval == 0:
            return True
    # 将三角形2转换为opencv能识别的多边形格式
    poly = np.zeros(((len(tri2), 1, 2))).astype(np.float32)
    for i in range(len(tri2)):
        poly[i, 0] = np.array([tri2[i][0], tri2[i][1]])
    # 利用opencv中的pointPolygonTest函数判断点是否在多边形外，1表示在多边形内，0表示在多边形边上，-1表示在多边形外
    for i in range(len(tri1)):
        retval = pointPolygonTest(poly, (np.float32(tri1[i][0]), np.float32(tri1[i][1])), measureDist=False)
        # 若三角形1的顶点在三角形2内或边上，说明两个小车相撞
        if retval == 1 or retval == 0:
            return True
    return False

@jit(nopython=True)
def get_foot_point(point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray):
    """
    计算点到直线的垂足坐标。

    输入：
        @param point: 点的坐标
        @param line_p1: 直线上任意两点之一坐标
        @param line_p2: 直线上任意两点之一坐标

    输出：
        @return: (xn, yn): 垂足坐标
    """
    x0 = point[0]
    y0 = point[1]
    z0 = 0  # point[2]

    x1 = line_p1[0]
    y1 = line_p1[1]
    z1 = 0  # line_p1[2]

    x2 = line_p2[0]
    y2 = line_p2[1]
    z2 = 0  # line_p2[2]

    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * 1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    zn = 0  # k * (z2 - z1) + z1

    return (xn, yn)

@jit(nopython=True)
def get_dis_point2line(point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray) -> float:
    """
    计算点到线段的距离。

    输入：
        @param point: 点的坐标
        @param line_p1: 线段起点坐标
        @param line_p2: 线段终点坐标
    输出：
        @return dist: 点到线段距离
    """
    footP = get_foot_point(point, line_p1, line_p2)
    if ((footP[0] - line_p1[0]) > 0) ^ ((footP[0] - line_p2[0]) > 0):  # 异或符号，符号不同是为1，,说明垂足落在直线中
        dist = norm(np.array([footP[0] - point[0], footP[1] - point[1]]))
    else:
        dist = min(norm(np.array([line_p1[0] - point[0], line_p1[1] - point[1]])),
                   norm(np.array([line_p2[0] - point[0], line_p2[1] - point[1]])))
    return dist

@jit(nopython=True)
def circle_triangle_test(center: np.ndarray, radius: float, tri: np.ndarray) -> bool:
    """
    判断输入的圆形和三角形是否有交集。

    输入：
        @param center: 圆心
        @param radius: 半径
        @param tri: 形如[顶点1:[x,y],顶点2:[x,y],顶点3:[x,y]]的三角形顶点list

    输出：
        @return: True表示有交集，False表示无交集
    """
    # 三角形的边[边1:[起点:[x,y],终点:[x,y]], 边2:...]
    tri_sides = np.array([[[tri[0][0], tri[0][1]], [tri[1][0], tri[1][1]]],
                 [[tri[1][0], tri[1][1]], [tri[2][0], tri[2][1]]],
                 [[tri[2][0], tri[2][1]], [tri[0][0], tri[0][1]]]])
    for i in range(3):
        if norm(np.array([tri[i][0]-center[0], tri[i][1]-center[1]])) <= radius:
            return True
    for i in range(3):
        if get_dis_point2line(center, tri_sides[i][0], tri_sides[i][1]) <= radius:
            return True
    return False


def two_polygon_test(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    判断输入的两个多边形是否有交集。思路：首先判断两个多边形的点是否在另一多边形内，然后判断两个多边形是否有边相交。

    输入：
        @param poly1: 多边形1
        @param poly2: 多边形2
    
    输出：
        @return: True表示有交集，False表示无交集
    """
    for i in range(poly1.shape[0]):
        # 利用opencv中的pointPolygonTest函数判断点是否在多边形外，1表示在多边形内，0表示在多边形边上，-1表示在多边形外
        retval = pointPolygonTest(poly2, (poly1[i, 0, 0], poly1[i, 0, 1]), measureDist = False)
        if retval == 1 or retval == 0:
            return True
    for i in range(poly2.shape[0]):
        # 利用opencv中的pointPolygonTest函数判断点是否在多边形外，1表示在多边形内，0表示在多边形边上，-1表示在多边形外
        retval = pointPolygonTest(poly1, (poly2[i, 0, 0], poly2[i, 0, 1]), measureDist = False)
        if retval == 1 or retval == 0:
            return True
    for i in range(poly1.shape[0]):
        for j in range(poly2.shape[0]):
            if i+1 == poly1.shape[0]:
                nextpt1 = 0
            else:
                nextpt1 = i+1
            if j+1 == poly2.shape[0]:
                nextpt2 = 0
            else:
                nextpt2 = j+1
            if two_line_segment_test(poly1[i, 0, 0], poly1[i, 0, 1], poly1[nextpt1, 0, 0], poly1[nextpt1, 0, 1], poly2[j, 0, 0], poly2[j, 0, 1], poly2[nextpt2, 0, 0], poly2[nextpt2, 0, 1]):
                return True
    return False



"""调试线段相交检测"""
# if __name__ == '__main__':
#     plt.figure()
#     L1 = [[8.191296551319033, 3.963838662354771],
#           [8.000003182679253, 4.079626602635465]]
#     L2 = [[8.276942620632184, 3.90124058871716],
#           [8.092506169040995, 4.027665251360376]]
#     plt.plot([L1[0][0], L1[1][0]], [L1[0][1], L1[1][1]],
#              linewidth=2.5, color='k')
#     plt.plot([L2[0][0], L2[1][0]], [L2[0][1], L2[1][1]],
#              linewidth=2.5, color='k')
#     intersect, intersection = two_line_segment_test(L1[0][0], L1[0][1], L1[1][0], L1[1][1], L2[0][0], L2[0][1], L2[1][0], L2[1][1])
#     if intersect:
#         ax = plt.gca()
#         ax.axis('equal')
#         print('相交，交点为:', intersection)
#         plt.plot(intersection[0], intersection[1], 'ro', markersize=10)
#         plt.show()
#     else:
#         print('不相交')
#         plt.show()

"""调试两个三角形碰撞检测"""
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     tri1 = [[-3.18364617942612, 9.389397096192521],
#             [-3.1451327045673216, 9.169132000786325],
#             [-3.336750170835798, 9.226427259061763]]
#     # tri1 = [[-3.229459491466627, 9.06743491395843], [-3.0800490105553564, 9.23379749446818], [-3.006723138512065, 9.047724077535262]]
#     tri2 = [[-3.229459491466627, 9.06743491395843],
#             [-3.0800490105553564, 9.23379749446818],
#             [-3.006723138512065, 9.047724077535262]]
#     # tri2 = [[-3.1594603272045108, 9.251192529287305], [-3.2719022671953506, 9.05791343023338], [-3.3815487704421536, 9.225178621847622]]
#     print('是否相撞', two_triangle_test(tri1, tri2))
#     plt.figure()
#     plt.plot([tri1[0][0], tri1[1][0]], [tri1[0][1], tri1[1][1]], linewidth=2.5, color='k')
#     plt.plot([tri1[1][0], tri1[2][0]], [tri1[1][1], tri1[2][1]], linewidth=2.5, color='k')
#     plt.plot([tri1[2][0], tri1[0][0]], [tri1[2][1], tri1[0][1]], linewidth=2.5, color='k')

#     plt.plot([tri2[0][0], tri2[1][0]], [tri2[0][1], tri2[1][1]], linewidth=2.5, color='k')
#     plt.plot([tri2[1][0], tri2[2][0]], [tri2[1][1], tri2[2][1]], linewidth=2.5, color='k')
#     plt.plot([tri2[2][0], tri2[0][0]], [tri2[2][1], tri2[0][1]], linewidth=2.5, color='k')
#     ax = plt.gca()
#     ax.axis('equal')
#     plt.show()

"""调试圆和三角形碰撞检测"""
# if __name__ == '__main__':
#     from math_func import norm, cos, sin
#     import matplotlib.pyplot as plt
#     center = [0, 5.29]
#     radius = 5
#     tri = [[-3.18364617942612, 9.389397096192521],
#            [-3.1451327045673216, 9.169132000786325],
#            [-3.336750170835798, 9.226427259061763]]
#     print(circle_triangle_test(center, radius, tri))
#     plt.figure()
#     theta = np.linspace(0, 6.28, 129)
#     Circle1 = center[0]+radius*cos(theta)
#     Circle2 = center[1]+radius*sin(theta)

#     plt.plot(Circle1, Circle2, 'k-', linewidth=1.0)

#     plt.plot([tri[0][0], tri[1][0]], [tri[0][1], tri[1][1]], linewidth=1.0, color='k')
#     plt.plot([tri[1][0], tri[2][0]], [tri[1][1], tri[2][1]], linewidth=1.0, color='k')
#     plt.plot([tri[2][0], tri[0][0]], [tri[2][1], tri[0][1]], linewidth=1.0, color='k')

#     ax = plt.gca()
#     ax.axis('equal')
#     plt.show()

"""调试两个多边形碰撞检测"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    poly1 = np.array([[[-3.18364617942612, 9.389397096192521]],
            [[-3.1451327045673216, 9.169132000786325]],
            [[-3.336750170835798, 9.226427259061763]]]).astype(np.float32)
    poly2 = np.array([[[-3.229459491466627, 9.06743491395843]],
            [[-3.0800490105553564, 9.23379749446818]],
            [[-3.006723138512065, 9.047724077535262]]]).astype(np.float32)
    print('是否相撞', two_polygon_test(poly1, poly2))
    plt.figure()
    plt.plot([poly1[0,0,0], poly1[1,0,0]], [poly1[0,0,1], poly1[1,0,1]], linewidth=2.5, color='k')
    plt.plot([poly1[1,0,0], poly1[2,0,0]], [poly1[1,0,1], poly1[2,0,1]], linewidth=2.5, color='k')
    plt.plot([poly1[2,0,0], poly1[0,0,0]], [poly1[2,0,1], poly1[0,0,1]], linewidth=2.5, color='k')

    plt.plot([poly2[0,0,0], poly2[1,0,0]], [poly2[0,0,1], poly2[1,0,1]], linewidth=2.5, color='k')
    plt.plot([poly2[1,0,0], poly2[2,0,0]], [poly2[1,0,1], poly2[2,0,1]], linewidth=2.5, color='k')
    plt.plot([poly2[2,0,0], poly2[0,0,0]], [poly2[2,0,1], poly2[0,0,1]], linewidth=2.5, color='k')
    ax = plt.gca()
    ax.axis('equal')
    plt.show()