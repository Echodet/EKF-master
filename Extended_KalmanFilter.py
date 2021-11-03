# -*- coding:utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator

# 模型初始条件
kx, ky = 0.01, 0.05
g = 9.8
De = np.matrix([[1.5 ** 2, 0],
                [0, 1.5 ** 2]])

D_delta = np.matrix([[1, 0],
                     [0, 1e-5]])

DT = 0.1  # time int [s] 
file_data = np.genfromtxt("observation.txt", usecols=(0, 1),
                          skip_header=1, dtype=float)
R = file_data[:, 0]
rad = file_data[:, 1]

dv_r = []
dv_a = []

delta_V = [np.matrix([0, 0, 0, 0]).T]


def timeupdate(hX, hD):
    vx, vy = hX[1, 0], hX[3, 0]
    # 状态转移矩阵（由xk-1->hX得到
    State_TransM = np.matrix([[1, 1 * 0.1, 0, 0],
                              [0, 1 - 2 * kx * vx * 0.1, 0, 0],
                              [0, 0, 1, 1 * 0.1],
                              [0, 0, 0, 1 + 2 * ky * vy * 0.1]])

    # 积分部分
    Fi = np.matrix([[DT, DT ** 2 / 2, 0, 0],
                    [0, DT - kx * vx * DT ** 2, 0, 0],
                    [0, 0, DT, DT ** 2 / 2],
                    [0, 0, 0, DT + ky * vy * DT ** 2]])

    G = Fi * np.transpose(np.matrix([0, kx * vx ** 2, 0, -ky * vy ** 2 - g]))

    delta_x, delta_y = math.sqrt(De[0, 0]), math.sqrt(De[1, 1])
    # 系统噪声方差
    Dw = np.matrix([[DT ** 3 / 3 * delta_x ** 2, (DT ** 2 / 2 - 2 * kx * vx * DT ** 3 / 3) * delta_x ** 2, 0, 0],
                    [(DT ** 2 / 2 - 2 * kx * vx * DT ** 3 / 3) * delta_x ** 2,
                     (3 * DT - 6 * kx * vx * DT ** 2 + 4 * (kx * vx) ** 2 * DT ** 3) / 3 * delta_x ** 2, 0, 0],
                    [0, 0, DT ** 3 / 3 * delta_y ** 2, (DT ** 2 / 2 + 2 * ky * vy * DT ** 3 / 3) * delta_y ** 2],
                    [0, 0, (DT ** 2 / 2 + 2 * ky * vy * DT ** 3 / 3) * delta_y ** 2,
                     (3 * DT + 6 * ky * vy * DT ** 2 + 4 * (ky * vy) ** 2 * DT ** 3) / 3 * delta_y ** 2]])

    # 时间预测
    X_Pred = State_TransM * hX + G
    Dx_Pred = State_TransM * hD * np.transpose(State_TransM) + Dw

    return X_Pred, Dx_Pred


def mre_ud(X_pred, Dx_pred, k):
    X1, Dx1 = X_pred, Dx_pred
    # 第1个观测值的新息矩阵
    x, y = X1[0, 0], X1[2, 0]
    Z1 = R[k]
    V1 = Z1 - math.sqrt(x ** 2 + y ** 2)

    dv_r.append(V1)
    # 第1个观测值的观测矩阵
    h1 = np.matrix([x / math.sqrt(x ** 2 + y ** 2), 0, y / math.sqrt(x ** 2 + y ** 2), 0])

    # 第1个观测值的增益矩阵
    K1 = Dx1 * np.transpose(h1)
    t=np.matrix(h1 * Dx1 * np.transpose(h1) + D_delta[0, 0] / DT)
    t=t.I
    K1*=t
    # 量测更新
    X2 = X1 + K1 * V1
    Dx2 = (np.eye(4) - K1 * h1) * Dx1

    # 第2个观测值的新息矩阵
    Z2 = rad[k]
    x, y = X2[0, 0], X2[2, 0]
    V2 = Z2 - math.atan(x / y)
    dv_a.append(V2)
    # 第2个观测值的观测矩阵
    h2 = np.matrix([(1. / y) / (1. + (x / y) ** 2), 0, (-1. * x / (y ** 2)) / (1. + (x / y) ** 2), 0])

    # 第2个观测值的增益矩阵
    K2 = Dx2 * np.transpose(h2) * np.matrix(h2 * Dx2 * np.transpose(h2) + D_delta[1, 1] / DT).I
    # 量测更新
    X3 = X2 + K2 * V2
    delta_V.append(K2 * V2 + K1 * V1)
    D3 = (np.eye(4) - K2 * h2) * Dx2
    return X3, D3


def main():
    X_estimate = np.transpose(np.matrix([0, 50, 500, 0]))
    Dx_estimate = np.matrix([[100, 0, 0, 0],
                             [0, 100, 0, 0],
                             [0, 0, 100, 0],
                             [0, 0, 0, 100]])

    k = 0
    # 估计轨迹
    est_trajectory_x = []
    est_trajectory_y = []
    # 预测轨迹
    pre_trajectory_x = []
    pre_trajectory_y = []
    # 量测轨迹
    mre_trajectory_x = []
    mre_trajectory_y = []
    # 滤波方差统计
    statc_x = []
    statc_y = []

    delta_X = []

    est_trajectory_x.append(X_estimate[0, 0])
    est_trajectory_y.append(X_estimate[2, 0])

    statc_x.append(Dx_estimate[0, 0])
    statc_y.append(Dx_estimate[3, 3])

    while k < len(R):
        X_pred, DX_Prex = timeupdate(X_estimate, Dx_estimate)
        X_estimate, Dx_estimate = mre_ud(X_pred, DX_Prex, k)
        # 记录关键变量
        est_trajectory_x.append(X_estimate[0, 0])
        est_trajectory_y.append(X_estimate[2, 0])
        pre_trajectory_x.append(X_pred[0, 0])
        pre_trajectory_y.append(X_pred[2, 0])
        mre_trajectory_y.append(R[k] * math.cos(rad[k]))
        mre_trajectory_x.append(R[k] * math.sin(rad[k]))
        statc_x.append(Dx_estimate[0, 0])
        statc_y.append(Dx_estimate[2, 2])
        delta_X.append(X_estimate - X_pred)
        k += 1

    # 绘图部分
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc",size=20)
    # plt.cla()
    plt.figure("抛体轨迹图")
    plt.scatter(np.array(est_trajectory_x),
                np.array(est_trajectory_y), s=75, c='w', edgecolors='r', marker='s', linewidths=1)
    plt.scatter(np.array(pre_trajectory_x),
                np.array(pre_trajectory_y), s=75, c='w', edgecolors='c', marker='o', linewidths=1)

    plt.scatter(np.array(mre_trajectory_x),
                np.array(mre_trajectory_y), s=75, c='w', edgecolors='b', marker='*', linewidths=1)
    # plt.plot(mre_trajectory_x,
    #          mre_trajectory_y, "-r")
    # plt.axis("equal")
    plt.xlim(-1, 200)
    plt.grid(True)

    plt.legend(["滤波值", "预测值", "观测值"], prop=my_font, loc="upper right",fontsize=50)
    plt.xlabel('X/m',fontsize=20)
    plt.ylabel('Y/m',fontsize=20)
    # plt.title("Extended Kalman Filter with Observations Updated Step-by-Step")
    plt.plot(est_trajectory_x,
             est_trajectory_y, "-r")
    plt.plot(pre_trajectory_x,
             pre_trajectory_y, "-c")
    # 
    plt.figure("状态滤波方差")
    plt.scatter(range(len(statc_x)),
                statc_x, s=75, c='w', edgecolors='b', marker='*', linewidths=1)
    plt.scatter(range(len(statc_y)),
                statc_y, s=75, c='w', edgecolors='r', marker='o', linewidths=1)
    plt.legend(["X的方差", "Y的方差"], prop=my_font, loc="upper right",fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('m^2',fontsize=20)
    plt.plot(range(len(statc_x)),
             statc_x, "-b")

    plt.plot(range(len(statc_y)),
             statc_y, "-r")
    plt.xlim(-1, 100)
    plt.grid(True)
    # 
    plt.figure("观测值新息")
    plt.subplot(211)
    plt.scatter(range(len(dv_r)),
                dv_r, s=75, c='w', edgecolors='b', marker='o', linewidths=1)
    plt.legend(["r的新息"], prop=my_font, loc="upper right")
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('m',fontsize=20)
    plt.ylim(-50, 50)
    plt.xlim(-1, 100)
    plt.grid(True)
    plt.subplot(212)
    plt.scatter(range(len(dv_a)),
                dv_a, s=75, c='w', edgecolors='r', marker='*', linewidths=1)
    plt.legend(["a的新息"], prop=my_font, loc="upper right")
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('rad',fontsize=20)
    plt.ylim(-0.05, 0.05)
    plt.xlim(-1, 100)
    plt.grid(True)
    plt.subplot(211)
    plt.plot(range(len(dv_r)),
             dv_r, "-b")
    plt.subplot(212)
    plt.plot(range(len(dv_a)),
             dv_a, "-r")

    delta = delta_V
    plt.figure("观测值对状态的增益")
    X = [line[0, 0] for line in delta]
    Vx = [line[1, 0] for line in delta]
    Y = [line[2, 0] for line in delta]
    Vy = [line[3, 0] for line in delta]
    plt.subplot(221)
    plt.plot(range(len(delta)), X, "-r", marker="o", markersize=2, linewidth=0.5)
    plt.legend(["X的增益"], prop=my_font, loc="upper right")
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('m',fontsize=20)
    plt.grid(True)
    y_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-5, 5)
    plt.xlim(0, 100)
    plt.subplot(222)
    plt.plot(range(len(delta)), Vx, "-g", marker="o", markersize=2, linewidth=0.5)
    plt.legend(["Vx的增益"], prop=my_font, loc="upper right")
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('m/s',fontsize=20)
    plt.grid(True)
    y_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-5, 5)
    plt.xlim(0, 100)
    plt.subplot(223)
    plt.plot(range(len(delta)), Y, "-b", marker="o", markersize=2, linewidth=0.5)
    plt.legend(["Y的增益"], prop=my_font, loc="upper right")
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('m',fontsize=20)
    plt.grid(True)
    y_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-5, 5)
    plt.xlim(0, 100)
    plt.subplot(224)
    plt.plot(range(len(delta)), Vy, "-k", marker="o", markersize=2, linewidth=0.5)
    plt.legend(["Vy的增益"], prop=my_font, loc="upper right")
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('m/s',fontsize=20)
    plt.grid(True)
    y_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-5, 5)
    plt.xlim(0, 100)
    plt.pause(1000)

if __name__ == '__main__':
    main()
