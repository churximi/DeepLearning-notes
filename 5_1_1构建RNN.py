#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：一步步构建RNN
时间：2018年04月16日13:11:29
"""

import numpy as np
from utils import softmax


def rnn_cell_forward(xt, a_prev, parameters):
    """ RNN_cell前向传播计算

    Arguments:
    xt -- t 时刻的输入x, shape = (n_x, m).
    a_prev -- t-1 时刻的隐层状态, shape = (n_a, m)
    parameters -- 参数:
                        Wax -- (n_a, n_x)，与 xt (n_x, m) 矩阵相乘
                        Waa -- (n_a, n_a)，与 a_prev (n_a, m) 矩阵相乘
                        Wya -- (n_y, n_a)，与 a_next (n_a, m) 矩阵相乘
                        ba --  (n_a, 1)，计算 a_next
                        by --  (n_y, 1)，计算 yt_pred
    Returns:
    a_next -- (n_a, m)，当前时刻的隐层状态，为下一个时刻所用，取决于当前输入和上一个隐层状态
    yt_pred -- (n_y, m)，当前时刻的预测值
    cache -- (a_next, a_prev, xt, parameters)，用于反向传播计算
    """

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算a_next
    a_next = np.tanh(np.matmul(Wax, xt) + np.matmul(Waa, a_prev) + ba)

    # 计算yt_pred
    yt_pred = softmax(np.matmul(Wya, a_next) + by)

    # 存储，用于反向传播计算
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """ RNN前向传播计算（循环cell）

    Arguments:
    x -- 所有时刻的输入，shape = (n_x, m, T_x)，T_x表示时刻数
    a0 -- 给定一个初始隐层状态，shape = (n_a, m)
    parameters -- 参数：
                        Wax -- (n_a, n_x)，与 xt (n_x, m) 矩阵相乘
                        Waa -- (n_a, n_a)，与 a_prev (n_a, m) 矩阵相乘
                        Wya -- (n_y, n_a)，与 a_next (n_a, m) 矩阵相乘
                        ba --  (n_a, 1)，计算 a_next
                        by --  (n_y, 1)，计算 yt_pred

    Returns:
    a -- 所有时刻的隐层状态，(n_a, m, T_x)
    y_pred -- 所有时刻的预测值， (n_y, m, T_x)
    caches -- (list of caches, x)，第一项存储所有时刻的cache，第二项存储输入x
    """

    caches = []

    n_x, m, T_x = x.shape  # 输入x（n_x个特征，m个实例，每个实例有T_x个时刻）
    n_y, n_a = parameters["Wya"].shape

    # 初始化a 和 y_pred ，用于更新和存储 T_x 个时刻的状态和预测值
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # 用a0初始化a_next
    a_prev = a0

    # 当 T_x = 0 时
    x0 = x[:, :, 0]
    a0, y0_pred, cache = rnn_cell_forward(x0, a_prev, parameters)
    a[:, :, 0] = a0
    y_pred[:, :, 0] = y0_pred
    caches.append(cache)

    # 当 T_x >= 1时
    for t in range(1, T_x):  # 对于每个时刻t，取t时刻的x，执行rnn_cell_forward
        xt = x[:, :, t]  # 当前输入x
        a_prev = a[:, :, t - 1]
        at, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)

        a[:, :, t] = at  # 保存当前隐层状态
        y_pred[:, :, t] = yt_pred  # 保存当前预测值yt_pred

        caches.append(cache)  # 保存当前cache

    caches = (caches, x)  # 注意保存的内容

    return a, y_pred, caches


def main():
    np.random.seed(1)
    x = np.random.randn(3, 10, 4)
    a0 = np.random.randn(5, 10)

    Waa = np.random.randn(5, 5)
    Wax = np.random.randn(5, 3)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    a, y_pred, caches = rnn_forward(x, a0, parameters)
    print("a[4][1] = ", a[4][1])
    print("a.shape = ", a.shape)
    print("y_pred[1][3] =", y_pred[1][3])
    print("y_pred.shape = ", y_pred.shape)
    print("caches[1][1][3] =", caches[1][1][3])
    print("len(caches) = ", len(caches))


if __name__ == "__main__":
    main()
