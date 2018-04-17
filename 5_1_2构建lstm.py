#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：构建LSTM前向传播计算
时间：2018年04月17日14:44:21
"""

import numpy as np
from utils import sigmoid, softmax


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """ 实现LSTM-cell前向传播

    Arguments:
    xt -- t 时刻的输入x，shape = (n_x, m).
    a_prev -- t-1 时刻的隐层状态，shape = (n_a, m)
    c_prev -- t-1 时刻的记忆状态, shape = (n_a, m)
    parameters -- 参数:
                        Wf -- 遗忘门权重，shape = (n_a, n_a + n_x)
                        bf -- 遗忘门偏置，shape = (n_a, 1)
                        Wi -- 更新门权重，shape = (n_a, n_a + n_x)
                        bi -- 更新门偏置，shape = (n_a, 1)
                        Wc -- c权重，    shape = (n_a, n_a + n_x)
                        bc -- c偏置，    shape = (n_a, 1)
                        Wo -- 输出门权重，shape = (n_a, n_a + n_x)
                        bo -- 输出门偏置，shape = (n_a, 1)
                        Wy -- 预测值权重，shape = (n_y, n_a)
                        by -- 预测值偏置，shape = (n_y, 1)

    Returns:
    a_next -- 下一刻隐层状态， (n_a, m)
    c_next -- 下一刻记忆状态， (n_a, m)
    yt_pred -- 当前时刻预测值， (n_y, m)
    cache -- (a_next, c_next, a_prev, c_prev, xt, parameters)，用于反向传播计算
    """

    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 连接 xt 和 a_prev
    concat = np.zeros((n_a + n_x, m))
    concat[n_a:, :] = xt
    concat[:n_a, :] = a_prev

    # 前向传播计算
    ft = sigmoid(np.matmul(Wf, concat) + bf)  # 遗忘门，控制过去记忆
    it = sigmoid(np.matmul(Wi, concat) + bi)  # 更新门，控制当前候选记忆
    ot = sigmoid(np.matmul(Wo, concat) + bo)  # 输出门

    # 计算c_next
    cct = np.tanh(np.matmul(Wc, concat) + bc)  # c_hat，候选c
    c_next = ft * c_prev + it * cct

    # 计算a_next
    a_next = ot * np.tanh(c_next)

    # 计算预测值 yt_pred
    yt_pred = softmax(np.matmul(Wy, a_next) + by)

    # 存储，用于反向传播计算
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """ 实现LSTM前向传播

    Arguments:
    x -- 所有时刻的输入，shape = (n_x, m, T_x)，T_x表示时刻数
    a0 -- 给定一个初始隐层状态，shape = (n_a, m)
    parameters -- 参数:
                        Wf -- 遗忘门权重，shape = (n_a, n_a + n_x)
                        bf -- 遗忘门偏置，shape = (n_a, 1)
                        Wi -- 更新门权重，shape = (n_a, n_a + n_x)
                        bi -- 更新门偏置，shape = (n_a, 1)
                        Wc -- c权重，    shape = (n_a, n_a + n_x)
                        bc -- c偏置，    shape = (n_a, 1)
                        Wo -- 输出门权重，shape = (n_a, n_a + n_x)
                        bo -- 输出门偏置，shape = (n_a, 1)
                        Wy -- 预测值权重，shape = (n_y, n_a)
                        by -- 预测值偏置，shape = (n_y, 1)

    Returns:
    a -- 所有时刻的隐层状态，(n_a, m, T_x)
    y -- 所有时刻的预测值， (n_y, m, T_x)
    caches -- (list of caches, x)，第一项存储所有时刻的cache，第二项存储输入x
    """

    caches = []

    n_x, m, T_x = x.shape  # 输入x（n_x个特征，m个实例，每个实例有T_x个时刻）
    n_y, n_a = parameters['Wy'].shape

    # 初始化a、c、y，用于更新和存储 T_x 个时刻的隐层状态、记忆状态和预测值
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # 初始化启动
    a_prev = a0
    c_prev = np.zeros(a_prev.shape)

    # 当 T_x = 0 时
    x0 = x[:, :, 0]
    a_0, c_0, y_0, cache = lstm_cell_forward(x0, a_prev, c_prev, parameters)

    # 保存
    a[:, :, 0] = a_0
    c[:, :, 0] = c_0
    y[:, :, 0] = y_0
    caches.append(cache)

    # loop over all time-steps
    for t in range(1, T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        xt = x[:, :, t]
        a_prev = a[:, :, t - 1]
        c_prev = c[:, :, t - 1]
        a_t, c_t, y_t, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

        # 保存
        a[:, :, t] = a_t
        c[:, :, t] = c_t
        y[:, :, t] = y_t

        # Append the cache into caches (≈1 line)
        caches.append(cache)

    caches = (caches, x)

    return a, y, c, caches


def main():
    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)
    print("a[4][3][6] = ", a[4][3][6])
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])
    print("y.shape = ", y.shape)
    print("caches[1][1[1]] =", caches[1][1][1])
    print("c[1][2][1]", c[1][2][1])
    print("len(caches) = ", len(caches))


if __name__ == "__main__":
    main()
