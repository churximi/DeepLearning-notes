#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def sigmoid(x):
    """计算Sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))

    return s


def sigmoid_output_to_derivative(output):
    """
    sigmoid函数求导
    :param output:
    :return:
    """
    return output * (1 - output)


def create_data():
    # 训练数据生成
    X_train = []
    Y_train = []

    for j in range(10000):
        X_data = []
        Y_data = []
        a_int = np.random.randint(largest_number / 2)  # int version
        a = int2binary[a_int]  # binary encoding

        b_int = np.random.randint(largest_number / 2)  # int version
        b = int2binary[b_int]  # binary encoding

        # true answer
        c_int = a_int + b_int
        c = int2binary[c_int]

        for position in range(8):
            X_data.append([a[position], b[position]])
            Y_data.append([c[position]])

        X_train.append(X_data)
        Y_train.append(Y_data)

    return X_train, Y_train


def init_parameters():
    W1 = 2 * np.random.random((input_dim, hidden_dim)) - 1
    W2 = 2 * np.random.random((hidden_dim, output_dim)) - 1
    W_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1  # 权值，连接相邻两个时刻的隐含层

    return W1, W2, W_h


def compute_loss(y_hat, Y):
    """计算交叉熵loss
    """

    m = Y.shape[1]
    epsilon = pow(10.0, -9)  # 补加，作用是防止loss计算log时结果为nan
    loss = -(np.sum(Y * np.log(y_hat + epsilon) + (1 - Y) * np.log(1 - y_hat + epsilon))) / m

    return loss


def costs_draw(costs, learning_rate, hidden_dim):
    """绘制cost学习曲线
    :return:
    """

    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = {}, hidden_dim = {}".format(learning_rate, hidden_dim))
    plt.show()


learning_rate = 0.25  # 学习率
hidden_dim = 32
input_dim = 2
output_dim = 1

int2binary = {}  # 十进制到二进制的查找表
binary_dim = 8  # 二进制数最大长度

largest_number = pow(2, binary_dim)  # 即256
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

X_train, Y_train = create_data()
W1, W2, W_h = init_parameters()  # 权值W1，W2，W_h

costs = []
count = 0
overall_loss = 0
for X_data, Y_data in zip(X_train, Y_train):  # 数据个数
    dW1 = np.zeros_like(W1)
    dW2 = np.zeros_like(W2)
    dW_h = np.zeros_like(W_h)

    count += 1

    prediction = np.zeros(8, dtype=np.int8)  # 每条数据的预测
    overallError = 0  # 每条数据的误差总和

    layer_1_values = list()  # 每个时刻的L1输出值
    L1_initial = np.zeros(hidden_dim)
    layer_1_values.append(L1_initial)  # 0时刻没有之前的隐含层，初始化一个全零的（初始值）

    layer_2_deltas = list()  # 每个时刻layer2的导数值

    for position in range(binary_dim):
        X = np.array([X_data[binary_dim - position - 1]])
        y = np.array([Y_data[binary_dim - position - 1]])

        # 计算隐藏层L1正向输出 (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X, W1) + np.dot(layer_1_values[-1], W_h))
        layer_1_values.append(copy.deepcopy(layer_1))  # 存储隐藏层L1的数据

        # 计算输出层L2正向输出（预测值）
        p_y_hat = sigmoid(np.dot(layer_1, W2))  # 每个位置的预测值
        prediction[binary_dim - position - 1] = np.round(p_y_hat[0][0])  # 保存预测值（对L2的输出四舍五入）

        # 计算loss
        loss = y - p_y_hat
        overallError += np.abs(loss[0])
        overall_loss += compute_loss(np.round(p_y_hat[0][0]), y)

        # 计算导数值
        L2_delta = loss * sigmoid_output_to_derivative(p_y_hat)
        layer_2_deltas.append(L2_delta)  # 导数值

    # 反向传播
    future_layer_1_delta = np.zeros(hidden_dim)  # 初始值
    for position in range(binary_dim):
        X = np.array([X_data[position]])  # 正向最后一个时刻的输入值

        layer_1 = layer_1_values[-position - 1]  # 正向最后一个时刻的L1值
        prev_layer_1 = layer_1_values[-position - 2]  # 正向前一个时刻的L1值

        layer_2_delta = layer_2_deltas[-position - 1]  # 最后一个时刻L2导数值
        layer_1_delta = (future_layer_1_delta.dot(W_h.T) + layer_2_delta.dot(  # L1导数值
            W2.T)) * sigmoid_output_to_derivative(layer_1)

        dW2 += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        dW_h += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        dW1 += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    W1 += dW1 * learning_rate
    W2 += dW2 * learning_rate
    W_h += dW_h * learning_rate

    if count % 100 == 0:
        print("loss:" + str(overall_loss))
        costs.append(overall_loss)
        overall_loss = 0

        # print("------------")


costs_draw(costs, learning_rate, hidden_dim)

if __name__ == "__main__":
    pass
