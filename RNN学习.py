#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""
import copy
import numpy as np

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


# 训练数据生成
int2binary = {}  # 十进制到二进制的查找表
binary_dim = 8  # 二进制数最大长度

largest_number = pow(2, binary_dim)  # 即256
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
learning_rate = 0.1  # 学习率
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1  # 权值，连接相邻两个时刻的隐含层

synapse_0_update = np.zeros_like(synapse_0)  # 存储权值更新
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

for j in range(1):

    a_int = np.random.randint(largest_number / 2)  # int version
    a = int2binary[a_int]  # binary encoding

    b_int = np.random.randint(largest_number / 2)  # int version
    b = int2binary[b_int]  # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()  # 每个时刻不断的记录layer2的导数值
    layer_1_values = list()  # 保存隐藏层L1每个时刻的数据
    layer_1_values.append(np.zeros(hidden_dim))  # 0时刻没有之前的隐含层，初始化一个全零的

    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])  # 输入[[x1 x2]]
        y = np.array([[c[binary_dim - position - 1]]]).T  # [[y]]

        # 计算隐藏层L1正向输出 (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # 计算输出层L2正向输出 (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))  # 预测值

        # 计算导数值
        layer_2_error = y - layer_2  # 计算误差
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))  # 导数值

        overallError += np.abs(layer_2_error[0])  # 每条数据的误差总和

        # 预测值d（对L2的输出四舍五入）
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # 存储隐藏层L1的数据
        layer_1_values.append(copy.deepcopy(layer_1))

    # 反向
    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])  # 从左向右作为输入数据
        layer_1 = layer_1_values[-position - 1]  #
        prev_layer_1 = layer_1_values[-position - 2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(
            synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * learning_rate
    synapse_1 += synapse_1_update * learning_rate
    synapse_h += synapse_h_update * learning_rate

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress
    if j % 1000 == 0:
        print("Error:" + str(overallError))
        print("预测值:" + str(d))
        print("真实值:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")

if __name__ == "__main__":
    pass
