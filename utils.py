#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：吴恩达深度学习课程
时间：2018年04月18日17:51:25
"""

import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    """计算Sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    """计算sigmoid函数的导数
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def relu(x):
    """计算Relu(x)
    """
    s = np.maximum(0, x)
    return s


def dim_reshape():
    """维度转换
    :return:
    """
    x = np.random.random((2, 2, 3))
    y = x.reshape((3, 4))

    print(x)
    print("==维度转换后==")
    print(y)


def normalize_rows():
    """
    归一化每一行
    :return:
    """

    x = np.array([[0, 3, 4],
                  [3, 4, 0],
                  [6, 0, 8]])

    x_norm_rows = np.linalg.norm(x, axis=1, keepdims=True)  # 按行求模
    x_norm_cols = np.linalg.norm(x, axis=0, keepdims=True)  # 按列求模

    x_sum_rows = np.linalg.norm(x, ord=1, axis=1, keepdims=True)  # 按行求和
    x_sum_cols = np.linalg.norm(x, ord=1, axis=0, keepdims=True)  # 按列求和

    x_max_rows = np.linalg.norm(x, ord=np.inf, axis=1, keepdims=True)  # 取每行最大
    x_max_cols = np.linalg.norm(x, ord=np.inf, axis=0, keepdims=True)  # 取每列最大

    x_norm_byrow = x / x_norm_rows
    x_norm_bycol = x / x_norm_cols

    print(x_norm_byrow, "\n\n", x_norm_bycol)
    print(x_sum_rows, x_sum_cols)
    print(x_max_rows, x_max_cols)


def softmax():
    """实现softmax
    :return:
    """

    x = np.array([[9, 2, 5, 0, 0],
                  [7, 5, 0, 0, 0]])

    x_exp = np.exp(x)  # 每项求exp
    x_exp_sum_byrow = np.sum(x_exp, axis=1, keepdims=True)  # 按行求和
    # 等同于 np.linalg.norm(x_exp, ord=1, axis=1, keepdims=True)

    softmax_x = x_exp / x_exp_sum_byrow

    print(softmax_x)


def matrix_calculate():
    """矩阵计算
    :return:
    """

    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    # 计算1：x, y对应位置元素两两相乘并求和【55】

    # 【## 方法一】：for循环
    count = 0
    for i, j in zip(x, y):
        count += i * j

    # 【## 方法二】：np.dot()
    # 当x, y是一维向量的时候，np.dot()求的是内积。

    count2 = np.dot(x, y)
    print(count, count2)

    # 计算2：计算一个5*5的矩阵，第i行第j列的元素等于x中第i个元素乘以y中第j个元素

    # 【## 方法一】：for循环
    result = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            result[i, j] = x[i] * y[j]

    # 【## 方法二】：np.outer()
    result2 = np.outer(x, y)
    print(result2)

    # 计算3：计算一个1*5的行向量，每个元素等于x, y对应位置元素两两相乘

    # 【## 方法一】：for循环
    result = np.zeros(5)
    for i in range(len(x)):
        result[i] = x[i] * y[i]

    # 【## 方法二】：np.multiply()
    result2 = np.multiply(x, y)
    print(result, result2)

    # 计算4：矩阵相乘
    x = [[1, 2, 3],
         [2, 3, 4]]
    y = [[1, 2],
         [3, 4],
         [5, 6]]

    print(np.dot(x, y))


def loss_l1():
    """L1 loss计算，差的绝对值求和
    :return:
    """
    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    loss = np.sum(abs(y - yhat))

    print(loss)


def loss_l2():
    """L2 loss计算，差的绝对值的平方再求和
    :return:
    """
    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    loss = np.sum(np.dot((y - yhat), (y - yhat)))
    print(loss)


def main():
    x = np.array([1, 2, 3])
    print(sigmoid(x))

    print(sigmoid_derivative(x))

    dim_reshape()

    normalize_rows()

    softmax()

    matrix_calculate()

    loss_l1()


if __name__ == "__main__":
    loss_l2()

