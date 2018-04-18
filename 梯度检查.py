#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：梯度检验，整合了斯坦福作业代码和吴恩达课程作业代码
时间：2018年04月09日08:55:49
"""

import numpy as np
from basic_nn import init_parameters, forward_propagation, compute_loss, backward_propagation


def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
    """
    梯度检验
    :param parameters:所有参数
    :param gradients:计算好的解析梯度
    :param X:输入值
    :param Y:真实labels
    :param epsilon:epsilon，斯坦福取1e-4（通常），吴恩达课程取1e-7
    :return:
    """

    error_count = 0
    for para in parameters:
        p = parameters[para]
        p_iter = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
        while not p_iter.finished:
            # 每次更新parameters里的其中一个参数（如W1）的其中一个值（如W1[0][0])
            parameters_temp = parameters
            index = p_iter.multi_index  # 索引
            p[index] += epsilon  # 将索引处的x值修改为 x_orig + epsilon
            parameters_temp[para] = p  # 修改后的参数替换原来的参数

            y_hat_plus, _ = forward_propagation(X, parameters_temp)
            J_plus = compute_loss(y_hat_plus, Y)

            p[index] -= 2 * epsilon  # 将索引处的x值修改为 x_orig - epsilon
            parameters_temp[para] = p
            y_hat_minus, _ = forward_propagation(X, parameters_temp)
            J_minus = compute_loss(y_hat_minus, Y)  # 计算J_minus

            gradapprox = (J_plus - J_minus) / (2 * epsilon)

            # 计算梯度差异
            grad = gradients["d{}".format(para)][index]  # 查找实际解析梯度
            numerator = np.linalg.norm(grad - gradapprox)  # 分子, 计算【平方和再开平方，即模】
            denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # 分母
            difference = numerator / (denominator + 1e-10)  # 防止分母为0

            if difference > 2e-7:
                print("d{}{} 处梯度差异 {} >= 2e-7 ，错误!".format(para, index, difference))
                error_count += 1

            p_iter.iternext()

    if error_count == 0:
        print("梯度检验通过！")
    else:
        print("【注意】梯度检验未通过！有 {} 处错误".format(error_count))


def main():
    np.random.seed(1)
    X_train = np.random.randn(4, 3)
    Y_train = np.array([[1, 1, 0]])
    layers_dims = [X_train.shape[0], 5, 3, 1]
    parameters = init_parameters(layers_dims, "he")

    y_hat, cache = forward_propagation(X_train, parameters)
    gradients = backward_propagation(X_train, Y_train, cache)
    gradient_check(parameters, gradients, X_train, Y_train)


main()

if __name__ == "__main__":
    pass
