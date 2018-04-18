#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：正则化
时间：2018年02月19日12:57:43
"""

import matplotlib.pyplot as plt
from data_utils import load_2D_dataset, plot_decision_boundary, costs_draw
from basic_nn import basic_model, predict, evaluate

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model_reg(lambd, keep_prob, title):
    # 加载数据
    X_train, Y_train, X_test, Y_test = load_2D_dataset()

    # 模型参数设置
    layers_dims = [X_train.shape[0], 20, 3, 1]
    num_iter = 30000
    learning_rate = 0.3
    print_cost = False
    initialization = "sqrt_n"

    # 调用模型
    parameters, costs = basic_model(X_train, Y_train,
                                    layers_dims=layers_dims,
                                    num_iter=num_iter,
                                    lr=learning_rate,
                                    print_cost=print_cost,
                                    initialization=initialization,
                                    lambd=lambd,
                                    keep_prob=keep_prob)

    costs_draw(costs, learning_rate=learning_rate)

    # 预测及评估
    prediction_train = predict(parameters, X_train)
    prediction_test = predict(parameters, X_test)

    print("Train准确率: {}".format(evaluate(prediction_train, Y_train)))
    print("test准确率: {}".format(evaluate(prediction_test, Y_test)))

    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict(parameters, x.T), X_train, Y_train)


def main():
    # 不使用正则化
    lambd = 0.
    keep_prob = 1.0
    title = "Model without regularization"
    model_reg(lambd, keep_prob, title)

    # L2正则化
    lambd = 0.7
    keep_prob = 1.0
    title = "Model with L2-regularization"
    model_reg(lambd, keep_prob, title)

    # dropout
    lambd = 0.
    keep_prob = 0.86
    title = "Model with dropout"
    model_reg(lambd, keep_prob, title)


main()

if __name__ == "__main__":
    pass
