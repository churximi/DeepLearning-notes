#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from data_utils import load_data, costs_draw, read_image, load_dataset
from data_utils import data_draw, plot_decision_boundary, load_planar_dataset, load_extra_datasets
from basic_nn import basic_model, predict, evaluate


def linear_unit():
    # 加载数据
    X_train = np.array([[1], [2], [3], [4], [5]]).reshape(1, 5)
    Y_train = np.array([[100, 200.1, 299, 402, 500]])

    print(X_train.shape, Y_train.shape)
    # 模型参数设置
    layers_dims = [X_train.shape[0], 1]
    num_iter = 1000
    learning_rate = 0.001
    print_cost = False
    initialization = "he"

    # 调用模型
    parameters, costs = basic_model(X_train, Y_train,
                                    layers_dims=layers_dims,
                                    num_iter=num_iter,
                                    lr=learning_rate,
                                    print_cost=print_cost,
                                    initialization=initialization)
    # costs_draw(costs, learning_rate=learning_rate)
    print(parameters)


linear_unit()
if __name__ == "__main__":
    pass
