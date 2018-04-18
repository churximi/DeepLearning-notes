#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：深层神经网络
时间：2018年02月01日13:46:39
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from data_utils import load_data, costs_draw, read_image, load_dataset
from data_utils import data_draw, plot_decision_boundary, load_planar_dataset, load_extra_datasets
from basic_nn import basic_model, predict, evaluate

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def different_lr(X_train, Y_train, X_test, Y_test):
    """
    测试不同的学习率
    :return:
    """
    learning_rates = [0.01, 0.001, 0.005, 0.0005, 0.0001]

    num_iter = 1500
    print_cost = False
    initialization = "zeros"

    for i in learning_rates:
        # 设置模型参数
        learning_rate = i
        layers_dims = [X_train.shape[0], 1]

        print("learning rate is: " + str(i))

        # 调用模型
        parameters, costs = basic_model(X_train, Y_train,
                                        layers_dims=layers_dims,
                                        num_iter=num_iter,
                                        lr=learning_rate,
                                        print_cost=print_cost,
                                        initialization=initialization)

        # 预测及评估
        prediction_train = predict(parameters, X_train)
        prediction_test = predict(parameters, X_test)

        print("Train准确率: {}".format(evaluate(prediction_train, Y_train)))
        print("test准确率: {}".format(evaluate(prediction_test, Y_test)))

        plt.plot(np.squeeze(costs), label=str(i))

    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def model_00():
    """
    猫图像识别，逻辑回归模型
    :return:
    """
    # 加载数据
    X_train, Y_train, X_test, Y_test, classes = load_data()

    # 模型参数设置
    layers_dims = [X_train.shape[0], 1]
    num_iter = 2000
    learning_rate = 0.005
    print_cost = True
    initialization = "zeros"

    # 调用模型
    parameters, costs = basic_model(X_train, Y_train,
                                    layers_dims=layers_dims,
                                    num_iter=num_iter,
                                    lr=learning_rate,
                                    print_cost=print_cost,
                                    initialization=initialization)

    costs_draw(costs, learning_rate=learning_rate)

    # 预测及评估
    prediction_train = predict(parameters, X_train)
    prediction_test = predict(parameters, X_test)

    print("Train准确率: {}".format(evaluate(prediction_train, Y_train)))
    print("test准确率: {}".format(evaluate(prediction_test, Y_test)))

    # 预测新图像
    new_image = "images/my_image.jpg"
    image_reshape = read_image(new_image, show_image=True)
    new_image_prediction = predict(parameters, image_reshape)
    print("新图像预测值 y = {}".format(int(np.squeeze(new_image_prediction))))

    # 不同学习率
    different_lr(X_train, Y_train, X_test, Y_test)


def model_01():
    """
    猫图像识别，两层网络结构
    :return:
    """
    X_train, Y_train, X_test, Y_test, classes = load_data()  # 猫图像数据

    # 模型参数设置
    layers_dims = [X_train.shape[0], 7, 1]
    num_iter = 2500
    learning_rate = 0.0075
    print_cost = True
    initialization = "sqrt_n"

    parameters, costs = basic_model(X_train, Y_train,
                                    layers_dims=layers_dims,
                                    num_iter=num_iter,
                                    lr=learning_rate,
                                    print_cost=print_cost,
                                    initialization=initialization)

    # 预测及评估
    prediction_train = predict(parameters, X_train)
    prediction_test = predict(parameters, X_test)

    print("Train准确率: {}".format(evaluate(prediction_train, Y_train)))
    print("test准确率: {}".format(evaluate(prediction_test, Y_test)))

    costs_draw(costs, learning_rate=learning_rate)


def model_02():
    """猫图像识别，四层网络结构
    :return:
    """

    X_train, Y_train, X_test, Y_test, classes = load_data()  # 猫图像数据

    # 模型参数设置
    layers_dims = [X_train.shape[0], 20, 7, 5, 1]
    num_iter = 2500
    learning_rate = 0.0075
    print_cost = True
    initialization = "sqrt_n"

    parameters, costs = basic_model(X_train, Y_train,
                                    layers_dims=layers_dims,
                                    num_iter=num_iter,
                                    lr=learning_rate,
                                    print_cost=print_cost,
                                    initialization=initialization)

    # 预测及评估
    prediction_train = predict(parameters, X_train)
    prediction_test = predict(parameters, X_test)

    print("Train准确率: {}".format(evaluate(prediction_train, Y_train)))
    print("test准确率: {}".format(evaluate(prediction_test, Y_test)))

    costs_draw(costs, learning_rate=learning_rate)

    # 预测新图像
    new_image = "images/my_image.jpg"
    image_reshape = read_image(new_image, show_image=True)
    new_image_prediction = predict(parameters, image_reshape)
    print("新图像预测值 y = {}".format(int(np.squeeze(new_image_prediction))))

    # 展示错误预测的图像
    # print_mislabeled_images(classes, test_x, test_y, pred_test)


def model_03():
    """
    单隐层平面数据分类
    :return:
    """
    # 加载数据集
    planar = load_planar_dataset()
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"planar": planar,
                "noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    data_set = "planar"  # 选择数据集
    X_train, Y_train = datasets[data_set]

    if data_set != "planar":
        X_train, Y_train = X_train.T, Y_train.reshape(1, Y_train.shape[0])

    if data_set == "blobs":
        Y_train = Y_train % 2

    # 绘制散点图
    data_draw(X_train, Y_train)

    # 模型参数设置
    layers_dims = [X_train.shape[0], 4, Y_train.shape[0]]
    num_iter = 10000
    learning_rate = 1.2
    print_cost = True
    initialization = "random_small"

    parameters, costs = basic_model(X_train, Y_train,
                                    layers_dims=layers_dims,
                                    num_iter=num_iter,
                                    lr=learning_rate,
                                    print_cost=print_cost,
                                    initialization=initialization)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X_train, Y_train)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    # 预测及评估
    prediction_train = predict(parameters, X_train)
    print("Train准确率: {}".format(evaluate(prediction_train, Y_train)))

    costs_draw(costs, learning_rate=learning_rate)


def model_04():
    """单隐层平面数据分类，测试不同的隐藏层size
    """
    # 加载数据集
    planar = load_planar_dataset()
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"planar": planar,
                "noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    data_set = "planar"  # 选择数据集
    X_train, Y_train = datasets[data_set]

    if data_set != "planar":
        X_train, Y_train = X_train.T, Y_train.reshape(1, Y_train.shape[0])

    if data_set == "blobs":
        Y_train = Y_train % 2

    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

    # 模型参数设置
    num_iter = 5000
    learning_rate = 1.2
    print_cost = False
    initialization = "random_small"

    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)

        layers_dims = [X_train.shape[0], n_h, Y_train.shape[0]]
        parameters, costs = basic_model(X_train, Y_train,
                                        layers_dims=layers_dims,
                                        num_iter=num_iter,
                                        lr=learning_rate,
                                        print_cost=print_cost,
                                        initialization=initialization)

        plot_decision_boundary(lambda x: predict(parameters, x.T), X_train, Y_train)

        # 预测及评估
        prediction_train = predict(parameters, X_train)
        accuracy = evaluate(prediction_train, Y_train)
        print("Accuracy for {} hidden units: {}".format(n_h, accuracy))

    plt.show()


def model_05():
    """使用sklearn现有的逻辑回归模型训练
    """
    # 加载数据集
    planar = load_planar_dataset()
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"planar": planar,
                "noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    data_set = "planar"  # 选择数据集
    X_train, Y_train = datasets[data_set]

    if data_set != "planar":
        X_train, Y_train = X_train.T, Y_train.reshape(1, Y_train.shape[0])

    if data_set == "blobs":
        Y_train = Y_train % 2

    clf = sklearn.linear_model.LogisticRegressionCV()
    true_y = np.squeeze(Y_train)
    clf.fit(X_train.T, true_y.T)

    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X_train, Y_train)
    plt.title("Logistic Regression")
    plt.show()

    # 预测及评估
    LR_predictions = clf.predict(X_train.T)
    accuracy = evaluate(LR_predictions, Y_train)

    print("准确率：", accuracy)


def model_06():
    # 加载数据集
    X_train, Y_train, X_test, Y_test = load_dataset()  # 数据

    # 设置参数
    layers_dims = [X_train.shape[0], 1]
    num_iter = 2000
    learning_rate = 0.5
    print_cost = False
    initialization = "he"

    parameters, costs = basic_model(X_train, Y_train,
                                    layers_dims=layers_dims,
                                    num_iter=num_iter,
                                    lr=learning_rate,
                                    print_cost=print_cost,
                                    initialization=initialization)

    # 预测及评估
    prediction_train = predict(parameters, X_train)
    prediction_test = predict(parameters, X_test)

    print("Train准确率: {}".format(evaluate(prediction_train, Y_train)))
    print("test准确率: {}".format(evaluate(prediction_test, Y_test)))

    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict(parameters, x.T), X_train, Y_train)
    plt.show()


def main():
    model_00()


main()

if __name__ == "__main__":
    pass
