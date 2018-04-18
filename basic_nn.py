#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：神经网络模型基本框架
时间：2018年02月17日21:07:14
"""

import numpy as np
from utils import sigmoid, relu


def init_zeros(layers_dims):
    """全零初始化
    不能打破对称（break symmetry），神经网络中的每个神经元都在计算同样的内容，
    使得神经网络并不比线性网络好多少（如逻辑回归）。
    :param layers_dims:各层维度
    :return:
    """

    parameters = {}
    L = len(layers_dims)  # 网络层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def init_random(layers_dims):
    """随机初始化，初始值太大会拖慢优化速度。
    """

    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def init_random_small(layers_dims):
    """小随机数初始化，W = 0.01 * np.random.randn(D,H)
    """

    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def init_he(layers_dims):
    """HE初始化（Xavier初始化的变种，多乘了√2，与Relu搭配效果好）
    """
    parameters = dict()
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def init_sqrt_n(layer_dims):
    """使用1/sqrt(layer_dims[l - 1])校准方差（Xavier初始化，与Relu很搭）
    """

    np.random.seed(3)
    parameters = dict()
    L = len(layer_dims)  # 神经网络层数

    for l in range(1, L):
        parameters['W{}'.format(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1 / layer_dims[l - 1])
        parameters['b{}'.format(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def init_parameters(layers_dims, initialization):
    """常见参数初始化方法
    :param layers_dims:各层维度
    :param initialization:初始化方法
    :return:
    """
    np.random.seed(3)

    if initialization in ["zeros", "random", "random_small", "he", "sqrt_n"]:
        print("参数初始化方法：{}".format(initialization))
    else:
        print("参数初始化方法：默认")

    parameters = None
    if initialization == "zeros":
        parameters = init_zeros(layers_dims)
    elif initialization == "random":
        parameters = init_random(layers_dims)
    elif initialization == "random_small":
        parameters = init_random_small(layers_dims)
    elif initialization == "he":
        parameters = init_he(layers_dims)
    elif initialization == "sqrt_n":
        parameters = init_sqrt_n(layers_dims)

    return parameters


def forward_propagation(X, parameters):
    """正向传播
    """

    cache = dict()
    L = len(parameters) // 2
    cache['A{}'.format(0)] = X  # A0 = X

    # 第1~(L-1)层，LINEAR -> RELU -> LINEAR -> RELU...
    for l in range(1, L):
        W = parameters['W{}'.format(l)]
        b = parameters['b{}'.format(l)]
        Z = np.dot(W, cache['A{}'.format(l - 1)]) + b  # Zl = Wl * A(l-1) + bl
        A = relu(Z)  # 该层的A值

        cache['W{}'.format(l)] = W
        cache['b{}'.format(l)] = b
        cache['Z{}'.format(l)] = Z
        cache['A{}'.format(l)] = A

    # 第L层，LINEAR -> SIGMOID
    WL = parameters['W{}'.format(L)]
    bL = parameters['b{}'.format(L)]
    ZL = np.dot(WL, cache['A{}'.format(L - 1)]) + bL
    AL = sigmoid(ZL)

    cache['W{}'.format(L)] = WL
    cache['b{}'.format(L)] = bL
    cache['Z{}'.format(L)] = ZL
    cache['A{}'.format(L)] = AL

    return AL, cache


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation:
    LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    """
    np.random.seed(1)

    cache = dict()
    L = len(parameters) // 2
    cache['A{}'.format(0)] = X  # A0 = X

    # 第1~(L-1)层，LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT...
    for l in range(1, L):
        W = parameters['W{}'.format(l)]
        b = parameters['b{}'.format(l)]
        Z = np.dot(W, cache['A{}'.format(l - 1)]) + b  # Zl = Wl * A(l-1) + bl
        A = relu(Z)  # 该层的A值

        D = np.random.rand(A.shape[0], A.shape[1])  # Step 1: initialize matrix D = np.random.rand(..., ...)
        D = (D < keep_prob)  # Step 2: convert entries of D to 0 or 1 (using keep_prob as the threshold)
        A = A * D  # Step 3: shut down some neurons of A
        A = A / keep_prob  # Step 4: scale the value of neurons that haven't been shut down

        cache['W{}'.format(l)] = W
        cache['b{}'.format(l)] = b
        cache['Z{}'.format(l)] = Z
        cache['A{}'.format(l)] = A
        cache['D{}'.format(l)] = D

    # 第L层，LINEAR -> SIGMOID
    WL = parameters['W{}'.format(L)]
    bL = parameters['b{}'.format(L)]
    ZL = np.dot(WL, cache['A{}'.format(L - 1)]) + bL
    AL = sigmoid(ZL)

    cache['W{}'.format(L)] = WL
    cache['b{}'.format(L)] = bL
    cache['Z{}'.format(L)] = ZL
    cache['A{}'.format(L)] = AL

    return AL, cache


def compute_loss(y_hat, Y):
    """计算交叉熵loss
    """

    m = Y.shape[1]
    epsilon = pow(10.0, -9)  # 补加，作用是防止loss计算log时结果为nan
    loss = -(np.sum(Y * np.log(y_hat + epsilon) + (1 - Y) * np.log(1 - y_hat + epsilon))) / m

    return loss


def compute_loss_with_regularization(y_hat, Y, parameters, lambd):
    """L2正则化计算成本函数
    """
    m = Y.shape[1]
    L = len(parameters) // 2

    cross_entropy_cost = compute_loss(y_hat, Y)
    temp = 0.
    for l in range(1, L + 1):
        temp += np.sum(np.square(parameters["W{}".format(l)]))  # np.square（元素计算平方）

    L2_regularization_cost = lambd * temp / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def backward_propagation(X, Y, cache):
    """反向传播
    :param X: 输入数据，shape (input size, number of examples)
    :param Y: 真实标签（0，1向量）
    :param cache: 正向传播中的缓存（A，Z，W，b），用于计算梯度
    :return: gradients，各梯度值（dA，dZ，dW，db）
    """

    m = X.shape[1]  # 实例个数
    L = (len(cache) - 1) // 4  # 层数，cache中包含A0

    gradients = dict()

    gradients["dZ{}".format(L)] = cache["A{}".format(L)] - Y
    gradients["dW{}".format(L)] = np.dot(gradients["dZ{}".format(L)], cache["A{}".format(L - 1)].T) / m
    gradients["db{}".format(L)] = np.sum(gradients["dZ{}".format(L)], axis=1, keepdims=True) / m

    for l in reversed(range(1, L)):  # reversed()
        gradients["dA{}".format(l)] = np.dot(cache["W{}".format(l + 1)].T, gradients["dZ{}".format(l + 1)])
        gradients["dZ{}".format(l)] = np.multiply(gradients["dA{}".format(l)], np.int64(cache["A{}".format(l)] > 0))
        gradients["dW{}".format(l)] = np.dot(gradients["dZ{}".format(l)], cache["A{}".format(l - 1)].T) / m
        gradients["db{}".format(l)] = np.sum(gradients["dZ{}".format(l)], axis=1, keepdims=True) / m

    return gradients


def backward_propagation_with_regularization(X, Y, cache, lambd):
    """L2正则化实现反向传播
    """

    m = X.shape[1]
    L = (len(cache) - 1) // 4  # 层数，cache中包含A0
    temp = lambd / m

    gradients = dict()

    gradients["dZ{}".format(L)] = cache["A{}".format(L)] - Y

    gradients["dW{}".format(L)] = np.dot(gradients["dZ{}".format(L)], cache["A{}".format(L - 1)].T) / m
    addition = temp * cache["W{}".format(L)]
    gradients["dW{}".format(L)] += addition

    gradients["db{}".format(L)] = np.sum(gradients["dZ{}".format(L)], axis=1, keepdims=True) / m

    for l in reversed(range(1, L)):
        gradients["dA{}".format(l)] = np.dot(cache["W{}".format(l + 1)].T, gradients["dZ{}".format(l + 1)])
        gradients["dZ{}".format(l)] = np.multiply(gradients["dA{}".format(l)], np.int64(cache["A{}".format(l)] > 0))

        gradients["dW{}".format(l)] = np.dot(gradients["dZ{}".format(l)], cache["A{}".format(l - 1)].T) / m
        addition = temp * cache["W{}".format(l)]
        gradients["dW{}".format(l)] += addition

        gradients["db{}".format(l)] = np.sum(gradients["dZ{}".format(l)], axis=1, keepdims=True) / m

    return gradients


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    带dropout的反向传播
    """

    m = X.shape[1]
    L = (len(cache)) // 5  # 层数，cache中包含A0，但缺少DL

    gradients = dict()

    gradients["dZ{}".format(L)] = cache["A{}".format(L)] - Y
    gradients["dW{}".format(L)] = np.dot(gradients["dZ{}".format(L)], cache["A{}".format(L - 1)].T) / m
    gradients["db{}".format(L)] = np.sum(gradients["dZ{}".format(L)], axis=1, keepdims=True) / m

    for l in reversed(range(1, L)):
        gradients["dA{}".format(l)] = np.dot(cache["W{}".format(l + 1)].T, gradients["dZ{}".format(l + 1)])
        gradients["dA{}".format(l)] = gradients["dA{}".format(l)] * cache["D{}".format(l)]
        gradients["dA{}".format(l)] = gradients["dA{}".format(l)] / keep_prob

        gradients["dZ{}".format(l)] = np.multiply(gradients["dA{}".format(l)], np.int64(cache["A{}".format(l)] > 0))
        gradients["dW{}".format(l)] = np.dot(gradients["dZ{}".format(l)], cache["A{}".format(l - 1)].T) / m
        gradients["db{}".format(l)] = np.sum(gradients["dZ{}".format(l)], axis=1, keepdims=True) / m

    return gradients


def update_parameters(parameters, grads, learning_rate):
    """更新参数，梯度下降法
    """

    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W{}".format(l)] = parameters["W{}".format(l)] - learning_rate * grads["dW{}".format(l)]
        parameters["b{}".format(l)] = parameters["b{}".format(l)] - learning_rate * grads["db{}".format(l)]

    return parameters


def predict(parameters, X):
    """输入新的X，用最终训练出的参数w，b来预测输出
    """

    y_hat, _ = forward_propagation(X, parameters)
    predictions = (y_hat > 0.5) * 1.0

    return predictions


def evaluate(prediction, Y):
    m = Y.shape[1]
    accuracy = np.sum(prediction == Y) / m

    return accuracy


def basic_model(X, Y, layers_dims, num_iter=2000, lr=0.5, print_cost=False,
                initialization="he", lambd=0., keep_prob=1.):
    """神经网络基本模型
    :param X: 训练集数据
    :param Y: 训练集标签
    :param layers_dims:各层维度
    :param num_iter: 迭代次数
    :param lr: 学习率
    :param print_cost: 是否输出cost
    :param initialization:参数初始化方法
    :param lambd:正则化参数
    :param keep_prob:
    :return:训练好的参数和costs记录
    """

    costs = []

    # 选择初始化方法
    parameters = init_parameters(layers_dims, initialization)

    # 循环 (梯度下降)
    for i in range(num_iter):
        # 正向传播: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        y_hat, cache = "", ""
        if keep_prob == 1:
            y_hat, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            y_hat, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # 计算Loss
        if lambd == 0:
            cost = compute_loss(y_hat, Y)
        else:
            cost = compute_loss_with_regularization(y_hat, Y, parameters, lambd)

        # 反向传播
        grads = ""
        assert (lambd == 0 or keep_prob == 1)  # L2正则化和dropout可以同时用，不过这里确保一次只用一种

        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # 更新参数
        parameters = update_parameters(parameters, grads, lr)

        # 输出/记录cost
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))

        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs


if __name__ == "__main__":
    pass
