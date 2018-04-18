#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：吴恩达深度学习课整理，猫图像识别，数据处理相关
时间：2018年01月23日15:43:25
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io
from skimage import transform
from scipy import ndimage


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.get_cmap("Spectral"))
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.get_cmap("Spectral"))
    plt.show()


def data_draw(X, Y):
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.get_cmap("Spectral"))  # 绘制散点图
    plt.show()


def data_check(X, Y):
    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[1]  # 训练集数据个数

    print('The shape of X is: {}'.format(shape_X))
    print('The shape of Y is: {}'.format(shape_Y))
    print('训练数据个数 = {} 个'.format(m))


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


def load_data():
    """从h5文件中加载数据（猫图像分类）
    :return:
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 训练集数据 209个，64*64*3维度
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 训练集标签 209个，一维

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 测试集数据 50个，64*64*3维度
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 测试集标签 50个，一维

    classes = np.array(test_dataset["list_classes"][:])  # 测试集类别列表，[b'non-cat' b'cat']

    # 将图像数据reshape
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # （12288, 209）
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T  # （12288, 50）

    # 标准化数据
    X_train = train_set_x_flatten / 255.
    X_test = test_set_x_flatten / 255.

    Y_train = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # (1, 209)
    Y_test = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # (1, 50)

    return X_train, Y_train, X_test, Y_test, classes


def load_dataset():
    """
    加载数据
    :return:
    """
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)

    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.get_cmap("Spectral"))
    plt.show()

    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))

    return train_X, train_Y, test_X, test_Y


def picture_show(index, train_set_x_orig, train_set_y, classes):
    """
    训练集图像示例展示
    :param index:图片索引编号
    :param train_set_x_orig:原始训练集数据
    :param train_set_y:原始训练集标签
    :param classes:两种类别，[b'non-cat' b'cat']
    :return:
    """

    plt.imshow(train_set_x_orig[index])
    label = train_set_y[0][index]
    print("图像标签：{}\nclass = {}".format(label, classes[label].decode("utf-8")))

    plt.show()


def costs_draw(costs, learning_rate):
    """绘制cost学习曲线
    :return:
    """

    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def read_image(image_path, show_image=False):
    """读入图像并预处理为可用数据
    :param image_path:图像路径
    :param show_image:是否展示图像
    :return:处理后的图像
    """
    num_px = 64
    image_read = np.array(ndimage.imread(image_path, flatten=False))  # 原始图片
    image_resize = transform.resize(image_read, (num_px, num_px), mode="reflect")
    image_reshape = image_resize.reshape((1, num_px * num_px * 3)).T

    if show_image:
        plt.rcParams['figure.figsize'] = (5.0, 4.0)
        plt.imshow(image_resize)
        plt.show()

    return image_reshape


def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.get_cmap("Spectral"))
    plt.show()

    return train_X, train_Y, test_X, test_Y


if __name__ == "__main__":
    pass
