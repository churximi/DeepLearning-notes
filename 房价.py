#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：2018年01月09日09:16:15
"""

import random
import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.Variable(0., name="weights")  # 权重，2行1列（列向量，即[0， 0]的专指）
b = tf.Variable(0., name="bias")  # 初始值为0


def create_data():
    data_x = []
    data_y = []

    for i in range(100):
        x0 = random.uniform(50, 200)  # 面积
        unit_price = random.uniform(1.1, 1.3)  # 单价，1.2万/平米
        y = x0 * unit_price + 100
        data_x.append([x0])
        data_y.append([y])

    return data_x, data_y


def draw():
    data_x, data_y = create_data()
    x = [d[0] for d in data_x]
    y = [d[0] for d in data_y]
    x_max = max(x)
    x_min = min(x)

    w0, b0 = main()

    x1 = [x_min, x_max]
    y1 = [w0 * x_min + b0, w0 * x_max + b0]
    plt.scatter(x, y)
    plt.plot(x1, y1, color="r")
    plt.show()


def inputs():
    """输入
    :return:
    """
    area, price = create_data()

    return tf.to_float(area), tf.to_float(price)


def inference(x):
    """计算推断模型在数据x上的输出，返回结果
    :param x:
    :return:
    """
    return x * W + b  # 矩阵乘法


def loss(x, y):
    """计算损失
    :param x:输入数据
    :param y:期望值/标准值
    :return:loss
    """
    y_predicted = inference(x)  # x为输入数据，y为期望值，y_predicted为预测值
    return tf.reduce_sum(tf.squared_difference(y, y_predicted))  # L2范数/L2损失函数


def train(total_loss):
    """训练，采用梯度下降算法对模型参数进行优化，使loss最小化
    :param total_loss: 总的损失
    :return:
    """
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)  # 最小化loss


def evaluate(sess, x, y):
    """评估
    :param sess:
    :param x:
    :param y:
    :return:
    """
    print(sess.run(inference([[80.]])))
    print(sess.run(inference([[65.]])))


def main():
    new_w = 0.
    new_b = 0.
    with tf.Session() as sess:
        init = tf.global_variables_initializer()  # 模型参数初始化
        sess.run(init)

        init_w = sess.run(W)
        init_b = sess.run(b)

        x, y = inputs()  # 读取数据、期望值
        print(x, y)
        total_loss = loss(x, y)  # 计算loss
        train_op = train(total_loss)  # train，调整参数

        print("初始loss：", sess.run(total_loss))  # 2559050
        print("初始方程 Y=%s*weight + %s" % (init_w, init_b))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 实际训练迭代次数
        training_steps = 100000
        for step in range(training_steps):
            sess.run(train_op)  # 训练，新的变量参数

            new_w = sess.run(W)
            new_b = sess.run(b)
            if step % 10000 == 0:
                print("loss：", sess.run(total_loss))  # loss
                print("方程 Y=%s*weight + %s" % (new_w, new_b))

        evaluate(sess, x, y)  # 评估

        coord.request_stop()
        coord.join(threads)
        sess.close()

    return new_w, new_b


draw()

if __name__ == "__main__":
    pass
