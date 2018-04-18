#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：房价预测，线性回归
时间：2018年01月09日09:16:15
"""

import random
import tensorflow as tf

n_features = 4  # 特征数量
batch_size = 128  # batch大小
n_groups = 100000  # 数据组数，总数据 = batch_size * n_groups

W = tf.Variable(tf.ones([n_features, 1]) * 0.01, name="weights")  # 权重，4行1列
b = tf.Variable(0., name="bias")  # 初始值为0


def create_data():
    """
    随机生成符合一定函数的数据
    :return:
    """

    data_x = []
    data_y = []

    for j in range(n_groups):  # 生成n_groups组
        temp_x = []
        temp_y = []
        for i in range(batch_size):  # 每组生成batch_size个数据

            x0 = random.uniform(50, 200)  # 房屋面积（m^2）
            x1 = random.uniform(0.1, 5)  # 距离地铁距离（km）
            x2 = random.randint(1, 10)  # 附近学校数量（个）
            x3 = random.uniform(0.1, 20)  # 到市中心距离（km）

            w0 = 1  # 模拟参数1
            w1 = -2  # 模拟参数2
            w2 = 3  # 模拟参数3
            w3 = -4  # 模拟参数4

            bias = 5  # 模拟bias

            y = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 + bias
            temp_x.append([x0, x1, x2, x3])
            temp_y.append([y])

        data_x.append(temp_x)
        data_y.append(temp_y)

    return data_x, data_y


def add_placeholders():
    """生成输入张量的占位符
    """
    x_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_features))
    y_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))

    return x_placeholder, y_placeholder


def inference(x_placeholder):
    """计算推断模型在数据x上的输出，返回结果
    :param x_placeholder:
    :return:
    """
    return tf.matmul(x_placeholder, W) + b  # 矩阵乘法


def loss(x_placeholder, y_placeholder):
    """计算损失
    :param x_placeholder:输入数据
    :param y_placeholder:期望值/标准值
    :return:loss
    """
    y_predicted = inference(x_placeholder)  # y_predicted为预测值
    return tf.reduce_sum(tf.squared_difference(y_placeholder, y_predicted))  # L2范数/L2损失函数


def train(total_loss):
    """训练，采用梯度下降算法对模型参数进行优化，使loss最小化
    :param total_loss: 总的损失
    :return:
    """
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)  # 最小化loss


def evaluate(sess):
    """评估
    :param sess:
    :return:
    """
    print("模型预测结果：", sess.run(inference([[80., 3, 5, 10]])))
    print("实际标准结果：", 1 * 80 - 2 * 3 + 3 * 5 - 4 * 10 + 5)


def main():
    x_placeholder, y_placeholder = add_placeholders()

    new_w = tf.zeros([n_features, 1])
    new_b = 0.

    with tf.Session() as sess:
        init = tf.global_variables_initializer()  # 模型参数初始化
        sess.run(init)

        init_w = sess.run(W)
        init_b = sess.run(b)

        x, y = create_data()  # 读取数据、期望值

        total_loss = loss(x[0], y[0])  # 计算loss
        train_op = train(total_loss)  # train，调整参数

        print("初始loss：", sess.run(total_loss))
        print("初始方程 Y = {}*x0 + {}*x1 + {}*x2 + {}*x3 + {}".format
              (init_w[0][0], init_w[1][0], init_w[2][0], init_w[3][0], init_b))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 训练迭代
        training_steps = len(x)  # 数据组数
        for step in range(training_steps):
            feed_dict = {
                x_placeholder: x[step],
                y_placeholder: y[step],
            }

            sess.run(train_op, feed_dict=feed_dict)  # 训练，新的变量参数

            new_w = sess.run(W)
            new_b = sess.run(b)

            if step % 1000 == 0:
                print("loss：", sess.run(total_loss))  # loss
                print("方程 Y = {:.3f}*x0 + {:.3f}*x1 + {:.3f}*x2 + {:.3f}*x3 + {:.3f}".format
                      (new_w[0][0], new_w[1][0], new_w[2][0], new_w[3][0], new_b))

        print("最终参数：\n"
              "w1——{:.3f}\n"
              "w2——{:.3f}\n"
              "w3——{:.3f}\n"
              "w4——{:.3f}\n"
              "b——{:.3f}\n".format(new_w[0][0], new_w[1][0], new_w[2][0], new_w[3][0], new_b))

        evaluate(sess)  # 评估

        coord.request_stop()
        coord.join(threads)
        sess.close()

    return new_w, new_b


main()

if __name__ == "__main__":
    pass
