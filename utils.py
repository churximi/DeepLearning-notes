#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
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


if __name__ == "__main__":
    pass
