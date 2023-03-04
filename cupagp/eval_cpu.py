# This file defines GP operators for CPU.
import copy
import math
import sys

import numpy as np
from numpy import ndarray
from numba import jit

from .fset import *
from .program import Program

MAX_PIXEL_VALUE = 255


@jit(nopython=True)
def __conv2d_3x3(region: ndarray, kernel) -> ndarray:
    buffer = np.zeros(shape=(len(region), len(region[0])), dtype=float)
    for i in range(0, len(region) - 2):
        for j in range(0, len(region[0]) - 2):
            conv_value = 0
            for ki in range(len(kernel)):
                for kj in range(len(kernel[0])):
                    kernel_value = kernel[ki][kj]
                    pix_value = region[i + ki][j + kj]
                    conv_value += pix_value * kernel_value
            buffer[i + 1][j + 1] = conv_value
    return buffer[1:-1, 1:-1]


@jit(nopython=True)
def __conv2d_5x5(region: ndarray, kernel) -> ndarray:
    buffer = np.zeros(shape=(len(region), len(region[0])), dtype=float)
    for i in range(0, len(region) - 4):
        for j in range(0, len(region[0]) - 4):
            conv_value = 0
            for ki in range(len(kernel)):
                for kj in range(len(kernel[0])):
                    kernel_value = kernel[ki][kj]
                    pix_value = region[i + ki][j + kj]
                    conv_value += pix_value * kernel_value
            buffer[i + 2][j + 2] = conv_value
    return buffer[2:-2, 2:-2]


@jit(nopython=True)
def _g_std(region: ndarray) -> float:
    if len(region) == 0:
        return 0
    std = float(np.std(region))
    return std


@jit(nopython=True)
def _hist_eq(region: ndarray):
    """Histogram Equalization"""
    buffer = np.zeros(shape=(len(region), len(region[0])), dtype=float)
    hist_buffer = np.zeros(shape=MAX_PIXEL_VALUE + 1, dtype=float)
    pixel_num = len(region) * len(region[0])

    for i in range(len(region)):
        for j in range(len(region[0])):
            pix_val = int(region[i][j])
            pix_val = max(0, pix_val)
            pix_val = min(MAX_PIXEL_VALUE, pix_val)
            hist_buffer[pix_val] += 1

    for i in range(1, len(hist_buffer)):
        hist_buffer[i] += hist_buffer[i - 1]

    for i in range(0, len(hist_buffer)):
        hist_buffer[i] /= pixel_num

    for i in range(len(region)):
        for j in range(len(region[0])):
            pix_val = int(region[i][j])
            pix_val = max(0, pix_val)
            pix_val = min(MAX_PIXEL_VALUE, pix_val)
            new_val = hist_buffer[pix_val] * 255
            buffer[i][j] = new_val
    return buffer


@jit(nopython=True)
def _lap(region):
    """
    The Laplacian kernel is: [0, 1, 0]
                             [1,-4, 1]
                             [0, 1, 0].
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return __conv2d_3x3(region, kernel)


@jit(nopython=True)
def _sobel_x(region):
    """The Sobel Vertical kernel is: [ 1, 2, 1]
                                     [ 0, 0, 0]
                                     [-1,-2,-1].
    """
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return __conv2d_3x3(region, kernel)


@jit(nopython=True)
def _sobel_y(region):
    """The Sobel Horizontal kernel is: [-1, 0, 1 ]
                                       [-2, 0, 2 ]
                                       [-1, 0, 1 ].
    """
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return __conv2d_3x3(region, kernel)


@jit(nopython=True)
def _gau1(region):
    """
    The Gaussian smooth kernel is: [1, 2, 1]
                                   [2, 4, 2] * (1 / 16).
                                   [1, 2, 1]
    """
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    return __conv2d_3x3(region, kernel)


@jit(nopython=True)
def _log1(region):
    kernel = [[0.109, 0.246, 0.270, 0.246, 0.109], [0.246, 0, -0.606, 0, 0.246], [0.270, -0.606, -2., -0.606, 0.270],
              [0.246, 0, -0.606, 0, 0.246], [0.109, 0.246, 0.270, 0.246, 0.109]]
    return __conv2d_5x5(region, kernel)


@jit(nopython=True)
def _log2(region):
    kernel = [[0, -0.1, -0.151, -0.1, 0], [-0.1, -0.292, -0.386, -0.292, -0.1], [-0.151, -0.386, -0.5, -0.386, -0.151],
              [-0.1, -0.292, -0.386, -0.292, -0.1], [0, -0.1, -0.151, -0.1, 0]]
    return __conv2d_5x5(region, kernel)


@jit(nopython=True)
def _lbp(region):
    """Perform Local Binary Pattern operation to images.
    Step 1:
        calculate the value of each pixel based on the threshold
        pixel_lbp(i) = 0 if pixel(i) < center else 1

    Step 2:
        calculate the value of the center pixel using the weights: [  1,  2,  4]
                                                                   [128,  C,  8]
                                                                   [ 64, 32, 16]
    """
    buffer = np.zeros(shape=(len(region), len(region[0])), dtype=float)

    for i in range(1, len(region) - 1):
        for j in range(1, len(region[0]) - 1):
            sum = 0
            center_px = region[i][j]
            if region[i - 1][j - 1] >= center_px:
                sum += 1
            if region[i - 1][j] >= center_px:
                sum += 2
            if region[i - 1][j + 1] >= center_px:
                sum += 4
            if region[i][j + 1] >= center_px:
                sum += 8
            if region[i + 1][j + 1] >= center_px:
                sum += 16
            if region[i + 1][j] >= center_px:
                sum += 32
            if region[i + 1][j - 1] >= center_px:
                sum += 64
            if region[i][j - 1] >= center_px:
                sum += 128
            buffer[i][j] = sum
    return buffer


@jit(nopython=True)
def _gau11(region):
    """Perform Gau11 on image.
    After Gau11 operation, rx += 2; ry += 2; rh -= 4; rw -= 4.
    """
    kernel0 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    kernel1 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    reg1 = __conv2d_3x3(region, kernel0)
    return __conv2d_3x3(reg1, kernel1)


@jit(nopython=True)
def _gauxy(region):
    buffer = np.zeros(shape=(len(region), len(region[0])), dtype=float)
    for i in range(0, len(region) - 2):
        for j in range(0, len(region[0]) - 2):
            dx = region[i][j + 1] - region[i][j - 1]
            dy = region[i + 1][j] - region[i - 1][j]
            buffer[i][j] = math.sqrt(dx * dx + dy * dy)
    return buffer[1:-1, 1:-1]


def infer_program(program: Program, img: ndarray) -> float:
    stack = []
    region: ndarray = ...

    # reverse iteration
    for node in reversed(program.prefix):
        rx, ry, rh, rw = node.rx, node.ry, node.rh, node.rw

        if node.name == Region_R or node.name == Region_S:
            region = img[rx:rx + rh, ry:ry + rw]

        elif node.name == G_Std:
            try:
                __reg = _g_std(region)
            except BaseException:
                ...
            stack.append(_g_std(region))

        elif node.name == Hist_Eq:
            region = _hist_eq(region)

        elif node.name == Gau1:
            region = _gau1(region)

        elif node.name == Gau11:
            region = _gau11(region)

        elif node.name == GauXY:
            region = _gauxy(region)

        elif node.name == Lap:
            region = _lap(region)

        elif node.name == Sobel_X:
            region = _sobel_x(region)

        elif node.name == Sobel_Y:
            region = _sobel_y(region)

        elif node.name == LoG1:
            region = _log1(region)

        elif node.name == LoG2:
            region = _log2(region)

        elif node.name == LBP:
            region = _lbp(region)

        elif node.name == HOG:
            pass

        elif node.name == Sub:
            std1 = stack.pop()
            std2 = stack.pop()
            stack.append(std2 - std1)

    assert len(stack) == 1
    return stack.pop()


class CPUEvaluator:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.data_size = len(data)

    def evaluate_program(self, program: Program):
        correct = 0
        for i in range(len(self.data)):
            res = infer_program(program, self.data[i])
            if res <= 0 and self.label[i] <= 0 or res > 0 and self.label[i] > 0:
                correct += 1
        program.fitness = correct / self.data_size

    def evaluate_population(self, population: List[Program]):
        for program in population:
            self.evaluate_program(program)
