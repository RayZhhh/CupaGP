# This file defines cuda_device side functions and fit evaluation kernels,
# which allows evaluating multiple programs simultaneously.


import math
import sys
import time

import numba
from numba import cuda
import numpy as np
from .fset import *
from .program import Program

MAX_PIXEL_VALUE = 255
MAX_PROGRAM_LEN = 200
MAX_TOP = 10


# -The cuda_device side dataset is structured as follows:
#                [0 0 0]  [1 1 1]  [2 2 2]                   [0 0 0 0 0 0 0 0 0]
#     raw image: [0 0 0]  [1 1 1]  [2 2 2] ... => reshape => [1 1 1 1 1 1 1 1 1] => transpose =>
#                [0 0 0]  [1 1 1]  [2 2 2]                   [2 2 2 2 2 2 2 2 2]
#
#              [<---- data_size ---->]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#     dataset: [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]
#              [0 1 2 3 4 5 6 7 8 9 .]


@cuda.jit(device=True, inline=True)
def __dataset_value(dataset, data_size, im_w, i, j):
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return dataset[pixel_row * data_size + pixel_col]


# -Also a buffer for storing temp conv value is allocated, which is structured as follows:
#   The shape of stack is the same as the conv buffer.
#                  [<---- data_size ---->]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#     program 0 => [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [=====================]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#     program 1 => [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [0 1 2 3 4 5 6 7 8 9 .]
#                  [.....................]


@cuda.jit(device=True, inline=True)
def __pixel_index_in_stack(data_size, im_h, im_w, i, j) -> int:
    program_no = cuda.blockIdx.y
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * data_size * im_h * im_w + pixel_row * data_size + pixel_col


@cuda.jit(device=True, inline=True)
def __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j):
    return stack[__pixel_index_in_stack(data_size, im_h, im_w, i, j)]


@cuda.jit(device=True, inline=True)
def __pixel_conv_buffer_index(data_size, im_h, im_w, i, j) -> int:
    program_no = cuda.blockIdx.y
    pixel_row = i * im_w + j
    pixel_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * data_size * im_h * im_w + pixel_row * data_size + pixel_col


@cuda.jit(device=True, inline=True)
def __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j):
    return buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)]


# -The buffer to store the std value is structured as follows:
#     top=0 => [000000000000000000000]
#              [111111111111111111111]
#              [222222222222222222222]
#              [-------MAX TOP-------]
#     top=1 => [000000000000000000000]
#              [111111111111111111111]
#              [222222222222222222222]
#              [-------MAX TOP-------]
#              [.....................]


@cuda.jit(device=True, inline=True)
def __std_res_index(top, data_size) -> int:
    program_no = cuda.blockIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * MAX_TOP * data_size + top * data_size + col


@cuda.jit(device=True, inline=True)
def __std_res_value(std_res, top, data_size):
    return std_res[__std_res_index(top, data_size)]


@cuda.jit(device=True)
def _g_std(stack, data_size, im_h, im_w, rx, ry, rh, rw, std_res, top):
    """Calculate the standard deviation of the region."""
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        avg = 0
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                avg += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j)
        avg /= rh * rw
        deviation = 0
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                value = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) - avg
                deviation += value * value
        deviation /= rh * rw
        deviation = math.sqrt(deviation)
        std_res[__std_res_index(top, data_size)] = deviation


@cuda.jit(device=True)
def _sub(std_res, top, data_size):
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        res1 = __std_res_value(std_res, top - 2, data_size)
        res2 = __std_res_value(std_res, top - 1, data_size)
        std_res[__std_res_index(top - 2, data_size)] = res1 - res2


@cuda.jit(device=True)
def _region(dataset, data_size, im_h, im_w, stack):
    """Both Region_S and Region_R execute this function,
    the only difference is the region size which is stored in the local memory.
    This function copy the image from dataset to stack.
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(im_h):
            for j in range(im_w):
                d = __dataset_value(dataset, data_size, im_w, i, j)
                stack[__pixel_index_in_stack(data_size, im_h, im_w, i, j)] = d


@cuda.jit(device=True)
def _lap(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Achieve parallel Lap function.
    In this implementation, a thread is responsible for an image. This function uses top + 1 as buffer
    After Lap operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Laplacian kernel is: [0, 1, 0]
                             [1,-4, 1]
                             [0, 1, 0].
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        # calculate each pixel result
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 4
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j)
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True)
def _gau1(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Achieve parallel Gau1 function
    In this implementation, a thread is responsible for an image.
    After Gau1 operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Gaussian smooth kernel is: [1, 2, 1]
                                   [2, 4, 2] * (1 / 16).
                                   [1, 2, 1]
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 2
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 4
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                sum /= 16
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True)
def _sobel_x(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform Sobel_X on image.
    In this implementation, a thread is responsible for an image.
    After Sobel_X operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Sobel Vertical kernel is: [ 1, 2, 1]
                                  [ 0, 0, 0]
                                  [-1,-2,-1].
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True)
def _sobel_y(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform Sobel_Y on image.
    In this implementation, a thread is responsible for an image.
    After Sobel_Y operation, rx += 1; ry += 1; rh -= 2; rw -= 2.

    The Sobel Horizontal kernel is: [-1, 0, 1 ]
                                    [-2, 0, 2 ]
                                    [-1, 0, 1 ].
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 2
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True)
def _gau11(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform Gau11 on image.
    In this implementation, a thread is responsible for an image.
    After Gau11 operation, rx += 2; ry += 2; rh -= 4; rw -= 4.
    """
    _sobel_x(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer)
    _sobel_y(stack, data_size, im_h, im_w, rx + 1, ry + 1, rh - 2, rw - 2, buffer)


@cuda.jit(device=True)
def _gauxy(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform GauXY on image.
    In this implementation, a thread is responsible for an image.
    After GauXY operation, rx += 1; ry += 1; rh -= 2; rw -= 2.
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                sum = 0
                dx = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) - \
                     __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1)
                dy = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) - \
                     __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j)
                sum = math.sqrt(dx * dx + dy * dy)
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True)
def _log1(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                sum = 0
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j - 2) * 0.109
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j - 1) * 0.246
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j) * 0.270
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j + 1) * 0.246
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j + 2) * 0.109
                #
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 2) * 0.246
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 0.606
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 2) * 0.246
                #
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 2) * 0.270
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 0.606
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 2
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 0.606
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 2) * 0.270
                #
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 2) * 0.246
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 0.606
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 2) * 0.246
                #
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j - 2) * 0.109
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j - 1) * 0.246
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j) * 0.270
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j + 1) * 0.246
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j + 2) * 0.109
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True)
def _log2(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                sum = 0
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j - 1) * 0.1
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j) * 0.151
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j + 1) * 0.1
                #
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 2) * 0.1
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1) * 0.292
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 0.386
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1) * 0.292
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 2) * 0.1
                #
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 2) * 0.151
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 0.386
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 0.5
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 0.386
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 2) * 0.151
                #
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 2) * 0.1
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1) * 0.292
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 0.386
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1) * 0.292
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 2) * 0.1
                #
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j - 1) * 0.1
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j) * 0.151
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j + 1) * 0.1
                #
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 2, rx + rh - 2):
            for j in range(ry + 2, ry + rw - 2):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True)
def _lbp(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer):
    """Perform Local Binary Pattern operation to images.
    Step 1:
        calculate the value of each pixel based on the threshold
        pixel_lbp(i) = 0 if pixel(i) < center else 1

    Step 2:
        calculate the value of the center pixel using the weights: [  1,  2,  4]
                                                                   [128,  C,  8]
                                                                   [ 64, 32, 16]
    """
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                cp_value = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j)
                p_1 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1)
                p_2 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j)
                p_4 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1)
                p_8 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1)
                p_16 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1)
                p_32 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j)
                p_64 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1)
                p_128 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1)
                sum = 0
                sum += p_1 if p_1 >= cp_value else 0
                sum += p_2 * 2 if p_2 >= cp_value else 0
                sum += p_4 * 4 if p_4 >= cp_value else 0
                sum += p_8 * 8 if p_8 >= cp_value else 0
                sum += p_16 * 16 if p_16 >= cp_value else 0
                sum += p_32 * 32 if p_32 >= cp_value else 0
                sum += p_64 * 64 if p_64 >= cp_value else 0
                sum += p_128 * 128 if p_128 >= cp_value else 0
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum

        # copy the result from buffer to stack
        for i in range(rx + 1, rx + rh - 1):
            for j in range(ry + 1, ry + rw - 1):
                stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j)
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j)


@cuda.jit(device=True, inline=True)
def __hist_buffer_index(data_size, value) -> int:
    program_no = cuda.blockIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    return program_no * (MAX_PIXEL_VALUE + 1) * data_size + value * data_size + col


@cuda.jit(device=True, inline=True)
def __hist_buffer_value(buffer, data_size, value):
    return buffer[__hist_buffer_index(data_size, value)]


@cuda.jit(device=True)
def _hist_eq(stack, data_size, im_h, im_w, rx, ry, rh, rw, hist_buffer):
    """Performing Historical Equalisation to the input region."""
    # image index this thread is response for
    img_index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if img_index < data_size:
        # clear the buffer
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[__hist_buffer_index(data_size, i)] = 0

        # statistic intensity of each pixel, this process can not perform coalesced access
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                pixel_value = int(__pixel_value_in_stack(stack, data_size, im_h, im_w, i, j))
                pixel_value = max(0, pixel_value)
                pixel_value = min(MAX_PIXEL_VALUE, pixel_value)
                hist_buffer[__hist_buffer_index(data_size, pixel_value)] += 1

        # uniform
        pixel_num = rh * rw
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[__hist_buffer_index(data_size, i)] /= pixel_num

        # add up
        for i in range(1, MAX_PIXEL_VALUE + 1):
            hist_buffer[__hist_buffer_index(data_size, i)] += \
                hist_buffer[__hist_buffer_index(data_size, i - 1)]

        # mapping table
        for i in range(0, MAX_PIXEL_VALUE + 1):
            hist_buffer[__hist_buffer_index(data_size, i)] *= (MAX_PIXEL_VALUE - 1)

        # update the intensity value of each pixel of the region
        for i in range(rx, rx + rh):
            for j in range(ry, ry + rw):
                raw = int(__pixel_value_in_stack(stack, data_size, im_h, im_w, i, j))
                raw = max(0, raw)
                raw = min(MAX_PIXEL_VALUE, raw)
                new_value = __hist_buffer_value(hist_buffer, data_size, raw)
                stack[__pixel_index_in_stack(data_size, im_h, im_w, i, j)] = new_value
    cuda.syncthreads()


@cuda.jit()
def infer_population(name, rx, ry, rh, rw, plen, img_h, img_w, data_size, dataset, stack, conv_buffer, hist_buffer,
                     std_res):
    """Infer the predicted result for a population.
    This kernel allows evaluating multiple programs in the same time.
    Grid dims when launching this kernel: ((DATA_SIZE - 1 + THREAD_PER_BLOCK) // THREAD_PER_BLOCK, POP_SIZE_TO_EVAL).
    Block dims when launching this kernel: THREAD_PER_BLOCK.

    Args:
        name       : name of each node, a 2D array
        rx         : region x, a 2D array
        ry         : region y, a 2D array
        rh         : region h, a 2D array
        rw         : region 2, a 2D array
        plen       : a 1D array to store the length of each program in the population
        img_h      : image height
        img_w      : image width
        data_size  : dataset size
        dataset    : cuda_device side dataset
        stack      : cuda_device side buffer to store the intermediate results
        conv_buffer: cuda_device side buffer for conv
        hist_buffer: cuda_device side histogram buffer
        std_res    : store the intermediate std value for regions
    """
    # the program that the thread is responsible for
    program_no = cuda.blockIdx.y

    # top which point to the std_res
    top = 0
    reg_x, reg_y, reg_h, reg_w = 0, 0, 0, 0

    # reverse iteration
    for i in range(plen[program_no] - 1, -1, -1):
        if name[program_no][i] == Region_R:
            reg_x, reg_y, reg_h, reg_w = rx[program_no][i], ry[program_no][i], rh[program_no][i], rw[program_no][i]
            _region(dataset, data_size, img_h, img_w, stack)

        elif name[program_no][i] == Region_S:
            reg_x, reg_y, reg_h, reg_w = rx[program_no][i], ry[program_no][i], rh[program_no][i], rw[program_no][i]
            _region(dataset, data_size, img_h, img_w, stack)

        elif name[program_no][i] == G_Std:
            _g_std(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, std_res, top)
            top += 1

        elif name[program_no][i] == Hist_Eq:
            _hist_eq(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, hist_buffer)

        elif name[program_no][i] == Gau1:
            _gau1(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == Gau11:
            _gau11(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 2, reg_y + 2, reg_h - 4, reg_w - 4

        elif name[program_no][i] == GauXY:
            _gauxy(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == Lap:
            _lap(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == Sobel_X:
            _sobel_x(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == Sobel_Y:
            _sobel_y(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == LoG1:
            _log1(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 2, reg_y + 2, reg_h - 4, reg_w - 4

        elif name[program_no][i] == LBP:
            _lbp(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 1, reg_y + 1, reg_h - 2, reg_w - 2

        elif name[program_no][i] == LoG2:
            _log2(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer)
            reg_x, reg_y, reg_h, reg_w = reg_x + 2, reg_y + 2, reg_h - 4, reg_w - 4

        elif name[program_no][i] == HOG:
            pass

        elif name[program_no][i] == Sub:
            _sub(std_res, top, data_size)
            top -= 1

        else:
            print('Error: Do not support the function')

    cuda.syncthreads()


@numba.jit(nopython=True)
def _cal_accuracy(res, label, data_size: int, program_no: int) -> float:
    """Calculate accuracy for a program in the eval batch.
    A '@numba.jit()' decorator is imposed on this function to speed up the loop statement of python.
    """
    correct = 0
    for j in range(data_size):
        if label[j] > 0 and res[program_no][j] > 0 or label[j] <= 0 and res[program_no][j] <= 0:
            correct += 1
    return correct / data_size


class NumbaCudaEvaluator:
    def __init__(self, data, label, eval_batch=1, thread_per_block=128):
        """
        Args:
            data            : train-set or test-set
            label           : train-label or test-label
            eval_batch      : the number of programs evaluates simultaneously
            thread_per_block: equals to blockDim.x
        """
        self.data = data
        self.label = label
        self.data_size = len(data)
        self.img_h = len(self.data[0])
        self.img_w = len(self.data[0][0])
        self.eval_batch = eval_batch
        self.thread_per_block = thread_per_block
        self.max_top = MAX_TOP
        self.max_program_len = MAX_PROGRAM_LEN

        # cuda_device side arrays
        self._d_dataset = cuda.to_device(self.data.reshape(self.data_size, -1).T.reshape(1, -1).squeeze())
        self._d_stack = self._allocate_device_stack()
        self._d_hist = self._allocate_device_hist_buffer()
        self._d_res = self._allocate_device_res_buffer()
        self._d_conv_buffer = self._allocate_device_conv_buffer()

        # profiling
        self.cuda_kernel_time = 0

    def _allocate_device_stack(self):
        return cuda.device_array(self.data_size * self.img_h * self.img_w * self.eval_batch, float)

    def _allocate_device_conv_buffer(self):
        return cuda.device_array(self.data_size * self.img_h * self.img_w * self.eval_batch, float)

    def _allocate_device_hist_buffer(self):
        return cuda.device_array(self.data_size * (MAX_PIXEL_VALUE + 1) * self.eval_batch, float)

    def _allocate_device_res_buffer(self):
        return cuda.device_array(self.max_top * self.data_size * self.eval_batch)

    def _infer_population_for_a_batch(self, pop_batch: List[Program]):
        cur_batch_size = len(pop_batch)

        # the size of the current pop to be eval must <= the size of eval_batch
        if cur_batch_size > self.eval_batch:
            raise RuntimeError('Error: pop size > eval batch.')

        # allocate cuda_device side programs
        name = np.zeros((cur_batch_size, self.max_program_len), int)
        rx = np.zeros((cur_batch_size, self.max_program_len), int)
        ry = np.zeros((cur_batch_size, self.max_program_len), int)
        rh = np.zeros((cur_batch_size, self.max_program_len), int)
        rw = np.zeros((cur_batch_size, self.max_program_len), int)
        plen = np.zeros(cur_batch_size, int)  # an array stores the length of each program

        # parse the program
        for i in range(cur_batch_size):
            program = pop_batch[i]
            plen[i] = len(program)
            for j in range(len(program)):
                name[i][j] = program[j].name
                if program[j].is_terminal_node():
                    rx[i][j] = program[j].rx
                    ry[i][j] = program[j].ry
                    rh[i][j] = program[j].rh
                    rw[i][j] = program[j].rw

        # copy to cuda_device
        name = cuda.to_device(name)
        rx, ry, rh, rw = cuda.to_device(rx), cuda.to_device(ry), cuda.to_device(rh), cuda.to_device(rw)
        plen = cuda.to_device(plen)

        # launch kernel
        grid = ((self.data_size - 1 + self.thread_per_block) // self.thread_per_block, cur_batch_size)

        kernel_start = time.time()
        infer_population[grid, self.thread_per_block](name, rx, ry, rh, rw, plen, self.img_h, self.img_w,
                                                      self.data_size, self._d_dataset, self._d_stack,
                                                      self._d_conv_buffer, self._d_hist, self._d_res)
        cuda.synchronize()
        self.cuda_kernel_time += time.time() - kernel_start

    def _fitness_evaluate_for_a_batch(self, pop_batch: List[Program]):
        """Evaluate a population"""
        self._infer_population_for_a_batch(pop_batch)
        res = self._d_res.copy_to_host().reshape(self.eval_batch, -1)
        for i in range(len(pop_batch)):
            pop_batch[i].fitness = _cal_accuracy(res, self.label, self.data_size, i)

    def infer_program_and_get_feature_vector(self, population: List[Program]) -> np.ndarray:
        """Infer a population. The result is stored"""
        pass

    def evaluate_population(self, population: List[Program]):
        """Evaluate fitness for a whole population.
        Args:
            population: the population to be evaluated
        """
        for i in range(0, len(population), self.eval_batch):
            last_pos = min(i + self.eval_batch, len(population))
            self._fitness_evaluate_for_a_batch(population[i:last_pos])
