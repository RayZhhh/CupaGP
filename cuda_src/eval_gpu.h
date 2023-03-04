//
// Created by Derek on 2022/11/15.
//

#ifndef CUDAGPIBC_EVAL_GPU_H
#define CUDAGPIBC_EVAL_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>
#include <utility>
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include <iostream>
#include <vector>
#include "program.h"


// the max value of pixels
// this value is for histogram based functions
#define MAX_PIXEL_VALUE 255

// max length of programs
// the kernel would crash if a program's length exceeds this value
#define MAX_PROGRAM_LEN 200

// the max stack size for evaluating
// the kernel would crash if the evaluation of a program need more stack spaces
// in default cases where the max depth is set to 8, a max top of 10 is enough (actually 8 is enough)
#define MAX_STACK_SIZE 10


__device__ inline
float __dataset_value(float *dataset, int data_size, int im_w, int i, int j) {
    int pixel_row = i * im_w + j;
    int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    return dataset[pixel_row * data_size + pixel_col];
}


__device__ inline
int __pixel_index_in_stack(int data_size, int im_h, int im_w, int i, int j) {
    int program_no = blockIdx.y;
    int pixel_row = i * im_w + j;
    int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    int res = program_no * data_size * im_h * im_w + pixel_row * data_size + pixel_col;
    return res;
}


__device__ inline
float __pixel_value_in_stack(float *stack, int data_size, int im_h, int im_w, int i, int j) {
    return stack[__pixel_index_in_stack(data_size, im_h, im_w, i, j)];
}


__device__ inline
int __pixel_conv_buffer_index(int data_size, int im_h, int im_w, int i, int j) {
    int program_no = blockIdx.y;
    int pixel_row = i * im_w + j;
    int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    return program_no * data_size * im_h * im_w + pixel_row * data_size + pixel_col;
}


__device__ inline
float __pixel_value_in_conv_buffer(float *buffer, int data_size, int im_h, int im_w, int i, int j) {
    return buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)];
}


__device__ inline
int __std_res_index(int top, int data_size) {
    int program_no = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    return program_no * MAX_STACK_SIZE * data_size + top * data_size + col;
}


__device__ inline
float __std_res_value(float *std_res, int top, int data_size) {
    return std_res[__std_res_index(top, data_size)];
}


// ===========================================================
// ===========================================================

__device__
void _g_std(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *std_res, int top) {
    // image index this thread is response for
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        float avg = 0;
        for (int i = rx; i < rx + rh; i++) {
            for (int j = ry; j < ry + rw; j++) {
                avg += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j);
            }
        }
        avg /= (float) (rh * rw);
        float deviation = 0;
        for (int i = rx; i < rx + rh; i++) {
            for (int j = ry; j < ry + rw; j++) {
                float value = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) - avg;
                deviation += value * value;
            }
        }
        deviation /= (float) (rh * rw);
        deviation = std::sqrt(deviation);
        std_res[__std_res_index(top, data_size)] = deviation;
    }
}


__device__
void _sub(float *std_res, int top, int data_size) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        float res1 = __std_res_value(std_res, top - 2, data_size);
        float res2 = __std_res_value(std_res, top - 1, data_size);
        std_res[__std_res_index(top - 2, data_size)] = res1 - res2;
    }
}


__device__
void _region(float *dataset, int data_size, int im_h, int im_w, float *stack) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = 0; i < im_h; i++) {
            for (int j = 0; j < im_w; j++) {
                float d = __dataset_value(dataset, data_size, im_w, i, j);
                stack[__pixel_index_in_stack(data_size, im_h, im_w, i, j)] = d;
            }
        }
    }
}


__device__ void
_lap(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                float sum = 0;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 4;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j);
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_gau1(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                float sum = 0;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 4;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1);
                sum /= 16;
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_sobel_x(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                float sum = 0;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1);
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1);
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1);
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_sobel_y(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                float sum = 0;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1);
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1);
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_gau11(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    _sobel_x(stack, data_size, im_h, im_w, rx, ry, rh, rw, buffer);
    _sobel_y(stack, data_size, im_h, im_w, rx + 1, ry + 1, rh - 2, rw - 2, buffer);
}


__device__ void
_gauxy(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                float dy = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) -
                           __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j);
                float dx = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) - 
                           __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1);
                float sum = std::sqrt(dx * dx + dy * dy);
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_log(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 2; i < rx + rh - 2; i++) {
            for (int j = ry + 2; j < ry + rw - 2; j++) {
                float sum = 0;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 2);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 2;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 16;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 2);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 2;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1);
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j);
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 2; i < rx + rh - 2; i++) {
            for (int j = ry + 2; j < ry + rw - 2; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_log1(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 2; i < rx + rh - 2; i++) {
            for (int j = ry + 2; j < ry + rw - 2; j++) {
                float sum = 0;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j - 2) * 0.109;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j - 1) * 0.246;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j) * 0.270;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j + 1) * 0.246;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j + 2) * 0.109;
                //
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 2) * 0.246;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 0.606;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 2) * 0.246;
                //
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 2) * 0.270;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 0.606;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 2;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 0.606;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 2) * 0.270;
                //
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 2) * 0.246;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 0.606;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 2) * 0.246;
                //
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j - 2) * 0.109;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j - 1) * 0.246;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j) * 0.270;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j + 1) * 0.246;
                sum += __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j + 2) * 0.109;
                //
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 2; i < rx + rh - 2; i++) {
            for (int j = ry + 2; j < ry + rw - 2; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_log2(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 2; i < rx + rh - 2; i++) {
            for (int j = ry + 2; j < ry + rw - 2; j++) {
                float sum = 0;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j - 1) * 0.1;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j) * 0.151;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 2, j + 1) * 0.1;
                //
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 2) * 0.1;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1) * 0.292;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j) * 0.386;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1) * 0.292;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 2) * 0.1;
                //
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 2) * 0.151;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1) * 0.386;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j) * 0.5;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1) * 0.386;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 2) * 0.151;
                //
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 2) * 0.1;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1) * 0.292;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j) * 0.386;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1) * 0.292;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 2) * 0.1;
                //
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j - 1) * 0.1;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j) * 0.151;
                sum -= __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 2, j + 1) * 0.1;
                //
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 2; i < rx + rh - 2; i++) {
            for (int j = ry + 2; j < ry + rw - 2; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ void
_lbp(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                float sum = 0;
                float cp_value = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j);
                float p1 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j - 1);
                float p2 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j);
                float p4 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i - 1, j + 1);
                float p8 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j + 1);
                float p16 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j + 1);
                float p32 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j);
                float p64 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i + 1, j - 1);
                float p128 = __pixel_value_in_stack(stack, data_size, im_h, im_w, i, j - 1);
                if (p1 >= cp_value) sum += p1;
                if (p2 >= cp_value) sum += p2;
                if (p4 >= cp_value) sum += p4;
                if (p8 >= cp_value) sum += p8;
                if (p16 >= cp_value) sum += p16;
                if (p32 >= cp_value) sum += p32;
                if (p64 >= cp_value) sum += p64;
                if (p128 >= cp_value) sum += p128;
                buffer[__pixel_conv_buffer_index(data_size, im_h, im_w, i, j)] = sum;
            }
        }

        // copy the result from buffer to stack
        for (int i = rx + 1; i < rx + rh - 1; i++) {
            for (int j = ry + 1; j < ry + rw - 1; j++) {
                int stack_index = __pixel_index_in_stack(data_size, im_h, im_w, i, j);
                stack[stack_index] = __pixel_value_in_conv_buffer(buffer, data_size, im_h, im_w, i, j);
            }
        }
    }
}


__device__ inline
int __hist_buffer_index(int data_size, int value) {
    int program_no = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    return program_no * (MAX_PIXEL_VALUE + 1) * data_size + value * data_size + col;
}


__device__ inline
float __hist_buffer_value(float *buffer, int data_size, int value) {
    return buffer[__hist_buffer_index(data_size, value)];
}


__device__
void _hist_eq(float *stack, int data_size, int im_h, int im_w, int rx, int ry, int rh, int rw, float *hist_buffer) {
    int img_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (img_index < data_size) {
        // clear the buffer
        for (int i = 0; i < MAX_PIXEL_VALUE + 1; i++)
            hist_buffer[__hist_buffer_index(data_size, i)] = 0;

        // statistic intensity of each pixel, this process can not perform coalesced access
        for (int i = rx; i < rx + rh; i++) {
            for (int j = ry; j < ry + rw; j++) {
                int pixel_value = (int) (__pixel_value_in_stack(stack, data_size, im_h, im_w, i, j));
                pixel_value = max(0, pixel_value);
                pixel_value = min(MAX_PIXEL_VALUE, pixel_value);
                hist_buffer[__hist_buffer_index(data_size, pixel_value)] += 1;
            }
        }

        // uniform
        float pixel_num = rh * rw;
        for (int i = 0; i < MAX_PIXEL_VALUE + 1; i++)
            hist_buffer[__hist_buffer_index(data_size, i)] /= pixel_num;

        // add up
        for (int i = 1; i < MAX_PIXEL_VALUE + 1; i++)
            hist_buffer[__hist_buffer_index(data_size, i)] +=
                    hist_buffer[__hist_buffer_index(data_size, i - 1)];

        // equalization
        for (int i = 0; i < MAX_PIXEL_VALUE + 1; i++)
            hist_buffer[__hist_buffer_index(data_size, i)] *= (MAX_PIXEL_VALUE - 1);

        // update
        for (int i = rx; i < rx + rh; i++) {
            for (int j = ry; j < ry + rw; j++) {
                int raw = (int) (__pixel_value_in_stack(stack, data_size, im_h, im_w, i, j));
                raw = max(0, raw);
                raw = min(MAX_PIXEL_VALUE, raw);
                float new_value = __hist_buffer_value(hist_buffer, data_size, raw);
                stack[__pixel_index_in_stack(data_size, im_h, im_w, i, j)] = new_value;
            }
        }
    }
}


__global__
void do_reduction(float *std_res, int data_size, int *label) {
    int tidx = threadIdx.x;

    // shared memory
    extern __shared__ int shared[];
    shared[tidx] = 0;
    __syncthreads();

    // compare predict value and label
    int img_index = blockDim.x * blockIdx.x + tidx;
    if (img_index < data_size) {
        float predicted_value = __std_res_value(std_res, 0, data_size);
        int label_value = label[img_index];

        if (predicted_value > 0 && label_value > 0 || predicted_value <= 0 && label_value <= 0) {
            shared[tidx] = 1;
        } else {
            shared[tidx] = 0;
        }
    }
    __syncthreads();

    // reduction
    if (blockDim.x >= 512) {
        if (tidx < 256) { shared[tidx] += shared[tidx + 256]; }
        __syncthreads();
    }

    if (blockDim.x >= 256) {
        if (tidx < 128) { shared[tidx] += shared[tidx + 128]; }
        __syncthreads();
    }

    if (blockDim.x >= 128) {
        if (tidx < 64) { shared[tidx] += shared[tidx + 64]; }
        __syncthreads();
    }

    if (tidx < 32) {
        volatile int *ws_sum = shared;
        if (blockDim.x >= 64) { ws_sum[threadIdx.x] += ws_sum[threadIdx.x + 32]; }
        if (blockDim.x >= 32) { ws_sum[threadIdx.x] += ws_sum[threadIdx.x + 16]; }
        if (blockDim.x >= 16) { ws_sum[threadIdx.x] += ws_sum[threadIdx.x + 8]; }
        if (blockDim.x >= 8) { ws_sum[threadIdx.x] += ws_sum[threadIdx.x + 4]; }
        if (blockDim.x >= 4) { ws_sum[threadIdx.x] += ws_sum[threadIdx.x + 2]; }
        if (blockDim.x >= 2) { ws_sum[threadIdx.x] += ws_sum[threadIdx.x + 1]; }

        if (tidx == 0) {
            int res_index = blockIdx.y * data_size * MAX_STACK_SIZE + blockIdx.x;
            std_res[res_index] = (float) ws_sum[0];
        }
        __syncthreads();
    }
}


__global__
void infer_population(int *name, int *rx, int *ry, int *rh, int *rw, int *plen, int img_h, int img_w, int data_size,
                      float *dataset, float *stack, float *conv_buffer, float *hist_buffer, float *std_res) {

    // the index of program that the current thread is responsible for
    int program_no = blockIdx.y;

    // init the top of the stack, the x, y, h, w of the region
    int top = 0, reg_x = 0, reg_y = 0, reg_h = 0, reg_w = 0;

    // reverse iteration
    int len = plen[program_no];
    for (int i = len - 1; i >= 0; i--) {

        // the offset of the node
        int node_offset = MAX_PROGRAM_LEN * program_no + i;
        int node_name = name[node_offset];

        // do correspond operations with respect to the type of the node
        if (node_name == Region_R) {
            reg_x = rx[node_offset], reg_y = ry[node_offset], reg_h = rh[node_offset], reg_w = rw[node_offset];
            _region(dataset, data_size, img_h, img_w, stack);

        } else if (node_name == Region_S) {
            reg_x = rx[node_offset], reg_y = ry[node_offset], reg_h = rh[node_offset], reg_w = rw[node_offset];
            _region(dataset, data_size, img_h, img_w, stack);

        } else if (node_name == G_Std) {
            _g_std(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, std_res, top);
            top++;

        } else if (node_name == Hist_Eq) {
            _hist_eq(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, hist_buffer);

        } else if (node_name == Gau1) {
            _gau1(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 1, reg_y += 1, reg_h -= 2, reg_w -= 2;

        } else if (node_name == Gau11) {
            _gau11(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 1, reg_y += 1, reg_h -= 2, reg_w -= 2;

        } else if (node_name == GauXY) {
            _gauxy(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 1, reg_y += 1, reg_h -= 2, reg_w -= 2;

        } else if (node_name == Lap) {
            _lap(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 1, reg_y += 1, reg_h -= 2, reg_w -= 2;

        } else if (node_name == Sobel_X) {
            _sobel_x(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 1, reg_y += 1, reg_h -= 2, reg_w -= 2;

        } else if (node_name == Sobel_Y) {
            _sobel_y(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 1, reg_y += 1, reg_h -= 2, reg_w -= 2;

        } else if (node_name == LoG) {
            _log(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 2, reg_y += 2, reg_h -= 4, reg_w -= 4;

        } else if (node_name == LoG1) {
            _log1(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 2, reg_y += 2, reg_h -= 4, reg_w -= 4;

        } else if (node_name == LoG2) {
            _log2(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 2, reg_y += 2, reg_h -= 4, reg_w -= 4;

        } else if (node_name == LBP) {
            _lbp(stack, data_size, img_h, img_w, reg_x, reg_y, reg_h, reg_w, conv_buffer);
            reg_x += 1, reg_y += 1, reg_h -= 2, reg_w -= 2;

        } else if (node_name == Sub) {
            _sub(std_res, top, data_size);
            top--;

        } else {
            printf("Error: Do not support the function, the value of the function is: %d. Try to debug.\n", node_name);
        }
    }

    // threads synchronization
    __syncthreads();
}


class GPUEvaluator {
    typedef vector<Program> Pop;

public:

    GPUEvaluator(vector<vector<float>> dataset, vector<int> label, int img_h, int img_w,
                 int eval_batch = 1, int thread_per_block = 256, bool parallel_reduction = false) {
        this->dataset = std::move(dataset);
        this->label = std::move(label);
        this->img_h = img_h;
        this->img_w = img_w;
        this->data_size = this->dataset.size();
        this->eval_batch = eval_batch;
        this->thread_per_block = thread_per_block;
        this->parallel_reduction = parallel_reduction;

        // dataset transfer
        transfer_dataset();
        transfer_label();

        // allocate device side buffers
        allocate_device_stack();
        allocate_device_conv_buffer();
        allocate_device_hist_buffer();
        allocate_device_res_buffer();

        // memory space for program
        allocate_program_buffer();
    }

    /**
     * Evaluate fitness for programs in the population.
     * @param pop population.
     */
    void evaluate_population(Pop &pop) {
        for (int i = 0; i < pop.size(); i += eval_batch) {
            int last_pos = std::min(i + eval_batch, (int) pop.size());
            fit_eval_for_a_batch(pop, i, last_pos);
        }
    }

    void evaluate_program(Program &program) {
        vector<Program> pop = { program };
        evaluate_population(pop);
    }

    ~GPUEvaluator() {
        cudaFree(this->d_dataset);
        cudaFree(this->d_label);
        cudaFree(this->d_stack);
        cudaFree(this->d_conv_buffer);
        cudaFree(this->d_hist_buffer);
        cudaFree(this->d_std_res);
        cudaFree(this->d_name);
        cudaFree(this->d_rx);
        cudaFree(this->d_ry);
        cudaFree(this->d_rh);
        cudaFree(this->d_rw);
        cudaFree(this->d_plen);
    }

protected:
    vector<vector<float>> dataset;
    vector<int> label;
    int img_h;
    int img_w;
    int data_size;
    int eval_batch;
    int thread_per_block;
    bool parallel_reduction;

    // device side arrays
    float *d_dataset{};
    int *d_label{};
    float *d_stack{};
    float *d_conv_buffer{};
    float *d_hist_buffer{};
    float *d_std_res{};

    // programs
    int *d_name{};
    int *d_rx{};
    int *d_ry{};
    int *d_rh{};
    int *d_rw{};
    int *d_plen{};

protected:

    /**
     * Evaluate programs with the indexes in of range [start_pos, end_pos) in the population.
     */
    void fit_eval_for_a_batch(Pop &pop, int start_pos, int end_pos) const {
        int cur_batch_size = end_pos - start_pos;

        if (cur_batch_size > eval_batch) {
            printf("Error: pop size > eval batch.\n");
        }

        vecI name(MAX_PROGRAM_LEN * cur_batch_size);
        vecI rx(MAX_PROGRAM_LEN * cur_batch_size);
        vecI ry(MAX_PROGRAM_LEN * cur_batch_size);
        vecI rh(MAX_PROGRAM_LEN * cur_batch_size);
        vecI rw(MAX_PROGRAM_LEN * cur_batch_size);
        vecI plen(cur_batch_size);

        for (int i = 0; i < cur_batch_size; i++) {
            auto &program_prefix = pop[i + start_pos].inner_prefix;
            plen[i] = program_prefix.size();
            for (int j = 0; j < program_prefix.size(); j++) {
                name[i * MAX_PROGRAM_LEN + j] = program_prefix[j].name;
                if (program_prefix[j].is_terminal()) {
                    rx[i * MAX_PROGRAM_LEN + j] = program_prefix[j].rx;
                    ry[i * MAX_PROGRAM_LEN + j] = program_prefix[j].ry;
                    rh[i * MAX_PROGRAM_LEN + j] = program_prefix[j].rh;
                    rw[i * MAX_PROGRAM_LEN + j] = program_prefix[j].rw;
                }
            }
        }

        // copy program to device
        auto cpy_size = sizeof(int) * MAX_PROGRAM_LEN * cur_batch_size;
        cudaMemcpy(d_name, thrust::raw_pointer_cast(name.data()), cpy_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rx, thrust::raw_pointer_cast(rx.data()), cpy_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ry, thrust::raw_pointer_cast(ry.data()), cpy_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rh, thrust::raw_pointer_cast(rh.data()), cpy_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rw, thrust::raw_pointer_cast(rw.data()), cpy_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_plen, thrust::raw_pointer_cast(plen.data()), sizeof(int) * cur_batch_size, cudaMemcpyHostToDevice);

        // launch kernel
        int block_dim_x = (int) ((this->data_size - 1 + this->thread_per_block) / this->thread_per_block);
        dim3 grid_dim;
        grid_dim.x = block_dim_x;
        grid_dim.y = cur_batch_size;
        infer_population<<<grid_dim, thread_per_block>>>(d_name, d_rx, d_ry, d_rh, d_rw, d_plen, img_h, img_w,
                                                         data_size, d_dataset, d_stack, d_conv_buffer, d_hist_buffer,
                                                         d_std_res);
        cudaDeviceSynchronize();

        if (parallel_reduction) {
            do_reduction<<<grid_dim, thread_per_block, sizeof(int) * thread_per_block>>>
                    (d_std_res, data_size, d_label);
            cudaDeviceSynchronize();
        }

        // copy result
        auto *h_res = new float[data_size * eval_batch * MAX_STACK_SIZE];
        cudaMemcpy(h_res, d_std_res, sizeof(float) * MAX_STACK_SIZE * data_size * eval_batch, cudaMemcpyDeviceToHost);

        // calculate accuracy
        if (parallel_reduction) {
            for (int i = 0; i < cur_batch_size; i++) {
                int correct = 0;
                for (int j = 0; j < block_dim_x; j++) {
                    correct += (int) h_res[i * data_size * MAX_STACK_SIZE + j];
                }
                pop[start_pos + i].fitness = (float) correct / (float) data_size;
            }
        } else {
            for (int i = 0; i < cur_batch_size; i++) {
                int correct = 0;
                for (int j = 0; j < data_size; j++) {
                    auto predict = h_res[i * data_size * MAX_STACK_SIZE + j];
                    if (predict > 0 && label[j] > 0 || predict < 0 && label[j] < 0) {
                        correct++;
                    }
                }
                pop[start_pos + i].fitness = (float) correct / (float) data_size;
            }
        }

        // delete h_res
        delete[] h_res;
    }

    void transfer_dataset() {
        cudaMalloc((void **) &(this->d_dataset), sizeof(float) * img_h * img_w * data_size);
        vector<float> device_dataset;
        for (int i = 0; i < dataset[0].size(); i++) {
            for (int j = 0; j < dataset.size(); j++) {
                device_dataset.emplace_back(dataset[j][i]);
            }
        }
        cudaMemcpy(this->d_dataset, thrust::raw_pointer_cast(device_dataset.data()),
                   sizeof(float) * img_h * img_w * data_size, cudaMemcpyHostToDevice);
    }

    void transfer_label() {
        cudaMalloc((void **) &(this->d_label), sizeof(int) * data_size);
        cudaMemcpy(this->d_label, thrust::raw_pointer_cast(this->label.data()),
                   sizeof(int) * data_size, cudaMemcpyHostToDevice);
    }

    void allocate_device_stack() {
        cudaMalloc((void **) &(this->d_stack), sizeof(float) * data_size * img_h * img_w * eval_batch);
    }

    void allocate_device_conv_buffer() {
        cudaMalloc((void **) &(this->d_conv_buffer), sizeof(float) * data_size * img_h * img_w * eval_batch);
    }

    void allocate_device_hist_buffer() {
        cudaMalloc((void **) &(this->d_hist_buffer), sizeof(float) * (MAX_PIXEL_VALUE + 1) * eval_batch * data_size);
    }

    void allocate_device_res_buffer() {
        cudaMalloc((void **) &(this->d_std_res), sizeof(float) * MAX_STACK_SIZE * data_size * eval_batch);
    }

    void allocate_program_buffer() {
        cudaMalloc((void **) &(this->d_name), sizeof(int) * eval_batch * MAX_PROGRAM_LEN);
        cudaMalloc((void **) &(this->d_rx), sizeof(int) * eval_batch * MAX_PROGRAM_LEN);
        cudaMalloc((void **) &(this->d_ry), sizeof(int) * eval_batch * MAX_PROGRAM_LEN);
        cudaMalloc((void **) &(this->d_rh), sizeof(int) * eval_batch * MAX_PROGRAM_LEN);
        cudaMalloc((void **) &(this->d_rw), sizeof(int) * eval_batch * MAX_PROGRAM_LEN);
        cudaMalloc((void **) &(this->d_plen), sizeof(int) * eval_batch);
    }
};


#endif //CUDAGPIBC_EVAL_GPU_H
