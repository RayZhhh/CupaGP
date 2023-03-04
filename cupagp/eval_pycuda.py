import time
from typing import List
from .program import Program
import numpy as np
from numba import jit

# PyCUDA will automatically create a new context for current environment
# through importing pycuda.autoprimaryctx
import pycuda.autoprimaryctx

from pycuda.compiler import SourceModule
from pycuda import gpuarray
from pycuda import driver

# the max value of pixels
# this value is for histogram based functions
MAX_PIXEL_VALUE = 255

# max length of programs
# the kernel would crash if a program's length exceeds this value
MAX_PROGRAM_LEN = 400

# the max stack size for evaluating
# the kernel would crash if the evaluation of a program need more stack spaces
# in default cases where the max depth is set to 8, a max top of 10 is enough (actually 8 is enough)
MAX_STACK_SIZE = 20

# device side kernels
# these functions are implemented by CUDA C/C++ layer
# these CUDA C/C++ code will be compiled by PyCUDA before fitness evaluation
_CUDA_CPP_SOURCE_CODE = r"""
enum F {
    Region_S, Region_R, G_Std, Hist_Eq, Gau1, Gau11, GauXY, Lap, Sobel_X, Sobel_Y, LoG1, LoG2, LBP, HOG, Sub
};


// the max value of pixels
// this value is for histogram based functions
#define MAX_PIXEL_VALUE 255

// max length of programs
// the kernel would crash if a program's length exceeds this value
#define MAX_PROGRAM_LEN 400

// the max stack size for evaluating
// the kernel would crash if the evaluation of a program need more stack spaces
// in default cases where the max depth is set to 8, a max top of 10 is enough (actually 8 is enough)
#define MAX_STACK_SIZE 20


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
"""


# each float value is 8 bytes
_SIZE_FLOAT = 8

# each int value is 4 bytes
_SIZE_INT = 4


@jit(nopython=True)
def _cal_accuracy(res, label, data_size: int, program_no: int) -> float:
    """Calculate accuracy for a program in the eval batch.
    Since a 'for' loop is waste of time in python, we separate the code and use @numba.jit() to speed up.
    """
    correct = 0
    for j in range(data_size):
        predict = res[program_no * data_size * MAX_STACK_SIZE + j]
        if label[j] <= 0 and predict <= 0 or label[j] > 0 and predict > 0:
            correct += 1
    return correct / data_size


class PyCudaEvaluator:
    def __init__(self, data, label, eval_batch=1, thread_per_block=128, cuda_arch=None, cuda_code=None):
        """
        Args:
            data            : train-set or test-set
            label           : train-label or test-label
            eval_batch      : the number of programs evaluates simultaneously
            thread_per_block: equals to blockDim.x
            cuda_arch       : CUDA arch, the argument of [nvcc -arch=compute_75 -o ...]
            cuda_code       : CUDA code, the argument of [nvcc -code=sm_75 -o ...]
        """
        self._infer_population_kernel = SourceModule(source=_CUDA_CPP_SOURCE_CODE, arch=cuda_arch, code=cuda_code) \
                .get_function('infer_population')

        self.data = data
        self.label = label
        self.data_size = len(data)
        self.img_h = len(self.data[0])
        self.img_w = len(self.data[0][0])
        self.eval_batch = eval_batch
        self.thread_per_block = thread_per_block

        # device side arrays
        self._d_dataset = ...
        self._d_stack = ...
        self._d_hist_buffer = ...
        self._d_std_res = ...
        self._d_conv_buffer = ...
        self._d_label = ...

        # device side program
        self._d_name = ...
        self._d_rx = ...
        self._d_ry = ...
        self._d_rh = ...
        self._d_rw = ...
        self._d_plen = ...

        # init device side arrays
        self._transfer_dataset()
        self._allocate_device_stack()
        self._allocate_device_conv_buffer()
        self._allocate_device_hist_buffer()
        self._allocate_device_res_buffer()

        # allocate memory space for program
        self._allocate_program_buffer()

        # profiling
        self.cuda_kernel_time = 0

    def load_data_and_label(self, data, label):
        """load new data and label for evaluation"""
        self.data = data
        self.label = label
        self.data_size = len(data)
        self.img_h = len(self.data[0])
        self.img_w = len(self.data[0][0])

        # free device side buffers
        del self._d_dataset
        del self._d_stack
        del self._d_hist_buffer
        del self._d_std_res
        del self._d_conv_buffer

        # init device side buffers
        self._transfer_dataset()
        self._allocate_device_stack()
        self._allocate_device_conv_buffer()
        self._allocate_device_hist_buffer()
        self._allocate_device_res_buffer()

    def _transfer_dataset(self):
        data_ = self.data.reshape(self.data_size, -1).T
        data_ = data_.reshape(-1).astype(np.float32)
        self._d_dataset = gpuarray.to_gpu(data_)

    def _transfer_label(self):
        label_ = self.label.astype(np.int)
        self._d_label = driver.mem_alloc(_SIZE_INT * self.data_size)
        driver.memcpy_htod(self._d_label, label_)

    def _allocate_device_stack(self):
        self._d_stack = driver.mem_alloc(_SIZE_FLOAT * self.data_size * self.img_h * self.img_w * self.eval_batch)

    def _allocate_device_conv_buffer(self):
        self._d_conv_buffer = driver.mem_alloc(_SIZE_FLOAT * self.data_size * self.img_h * self.img_w * self.eval_batch)

    def _allocate_device_hist_buffer(self):
        self._d_hist_buffer = driver.mem_alloc(_SIZE_FLOAT * self.data_size * (MAX_PIXEL_VALUE + 1) * self.eval_batch)

    def _allocate_device_res_buffer(self):
        self._d_std_res = driver.mem_alloc(8 * MAX_STACK_SIZE * self.data_size * self.eval_batch)

    def _allocate_program_buffer(self):
        self._d_name = driver.mem_alloc(_SIZE_INT * self.eval_batch * MAX_PROGRAM_LEN)
        self._d_rx = driver.mem_alloc(_SIZE_INT * self.eval_batch * MAX_PROGRAM_LEN)
        self._d_ry = driver.mem_alloc(_SIZE_INT * self.eval_batch * MAX_PROGRAM_LEN)
        self._d_rh = driver.mem_alloc(_SIZE_INT * self.eval_batch * MAX_PROGRAM_LEN)
        self._d_rw = driver.mem_alloc(_SIZE_INT * self.eval_batch * MAX_PROGRAM_LEN)
        self._d_plen = driver.mem_alloc(_SIZE_INT * self.eval_batch)

    def _infer_population_for_a_batch(self, pop_batch: List[Program]):
        cur_batch_size = len(pop_batch)

        # the size of the current pop to be eval must <= the size of eval_batch
        if cur_batch_size > self.eval_batch:
            raise RuntimeError('Error: pop size > eval batch.')

        # allocate cuda_device side programs
        name = np.zeros((MAX_PROGRAM_LEN * self.eval_batch), np.int32)
        rx = np.zeros((MAX_PROGRAM_LEN * self.eval_batch), np.int32)
        ry = np.zeros((MAX_PROGRAM_LEN * self.eval_batch), np.int32)
        rh = np.zeros((MAX_PROGRAM_LEN * self.eval_batch), np.int32)
        rw = np.zeros((MAX_PROGRAM_LEN * self.eval_batch), np.int32)
        plen = np.zeros(self.eval_batch, np.int32)  # an array stores the length of each program

        # parse the program
        for i in range(cur_batch_size):
            program = pop_batch[i]
            plen[i] = len(program)
            for j in range(len(program)):
                name[i * MAX_PROGRAM_LEN + j] = program[j].name
                if program[j].is_terminal_node():
                    rx[i * MAX_PROGRAM_LEN + j] = program[j].rx
                    ry[i * MAX_PROGRAM_LEN + j] = program[j].ry
                    rh[i * MAX_PROGRAM_LEN + j] = program[j].rh
                    rw[i * MAX_PROGRAM_LEN + j] = program[j].rw

        # copy to cuda_device
        driver.memcpy_htod(self._d_name, name)
        driver.memcpy_htod(self._d_rx, rx)
        driver.memcpy_htod(self._d_ry, ry)
        driver.memcpy_htod(self._d_rh, rh)
        driver.memcpy_htod(self._d_rw, rw)
        driver.memcpy_htod(self._d_plen, plen)

        # launch kernel
        grid = ((self.data_size - 1 + self.thread_per_block) // self.thread_per_block, cur_batch_size)
        block = (self.thread_per_block, 1, 1)

        kernel_start = time.time()
        self._infer_population_kernel(self._d_name, self._d_rx, self._d_ry, self._d_rh, self._d_rw, self._d_plen,
                                      np.int32(self.img_h), np.int32(self.img_w), np.int32(self.data_size),
                                      self._d_dataset, self._d_stack, self._d_conv_buffer, self._d_hist_buffer,
                                      self._d_std_res, grid=grid, block=block)
        driver.Context.synchronize()
        self.cuda_kernel_time += time.time() - kernel_start

    def _fitness_evaluate_for_a_batch(self, pop_batch: List[Program]):
        """Evaluate a population"""
        self._infer_population_for_a_batch(pop_batch)

        # copy result from device to host
        res = np.zeros(MAX_STACK_SIZE * self.eval_batch * self.data_size, np.float32)
        driver.memcpy_dtoh(res, self._d_std_res)

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
