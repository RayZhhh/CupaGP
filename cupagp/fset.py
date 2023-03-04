# This file defines labels of various GP functions.

from typing import List

# input type : img, int1, int2
# output type: Region
# description: detect a square region from a large input image
Region_S = 0

# input type : img, int1, int2, int3, int4
# output type: Region
# description: detect a rectangle region from a large input image
Region_R = 1

# input type : Region
# output type: double
# description: extract standard deviation of a region
G_Std = 2

# input type : Region
# output type: Region
# description: perform histogram equalisation to the input region
Hist_Eq = 3

# input type : Region
# output type: Region
# description: perform Gaussian smooth filtering with σ=1 to the input region
#              a 3x3 Gaussian smooth filter is defined by:
#              [1, 2, 1]
#              [2, 4, 2] * (1 / 16)
#              [1, 2, 1]
Gau1 = 4

# input type : Region
# output type: Region
# description: calculate the first derivatives of Gaussian filter with σ=1 of a region
Gau11 = 5

# input type : Region
# output type: Region
# description: calculate the gradient magnitude using Gaussian derivatives with σ=1 to the input region
GauXY = 6

# input type : Region
# output type: Region
# description: perform Laplacian filtering to the input region
#              the 3x3 Laplacian filter is defined by:
#              [0, 1, 0]
#              [1,-4, 1]
#              [0, 1, 0]
Lap = 7

# input type : Region
# output type: Region
# description: perform Sobel filtering along X axis to the input region
#              the Sobel filter is defined by:
#              [ 1, 2, 1 ]               [ -1, 0, 1 ]
#              [ 0, 0, 0 ] (horizontal), [ -2, 0, 2 ] (vertical).
#              [-1,-2,-1 ]               [ -1, 0, 1 ]
Sobel_X = 8

# input type : Region
# output type: Region
# description: perform Sobel filtering along Y axis to the input region
#              the Sobel filter is defined by:
#              [ 1, 2, 1 ]               [ -1, 0, 1 ]
#              [ 0, 0, 0 ] (horizontal), [ -2, 0, 2 ] (vertical).
#              [-1,-2,-1 ]               [ -1, 0, 1 ]
Sobel_Y = 9

# input type :
# output type:
# description:
LoG1 = 10

# input type :
# output type:
# description:
LoG2 = 11

# input type :
# output type:
# description:
LBP = 12

# input type :
# output type:
# description:
HOG = 13

# input type : double
# output type: double
# description: the standard subtraction (-)
Sub = 14


def func_to_str(func: int) -> str:
    l: List[str] = ['Region_S', 'Region_R', 'G_Std', 'Hist_Eq', 'Gau1', 'Gau11', 'GauXY',
                    'Lap', 'Sobel_X', 'Sobel_Y', 'LoG1', 'LoG2', 'LBP', 'HOG', 'Sub']
    return l[func]
