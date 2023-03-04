//
// Created by Derek on 2022/11/14.
//

#ifndef CUDAGPIBC_RAND_ENGINE_H
#define CUDAGPIBC_RAND_ENGINE_H

#include <random>

static int randint_(int loBound, int upBound) {
    int bound_width = upBound - loBound + 1;
    return rand() % bound_width + loBound;
}

static float randfloat_(float loBound, float upBound) {
    float rd = loBound + (float) (rand()) / (float) (RAND_MAX / (upBound - loBound));
    return rd;
}


static float random_() {
    return randfloat_(0, 1);
}


#endif //CUDAGPIBC_RAND_ENGINE_H
