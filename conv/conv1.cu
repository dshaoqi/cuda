/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>
#include<iostream>

// CUDA runtime
#include <cuda_runtime.h>
//#include<device_launch_parameters.h>
// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

typedef struct {
    int width;
    int heigth;
    int size;
    int *pixel;
}Matrix;

__global__ void Conv(const Matrix input,const Matrix core,const Matrix res) { //_ 每个线程计算一个卷积结果 
    size_t x = threadIdx.x;
    size_t y = threadIdx.y;
    size_t i, j, tmp=0;
    for (j = 0; j < core.heigth; j++) {
        for (i = 0; i < core.width; i++)
            tmp += input.pixel[(y + j) * input.width + x + i] * core.pixel[core.width * j + i]; 
    }
    res.pixel[y * res.width + x] = tmp;
}
//_ 由于所有的线程都使用相同的core,这里可以使用共享内存加载core
//_ 由于卷积结果的相邻点的计算使用了图片的共同像素，应该也可以通过共享内存优化，而且core越大，优化效果越明显，
//_ 是否可以考虑直接把input矩阵加载到共享内存，考虑图片大小和共享内存大小



//_ create host matrix
Matrix createHMatrix(int width, int heigth) {
    Matrix tmp;
    tmp.width = width;
    tmp.heigth = heigth;
    tmp.size = width * heigth * sizeof(int);
    tmp.pixel = (int*)malloc(tmp.size);
    return tmp;
}

//_ create device matrix
Matrix createDMatrix(int width, int heigth) {
    Matrix tmp;
    tmp.width = width;
    tmp.heigth = heigth;
    tmp.size = width * heigth * sizeof(int);
    cudaMalloc(&tmp.pixel,tmp.size);
    return tmp;
}

#define IW 5
#define IH 5
#define CW 2
#define CH 2

void initInput(const Matrix &in) {
    for (int i = 0; i < in.heigth; i++) {
        for (int j = 0; j < in.width; j++) {
            in.pixel[i * in.width + j] = i * j;
        }
    }
}

void initCore(const Matrix& core) {
    for (int i = 0; i < core.heigth; i++) {
        for (int j = 0; j < core.width; j++) {
            core.pixel[i * core.width + j] = i + j;
        }
    }
}

void printMatrix(const Matrix &m) {
    std::cout << "w=" << m.width << "\t" << "h=" << m.heigth << std::endl;
    for (int i = 0; i < m.heigth; i++) {
        for (int j = 0; j < m.width; j++) {
            std::cout << m.pixel[i*m.width+j]<<'\t';
        }
        std::cout << std::endl;
    }
}

int main() {
    Matrix inpic, core, outpic;
    Matrix d_inpic, d_core, d_outpic;
    inpic = createHMatrix(IW, IH);
    core = createHMatrix(CW, CH);
    outpic = createHMatrix(IW - CW + 1, IH - CH + 1);
    initInput(inpic);
    initCore(core);

    d_inpic = createDMatrix(IW, IH);
    d_core = createDMatrix(CW, CH);
    d_outpic = createDMatrix(IW - CW + 1, IH - CH + 1);
    cudaMemcpy(d_inpic.pixel, inpic.pixel, inpic.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_core.pixel, core.pixel, core.size, cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(IW - CW + 1, IH - CH + 1);
    Conv << <grid,block >> > (d_inpic, d_core, d_outpic); //grid为1，block的形状和输出矩阵的形状相同
    cudaMemcpy(outpic.pixel, d_outpic.pixel, d_outpic.size, cudaMemcpyDeviceToHost);

    printMatrix(outpic);
    cudaFree(d_inpic.pixel);
    cudaFree(d_core.pixel);
    cudaFree(d_outpic.pixel);
}
