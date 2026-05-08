#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

//矩阵规模 
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 1024
#endif

float* matrix;

// 获取矩阵索引
inline int idx(int r, int c) { return r * MATRIX_SIZE + c; }

// 初始化矩阵
void initialize_matrix() {
    srand(2026);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrix[i] = (float)rand() / RAND_MAX + 1.0f;
    }
}

// 串行算法
void gauss_serial() {
    for (int k = 0; k < MATRIX_SIZE; k++) {
        float pivot = matrix[idx(k, k)];
        for (int j = k + 1; j < MATRIX_SIZE; j++) {
            matrix[idx(k, j)] /= pivot;
        }
        matrix[idx(k, k)] = 1.0f;

        for (int i = k + 1; i < MATRIX_SIZE; i++) {
            for (int j = k + 1; j < MATRIX_SIZE; j++) {
                matrix[idx(i, j)] -= matrix[idx(i, k)] * matrix[idx(k, j)];
            }
            matrix[idx(i, k)] = 0.0f;
        }
    }
}

// 部分向量化 ( 8-13 行消去部分使用 NEON)
void gauss_simd_partial() {
    for (int k = 0; k < MATRIX_SIZE; k++) {
        float pivot = matrix[idx(k, k)];
        for (int j = k + 1; j < MATRIX_SIZE; j++) {
            matrix[idx(k, j)] /= pivot;
        }
        matrix[idx(k, k)] = 1.0f;

        for (int i = k + 1; i < MATRIX_SIZE; i++) {
            float32x4_t v_elim = vdupq_n_f32(matrix[idx(i, k)]);
            int j = k + 1;
            for (; j <= MATRIX_SIZE - 4; j += 4) {
                float32x4_t v_pivot = vld1q_f32(&matrix[idx(k, j)]);
                float32x4_t v_curr = vld1q_f32(&matrix[idx(i, j)]);
                vst1q_f32(&matrix[idx(i, j)], vmlsq_f32(v_curr, v_elim, v_pivot));
            }
            for (; j < MATRIX_SIZE; j++) {
                matrix[idx(i, j)] -= matrix[idx(i, k)] * matrix[idx(k, j)];
            }
            matrix[idx(i, k)] = 0.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./main [serial|simd]" << std::endl;
        return 1;
    }

    // 申请 64 字节对齐的内存
    if (posix_memalign((void**)&matrix, 64, MATRIX_SIZE * MATRIX_SIZE * sizeof(float)) != 0) {
        return 1;
    }

    initialize_matrix();
    std::string mode = argv[1];
    
    auto start = std::chrono::steady_clock::now();

    if (mode == "serial") gauss_serial();
    else if (mode == "simd") gauss_simd_partial();

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Mode: " << mode << " | Size: " << MATRIX_SIZE 
              << " | Time: " << duration << " us" << std::endl;

    free(matrix);
    return 0;
}
