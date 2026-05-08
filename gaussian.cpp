#include <iostream>
#include <chrono>
#include <immintrin.h> // SIMD AVX2 头文件
#include <stdlib.h>

using namespace std;
using namespace std::chrono;

// 矩阵规模 N，建议设为 8 的倍数，方便 AVX2 处理
#define N 1024
float A[N][N];
float B[N][N];

void init_matrix() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = B[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

// 串行算法
void serial_gauss() {
    for (int k = 0; k < N; k++) {
        float pivot = A[k][k];
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++) {
            float temp = A[i][k];
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= temp * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

// AVX2 优化算法
void simd_gauss() {
    for (int k = 0; k < N; k++) {
        float pivot = B[k][k];
        for (int j = k + 1; j < N; j++) {
            B[k][j] /= pivot;
        }
        B[k][k] = 1.0f;

        for (int i = k + 1; i < N; i++) {
            // 将 Factor 广播到 256 位寄存器的 8 个槽位
            __m256 v_ik = _mm256_set1_ps(B[i][k]);
            
            int j = k + 1;
            // 向量化循环：一次处理 8 个 float
            for (; j <= N - 8; j += 8) {
                __m256 v_kj = _mm256_loadu_ps(&B[k][j]);
                __m256 v_ij = _mm256_loadu_ps(&B[i][j]);
                // v_ij = v_ij - (v_ik * v_kj)
                __m256 v_res = _mm256_sub_ps(v_ij, _mm256_mul_ps(v_ik, v_kj));
                _mm256_storeu_ps(&B[i][j], v_res);
            }
            // 处理不足 8 个的尾部
            for (; j < N; j++) {
                B[i][j] -= B[i][k] * B[k][j];
            }
            B[i][k] = 0;
        }
    }
}

// 必须包含此标准的 main 函数入口
int main() {
    init_matrix();
    cout << "Matrix Size: " << N << "x" << N << endl;

    auto start = high_resolution_clock::now();
    serial_gauss();
    auto end = high_resolution_clock::now();
    cout << "Serial Time: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    start = high_resolution_clock::now();
    simd_gauss();
    end = high_resolution_clock::now();
    cout << "AVX2 SIMD Time: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    return 0;
}