#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
using namespace std::chrono;

// 矩阵参数
#define N 1024
// 每一行实际含有的字节数 
#define BYTE_PER_ROW (N / 8)
// 内存对齐步长：填充到 64 字节的倍数
#define STRIDE ((BYTE_PER_ROW + 63) / 64 * 64)

unsigned char* matrix;
unsigned char* eliminators;

void init_data() {
    // 模拟数据初始化
    for (int i = 0; i < N * STRIDE; i++) {
        matrix[i] = rand() % 256;
        eliminators[i] = rand() % 256;
    }
}

// 串行算法
void special_gauss_serial() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // 如果对应的位为1则进行消元
            if (matrix[i * STRIDE] & (1 << (j % 8))) {
                for (int k = 0; k < BYTE_PER_ROW; k++) {
                    matrix[i * STRIDE + k] ^= eliminators[j * STRIDE + k];
                }
            }
        }
    }
}

//NEON 向量化算法
void special_gauss_neon() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (matrix[i * STRIDE] & (1 << (j % 8))) {
                int k = 0;
                unsigned char* m_row = &matrix[i * STRIDE];
                unsigned char* e_row = &eliminators[j * STRIDE];

                // 一次处理 128 位 (
                for (; k <= BYTE_PER_ROW - 16; k += 16) {
                    uint8x16_t vm = vld1q_u8(m_row + k);
                    uint8x16_t ve = vld1q_u8(e_row + k);
                    uint8x16_t res = veorq_u8(vm, ve);
                    vst1q_u8(m_row + k, res);
                }

                // 处理剩余字节
                for (; k < BYTE_PER_ROW; k++) {
                    m_row[k] ^= e_row[k];
                }
            }
        }
    }
}

int main() {
    // 内存对齐申请
    if (posix_memalign((void**)&matrix, 64, N * STRIDE) != 0) return 1;
    if (posix_memalign((void**)&eliminators, 64, N * STRIDE) != 0) return 1;

    init_data();
    cout << "Special Gaussian (XOR Mode) | Matrix: " << N << "x" << N << endl;

    auto s1 = high_resolution_clock::now();
    special_gauss_serial();
    auto e1 = high_resolution_clock::now();
    cout << "Serial Time: " << duration_cast<microseconds>(e1 - s1).count() << " us" << endl;

    init_data(); // 重置数据

    auto s2 = high_resolution_clock::now();
    special_gauss_neon();
    auto e2 = high_resolution_clock::now();
    cout << "NEON Time:   " << duration_cast<microseconds>(e2 - s2).count() << " us" << endl;

    free(matrix);
    free(eliminators);
    return 0;
}
