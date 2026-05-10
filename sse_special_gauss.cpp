#include <iostream>
#include <chrono>
#include <immintrin.h> 
#include <stdlib.h>
#include <string.h>

using namespace std;
using namespace std::chrono;

// 矩阵参数
#define N 1024
#define BYTE_PER_ROW (N / 8)
// 内存对齐步长128位
#define STRIDE ((BYTE_PER_ROW + 63) / 64 * 64)

unsigned char* matrix;
unsigned char* eliminators;

void init_data() {
    for (int i = 0; i < N * STRIDE; i++) {
        matrix[i] = rand() % 256;
        eliminators[i] = rand() % 256;
    }
}

// 使用SSE指令进行特殊高斯消元
void special_gauss_sse() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // 判定逻辑：检查 matrix[i] 的第 j 位是否为 1
            if (matrix[i * STRIDE + (j / 8)] & (1 << (j % 8))) {
                
                unsigned char* m_row = &matrix[i * STRIDE];
                unsigned char* e_row = &eliminators[j * STRIDE];

                int k = 0;
                // 每次处理 128 位 (16 字节)
                for (; k <= BYTE_PER_ROW - 16; k += 16) {
                    // 加载 128 位数据
                    __m128i v_m = _mm_loadu_si128((__m128i*)(m_row + k));
                    __m128i v_e = _mm_loadu_si128((__m128i*)(e_row + k));
                    
                    // 执行按位异或
                    __m128i v_res = _mm_xor_si128(v_m, v_e);
                    
                    // 存储结果回内存
                    _mm_storeu_si128((__m128i*)(m_row + k), v_res);
                }

                // 处理不足 16 字节的剩余部分
                for (; k < BYTE_PER_ROW; k++) {
                    m_row[k] ^= e_row[k];
                }
            }
        }
    }
}

int main() {
    // 在 X86 平台上申请对齐内存
    // SSE 指令要求 16 字节对齐，申请 64 字节对齐以优化缓存行访问
    matrix = (unsigned char*)_mm_malloc(N * STRIDE, 64);
    eliminators = (unsigned char*)_mm_malloc(N * STRIDE, 64);

    if (!matrix || !eliminators) return 1;

    init_data();
    cout << "X86 SSE Special Gaussian (XOR Mode) | Matrix: " << N << "x" << N << endl;

    auto start = high_resolution_clock::now();
    special_gauss_sse();
    auto end = high_resolution_clock::now();

    cout << "SSE Execution Time: " << duration_cast<microseconds>(end - start).count() << " us" << endl;

    _mm_free(matrix);
    _mm_free(eliminators);
    return 0;
}
