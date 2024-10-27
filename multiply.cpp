#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h> // Для SIMD команд
#include <cstdlib>     // Для функции rand()

int main() {
    const double maxDuration = 10.0; // Максимальное время выполнения в секундах

    // --- Eigen умножение ---
    {
        int N = 64;

        while (true) {
            // Создаем две случайные матрицы размером N x N
            Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
            Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);

            // Измеряем время умножения
            auto start = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXd C = A * B;
            auto end = std::chrono::high_resolution_clock::now();

            // Вычисляем прошедшее время в секундах
            std::chrono::duration<double> duration = end - start;
            std::cout << "Eigen multiplication for size " << N << " took " << duration.count() << " seconds" << std::endl;

            if (duration.count() > maxDuration) {
                break;
            }

            N += 64; // Увеличиваем размер матрицы
        }
    }

    // --- Ручное умножение с блокировкой, OpenMP и SIMD ---
    {
        const int blockSize = 32;
        int N = 64;

        while (true) {
            // Инициализация матриц
            double* A = new double[N * N];
            double* B = new double[N * N];
            double* C = new double[N * N]();

            // Заполнение матриц случайными значениями
            for (int i = 0; i < N * N; ++i) {
                A[i] = static_cast<double>(rand()) / RAND_MAX;
                B[i] = static_cast<double>(rand()) / RAND_MAX;
            }

            // Измеряем время умножения
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i += blockSize) {
                for (int j = 0; j < N; j += blockSize) {
                    for (int k = 0; k < N; k += blockSize) {
                        for (int ii = 0; ii < blockSize; ++ii) {
                            for (int jj = 0; jj < blockSize; ++jj) {
                                __m256d c = _mm256_loadu_pd(&C[(i + ii) * N + (j + jj)]);
                                for (int kk = 0; kk < blockSize; kk += 4) {
                                    __m256d a = _mm256_loadu_pd(&A[(i + ii) * N + (k + kk)]);
                                    __m256d b = _mm256_loadu_pd(&B[(k + kk) * N + (j + jj)]);
                                    c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
                                }
                                _mm256_storeu_pd(&C[(i + ii) * N + (j + jj)], c);
                            }
                        }
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();

            // Вычисляем прошедшее время в секундах
            std::chrono::duration<double> duration = end - start;
            std::cout << "Manual block multiplication with OpenMP and SIMD for size " << N << " took " << duration.count() << " seconds" << std::endl;

            // Освобождаем память
            delete[] A;
            delete[] B;
            delete[] C;

            if (duration.count() > maxDuration) {
                break;
            }

            N += 64; // Увеличиваем размер матрицы
        }
    }

    return 0;
}