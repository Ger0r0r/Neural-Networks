#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h> // Для SIMD команд
#include <cstdlib>     // Для функции rand()
#include <fstream>     // Для работы с файлами

int main() {
    const double maxDuration = 10.0; // Максимальное время выполнения в секундах

    // --- Eigen умножение ---
    {
        std::ofstream eigenFile("data/eigen_times.txt", std::ios::app); // Файл для записи времен Eigen


		int N = 64; // Начинаем с 64
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
			eigenFile << N << "," << duration.count() << std::endl;; // Записываем только время

			// Выводим время в терминал
			std::cout << "Eigen multiplication for size " << N << " took " << duration.count() << " seconds" << std::endl;

			if (duration.count() > maxDuration) {
				break;
			}

			N += 64; // Увеличиваем размер матрицы на 64
		}

        eigenFile.close(); // Закрываем файл
    }

    // --- Ручное умножение с блокировкой, OpenMP и SIMD ---
    {
        const int blockSize = 32;
        std::ofstream manualFile("data/manual_times.txt", std::ios::app);

		int N = 64;

		while (true) {
			// Инициализация матриц с выравниванием
			double* A = static_cast<double*>(_aligned_malloc(N * N * sizeof(double), 32));
			double* B = static_cast<double*>(_aligned_malloc(N * N * sizeof(double), 32));
			double* C = static_cast<double*>(_aligned_malloc(N * N * sizeof(double), 32));

			// Заполнение матриц случайными значениями
			for (int i = 0; i < N * N; ++i) {
				A[i] = static_cast<double>(rand()) / RAND_MAX;
				B[i] = static_cast<double>(rand()) / RAND_MAX;
				C[i] = 0.0; // Инициализируем C нулями
			}

			// Измеряем время умножения
			auto start = std::chrono::high_resolution_clock::now();

			#pragma omp parallel for collapse(2)
			for (int i = 0; i < N; i += blockSize) {
				for (int j = 0; j < N; j += blockSize) {
					for (int k = 0; k < N; k += blockSize) {
						for (int ii = 0; ii < blockSize; ++ii) {
							for (int jj = 0; jj < blockSize; jj += 4) {
								__m256d c = _mm256_load_pd(&C[(i + ii) * N + (j + jj)]);
								for (int kk = 0; kk < blockSize; kk += 4) {
									__m256d a = _mm256_load_pd(&A[(i + ii) * N + (k + kk)]);
									__m256d b = _mm256_load_pd(&B[(j + jj) * N + (k + kk)]);
									c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
								}
								_mm256_store_pd(&C[(i + ii) * N + (j + jj)], c);
							}
						}
					}
				}
			}

			auto end = std::chrono::high_resolution_clock::now();

			// Вычисляем прошедшее время в секундах
			std::chrono::duration<double> duration = end - start;
			manualFile << N << "," << duration.count() << std::endl;; // Записываем только время

			// Выводим время в терминал
			std::cout << "Manual block multiplication for size " << N << " took " << duration.count() << " seconds" << std::endl;

			// Освобождаем память
			_aligned_free(A);
			_aligned_free(B);
			_aligned_free(C);

			if (duration.count() > maxDuration) {
				break;
			}

			N += 64; // Увеличиваем размер матрицы на 64
		}
        manualFile.close(); // Закрываем файл
    }

	// --- Простейшее умножение ---
    {
        std::ofstream simpleFile("data/simple_times.txt", std::ios::app);

		int N = 64; // Начинаем с 64

		while (true) {
			// Инициализация матриц
			double* A = new double[N * N];
			double* B = new double[N * N];
			double* C = new double[N * N];

			// Заполнение матриц случайными значениями
			for (int i = 0; i < N * N; ++i) {
				A[i] = static_cast<double>(rand()) / RAND_MAX;
				B[i] = static_cast<double>(rand()) / RAND_MAX;
				C[i] = 0.0; // Инициализируем C нулями
			}

			// Измеряем время умножения
			auto start = std::chrono::high_resolution_clock::now();

			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < N; ++j) {
					for (int k = 0; k < N; ++k) {
						C[i * N + j] += A[i * N + k] * B[k * N + j];
					}
				}
			}

			auto end = std::chrono::high_resolution_clock::now();

			// Вычисляем прошедшее время в секундах
			std::chrono::duration<double> duration = end - start;
			simpleFile << N << "," << duration.count() << std::endl;; // Записываем только время

			// Выводим время в терминал
			std::cout << "Simple multiplication for size " << N << " took " << duration.count() << " seconds" << std::endl;

			// Освобождаем память
			delete[] A;
			delete[] B;
			delete[] C;

			if (duration.count() > maxDuration) {
				break;
			}

			N += 64; // Увеличиваем размер матрицы на 64
		}
        simpleFile.close(); // Закрываем файл
    }

    return 0;
}
