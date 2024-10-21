#include <random>
#include <iostream>
#include <vector>
#include <immintrin.h>  // Для работы с SIMD
#include <omp.h>        // Для многопоточности
#include <algorithm>
#include <cassert>
#include <utility>
#include <numeric>
#include <chrono>
#include <Eigen/Dense> // Подключаем основной заголовочный файл Eigen
#include <ctime>
#include <random>
#include <sys/sysinfo.h> // Для получения информации о системе
#include <unistd.h>      // Для работы с системными вызовами

double getCpuTime() {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9; // Переводим в секунды
}

#define BLOCK_SIZE 32  // Размер блока для оптимизации кэша


void matrix_multiply(const double* M, const double* V, double* R, int n) {
	double* MT = new double[n * n];

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			MT[j * n + i] = M[i * n + j];
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < n; i += BLOCK_SIZE) {
		// std::cout << "Iter i " << i << std::endl;
		for (int j = 0; j < n; j += BLOCK_SIZE) {
			// std::cout << "Iter j " << j << std::endl;
			// Блоки умножаются блоками
			for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ++ii) {
				// std::cout << "Iter ii " << ii << std::endl;
				double sum = 0.0;
				int jj = j;
				// SIMD обработка блоков кратных 4
				for (; jj <= n - 4 && jj < j + BLOCK_SIZE; jj += 4) {
					// std::cout << "Iter 1 jj " << jj << std::endl;
					__m256d a = _mm256_loadu_pd(&V[ii * n + jj]);  // Загружаем 4 элемента из V
					__m256d b = _mm256_loadu_pd(&MT[ii * n + jj]);
					__m256d prod = _mm256_mul_pd(a, b);  // Умножаем
					__m256d hsum = _mm256_hadd_pd(prod, prod);  // Суммируем попарно
					sum += ((double*)&hsum)[0] + ((double*)&hsum)[2];  // Собираем сумму элементов
				}
				// Суммируем результат
				R[ii] += sum;
			}
		}
	}
	delete [] MT;
	// std::cout << "MM done\n";
}

int main () {


	srand(static_cast<unsigned int>(time(0)));
	std::random_device rd;  // Получаем случайное начальное значение
    std::mt19937 gen(rd()); // Инициализируем генератор
    std::uniform_real_distribution<> dis(0.0, 10.0); // Устанавливаем диапазон

    struct sysinfo sysInfo;
    sysinfo(&sysInfo);
    unsigned int numCores = sysInfo.procs; // Количество логических ядер
    std::cout << "Количество логических ядер: " << numCores << std::endl;

	int i = 0;
	bool flag = 1;
	while (flag) {
		i++;
		const int size = i * 64;
		Eigen::MatrixXf mat1(size, size);
		Eigen::MatrixXf mat2(size, size);
		mat1 = Eigen::MatrixXf::Random(size, size); // Диапазон [0, 10]
		mat1 = (mat1 + Eigen::MatrixXf::Constant(size, size, 1)) * 5;
        mat2 = Eigen::MatrixXf::Random(size, size);
		mat2 = (mat2 + Eigen::MatrixXf::Constant(size, size, 1)) * 5;
		auto startRealTime = std::chrono::high_resolution_clock::now();

		Eigen::MatrixXf result = mat1 * mat2;

		auto endRealTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> realTime = endRealTime - startRealTime;
		double time_result = realTime.count();
		std::cout << "EIGEN: " << size << " " << time_result << std::endl;

		if (time_result > 1) break;

	}


	i = 0;
	flag = 1;
	while (flag) {
		i++;
		int size = i * 64;
		double* arr_1 = new double[size*size];
		double* arr_2 = new double[size*size];
		double* arr_r = new double[size*size];
		for (int i = 0; i < size; ++i) {
			arr_1[i] = dis(gen);
			arr_2[i] = dis(gen);
		}
		auto startRealTime = std::chrono::high_resolution_clock::now();

		matrix_multiply(arr_1,arr_2,arr_r,size);

		auto endRealTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> realTime = endRealTime - startRealTime;
		double time_result = realTime.count();
		std::cout << "SIMD: " << size << " " << time_result << std::endl;

		delete [] arr_1;
		delete [] arr_2;
		delete [] arr_r;

		if (time_result > 1) break;

	}

	return 0;
}