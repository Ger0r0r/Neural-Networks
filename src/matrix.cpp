#include "matrix.hpp"

matrix::matrix(int ll, int cc, double init_value = 0.5) : rows(ll), cols(cc), data(new double[ll*cc]()) {
	std::cout << "init matrix " << data << " size " << ll*cc << std::endl;
	std::fill(data, data + rows * cols, init_value);

	// for (int i = 0; i < ll; i++) {
	// 	for (int j = 0; j < cc; j++) {
	// 		data[i * cc + j] = (j + 1) * 0.1;
	// 	}
	// }

	// for (int i = 0; i < ll; i++) {
	// 	for (int j = 0; j < cc; j++) {
	// 		data[i * cc + j] = (i + 1) * 0.1;
	// 	}
	// }

	// print();
	std::cout << "end of init" << std::endl;
};
matrix::~matrix() {
	delete data;
};

double& matrix::operator()(int row, int col) {
	return data[row * cols + col];
}
int matrix::getRows() {
	return rows;
}
int matrix::getCols() {
	return cols;
}

matrix::matrix(matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
	std::cout << "COPY!!!" << std::endl;
	other.data = nullptr; // Забираем данные, обнуляем указатель у исходного объекта
}

void matrix::print() {
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			std::cout << data[r * cols + c] << " ";
		}
		std::cout << std::endl;
	}
}

// void matrix_multiply(const double* A, const double* B, double* C, int n, int m, int k) {
// 	std::cout << "MM " << A << " " << B << " " << C << " " << n << " " << m << " " << k << "\n";
// 	#pragma omp parallel for
// 	for (int i = 0; i < n; i += BLOCK_SIZE) {
// 		std::cout << "Iter i " << i << "\n";
// 		for (int j = 0; j < m; j += BLOCK_SIZE) {
// 			std::cout << "Iter j " << j << "\n";
// 			for (int p = 0; p < k; p += BLOCK_SIZE) {
// 				std::cout << "Iter p " << p << "\n";
// 				// Блоки умножаются блоками
// 				for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ++ii) {
// 					std::cout << "Iter ii " << ii << "\n";
// 					for (int kk = p; kk < p + BLOCK_SIZE && kk < k; ++kk) {
// 						std::cout << "Iter kk " << kk << "\n";
// 						double sum = 0.0;
// 						int jj = j;
// 						// SIMD обработка блоков кратных 4
// 						for (; jj <= m - 4 && jj < j + BLOCK_SIZE; jj += 4) {
// 							std::cout << "Iter 1 jj " << jj << "\n";
// 							__m256d a = _mm256_loadu_pd(&A[ii * m + jj]);  // Загружаем 4 элемента из A
// 							__m256d b = _mm256_loadu_pd(&B[kk * m + jj]);
// 							// __m256d b = _mm256_set_pd(B[(kk + 3) * m + jj], B[(kk + 2) * m + jj],
// 							//						  B[(kk + 1) * m + jj], B[kk * m + jj]);  // Загружаем элементы B вручную
// 							__m256d prod = _mm256_mul_pd(a, b);  // Умножаем
// 							__m256d hsum = _mm256_hadd_pd(prod, prod);  // Суммируем попарно
// 							sum += ((double*)&hsum)[0] + ((double*)&hsum)[2];  // Собираем сумму элементов
// 						}

// 						// Обработка оставшихся элементов, не кратных 4
// 						for (; jj < j + BLOCK_SIZE && jj < m; ++jj) {
// 							std::cout << "Iter 2 jj " << jj << "\n";
// 							sum += A[ii * m + jj] * B[kk * m + jj];
// 						}

// 						// Суммируем результат
// 						C[ii * k + kk] += sum;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// std::cout << "MM done\n";
// }

void matrix_multiply_v_m(const double* V, const double* M, double* R, int n, int m) {
    double* MT = new double[m * n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            MT[j * n + i] = M[i * m + j];
        }
    }
    matrix_multiply_m_v(MT, V, R, m, n);
    delete[] MT;
}

void matrix_multiply_m_v(const double* M, const double* V, double* R, int n, int m) {
	// Result have size n
	// std::cout << "MM " << V << " " << M << " " << R << " " << n << " " << m << "\n";
	#pragma omp parallel for
	for (int i = 0; i < n; i += BLOCK_SIZE) {
		// std::cout << "Iter i " << i << std::endl;
		for (int j = 0; j < m; j += BLOCK_SIZE) {
			// std::cout << "Iter j " << j << std::endl;
			// Блоки умножаются блоками
			for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ++ii) {
				// std::cout << "Iter ii " << ii << std::endl;
				double sum = 0.0;
				int jj = j;
				// SIMD обработка блоков кратных 4
				for (; jj <= m - 4 && jj < j + BLOCK_SIZE; jj += 4) {
					// std::cout << "Iter 1 jj " << jj << std::endl;
					__m256d a = _mm256_loadu_pd(&V[jj]);  // Загружаем 4 элемента из V
					__m256d b = _mm256_loadu_pd(&M[ii * m + jj]);
					__m256d prod = _mm256_mul_pd(a, b);  // Умножаем
					__m256d hsum = _mm256_hadd_pd(prod, prod);  // Суммируем попарно
					sum += ((double*)&hsum)[0] + ((double*)&hsum)[2];  // Собираем сумму элементов
				}

				// Обработка оставшихся элементов, не кратных 4
				for (; jj < j + BLOCK_SIZE && jj < m; ++jj) {
					// std::cout << "Iter 2 jj " << jj << std::endl;
					sum += V[jj] * M[ii * m + jj];
				}
				// Суммируем результат
				R[ii] += sum;
			}
		}
	}
	// std::cout << "MM done\n";
}