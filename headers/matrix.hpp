#include <iostream>
#include <vector>
#include <immintrin.h>  // Для работы с SIMD
#include <omp.h>        // Для многопоточности

#define BLOCK_SIZE 32  // Размер блока для оптимизации кэша

class matrix {
	private:
		int rows;
		int cols;
	public:
		matrix(int ll, int cc, double init_value);
		~matrix();

    	matrix(matrix&& other) noexcept;

		double& operator()(int row, int col);
		int getRows();
		int getCols();
		void print();

		double * data;
};

void matrix_multiply(const double* A, const double* B, double* C, int n, int m, int k);