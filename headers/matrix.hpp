#include <iostream>
#include <vector>
#include <immintrin.h>  // Для работы с SIMD
#include <omp.h>        // Для многопоточности

#define BLOCK_SIZE 8  // Размер блока для оптимизации кэша

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

// void matrix_multiply(const double* A, const double* B, double* C, int n, int m, int k);
void matrix_multiply_v_m(const double* V, const double* M, double* R, int n, int m);
void matrix_multiply_m_v(const double* M, const double* V, double* R, int n, int m);
