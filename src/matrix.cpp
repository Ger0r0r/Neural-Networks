#include "matrix.hpp"

matrix::matrix(int ll, int cc, double init_value = 1.0) : rows(ll), cols(cc), data(ll, std::vector<double>(cc, init_value)) {};
matrix::~matrix() {};

double& matrix::operator()(int row, int col) {
	return data[row][col];
}
int matrix::getRows() {
	return rows;
}
int matrix::getCols() {
	return cols;
}

void matrix::print() {
	for (auto& row : data) {
		for (auto& val : row) {
			std::cout << val << " ";
		}
		std::cout << std::endl;
	}
}

matrix matrix_multiply(matrix& a, matrix& b) {
	if (a.getCols() != b.getRows()) {
		throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix.");
	}

	int ar = a.getRows();
	int ac = a.getCols();
	int bc = b.getCols();

	matrix result(ar, bc);
	for (int i = 0; i < ar; ++i) {
		for (int j = 0; j < bc; ++j) {
			for (int k = 0; k < ac; ++k) {
				result(i, j) += a(i, k) * b(k, j);
			}
		}
	}
	return result;
}