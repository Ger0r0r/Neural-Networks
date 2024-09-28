#include <iostream>
#include <vector>

class matrix {
	public:
		matrix(int ll, int cc, double init_value);
		~matrix();

		double& operator()(int row, int col) {};
		int getRows();
		int getCols();
		void print() {};
	private:
		int rows, cols;
		std::vector<std::vector <double>> data;
};

matrix matrix_multiply(const matrix& a, const matrix& b) {}