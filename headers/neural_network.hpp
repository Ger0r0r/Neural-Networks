#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "matrix.hpp"

// Перечисление для выбора функции активации
enum class Act_Func {
	Sigmoid,
	ReLU
};

class neural_network {
	public:
		neural_network(std::vector<int>& layers, Act_Func act) {}

		std::vector<double> forward(std::vector<double>& input) {}

	private:
		std::vector<int> layers;
		std::vector<matrix> weights;
		Act_Func act;

		void init_weights() {};
		std::vector<double> multiply(matrix& m, std::vector<double>& v) {};
		std::vector<double> active(std::vector<double>& z) {};
		std::vector<double> sigmoid(std::vector<double>& z) {};
		std::vector<double> relu(std::vector<double>& z) {};
};