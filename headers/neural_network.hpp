#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include "matrix.hpp"

void print_container(const std::vector<double>& c);
std::vector<double> get_rand_container(int size, int minimum = 0, int maximum = 100);

// Перечисление для выбора функции активации
enum class Act_Func {
	Sigmoid,
	ReLU,
	LReLU
};

class neural_network {
	public:
		neural_network(std::vector<int>& layers, Act_Func act, double learn_factor);

		std::vector<double> forward(std::vector<double>& input);
		double learn(std::vector<double>& input, std::vector<double>& aim);

	private:
		std::vector<int> layers;
		std::vector<matrix> weights;
		Act_Func act;
		double learn_factor;

		void init_weights();
		void recalc_weights(std::vector <std::vector <double>> & values, std::vector <std::vector <double>> & deltas);
		std::vector<double> multiply(matrix& m, std::vector<double>& v);
		std::vector<double> multiply(std::vector<double>& v, matrix& m);
		std::vector<double> active(std::vector<double>& z);
		std::vector<double> diff_active(std::vector<double>& z);
		std::vector<double> sigmoid(std::vector<double>& z);
		std::vector<double> relu(std::vector<double>& z);
		std::vector<double> lrelu(std::vector<double>& z);
		std::vector<double> diff_sigmoid(std::vector<double>& z);
		std::vector<double> diff_relu(std::vector<double>& z);
		std::vector<double> diff_lrelu(std::vector<double>& z);
};