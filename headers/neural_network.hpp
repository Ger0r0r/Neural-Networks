#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include "matrix.hpp"

#define BETA_MOMENTUM 0.9
#define BETA_GRADIENT 0.99
#define BORDER_FROM_ZERO 0.00000001

void print_container(const std::vector<double>& c);
std::vector<double> get_rand_container(int size, int minimum = 0, int maximum = 100);

// Перечисление для выбора функции активации
enum class Act_Func {
	Sigmoid,
	ReLU,
	LReLU
};

// Перечисление для выбора метода обучения
enum class Grad_Func {
	Grad,
	Momentum,
	Nesterov,
	AdMomentum,
	AdNesterov,
	Adagrad,
	RMSProp,
	Adam
};

class neural_network {
	public:
		neural_network(std::vector<int>& layers, Act_Func act, double learn_factor);

		std::vector<double> forward(std::vector<double>& input);
		double learn(std::vector<double>& input, std::vector<double>& aim, int epoch);
		void update_learning_factor (double new_learn_factor);

	private:
		std::vector<int> layers;
		std::vector<matrix> weights;
		std::vector<matrix> momentum;
		std::vector<matrix> sqgrad;
		Act_Func act;
		double learn_factor;

		void init_weights();
		void recalc_weights(std::vector <std::vector <double>> & values, std::vector <std::vector <double>> & deltas, int epoch);
		double calc_error(std::vector <double> & result, std::vector <double> & aim);
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