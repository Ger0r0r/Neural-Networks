
#include "neural_network.hpp"

neural_network::neural_network(std::vector<int>& layers, Act_Func act) : act(act) {
		this->layers = layers;
		init_weights();
	}

std::vector<double> neural_network::forward(std::vector<double>& input) {
	std::vector<double> a = input;
	for (size_t i = 0; i < weights.size(); ++i) {
		a = multiply(weights[i], a);
		a = active(a);
	}
	return a;
}

void neural_network::init_weights() {
	for (size_t i = 0; i < layers.size() - 1; ++i) {
		weights.emplace_back(layers[i], layers[i + 1], 1.0); // инициализация весов как 1
	}
}

std::vector<double> neural_network::multiply(matrix& m, std::vector<double>& v) {
	std::vector<double> result(m.getCols(), 0.0);
	for (int j = 0; j < m.getCols(); ++j) {
		for (int i = 0; i < m.getRows(); ++i) {
			result[j] += m(i, j) * v[i];
		}
	}
	return result;
}

std::vector<double> neural_network::active(std::vector<double>& z) {
	switch (act) {
		case Act_Func::Sigmoid:
			return neural_network::sigmoid(z);
		case Act_Func::ReLU:
			return neural_network::relu(z);
		default:
			throw std::invalid_argument("Unknown activation function");
	}
}

std::vector<double> neural_network::sigmoid(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (size_t i = 0; i < z.size(); ++i) {
		result[i] = 1.0 / (1.0 + exp(-z[i]));
	}
	return result;
}

std::vector<double> neural_network::relu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (size_t i = 0; i < z.size(); ++i) {
		result[i] = std::max(0.0, z[i]); // ReLU
	}
	return result;
}