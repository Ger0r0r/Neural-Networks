
#include "neural_network.hpp"

neural_network::neural_network(std::vector<int>& layers, Act_Func act, double learn_factor) : act(act), learn_factor(learn_factor) {
		this->layers = layers;
		init_weights();
	}

std::vector<double> neural_network::forward(std::vector<double>& input) {
	if (input.size() != layers[0]) {
		std::cout << "Wrong input vector to learn\n";
		return input;
	}

	std::vector<double> a = input;
	for (int i = 0; i < weights.size(); ++i) {
		a = multiply(weights[i], a);
		a = active(a);
	}

	std::vector <double> aim = {0,1,2,3,4,5,6,7,8,9};
	double MSE_error = calc_error(a, aim);
	std::cout << "MSE Error " << MSE_error << std::endl;

	return a;
}

double neural_network::learn(std::vector<double>& input, std::vector<double>& aim) {

	int layers_num = layers.size();

	// ************************************************************ //
	// __________________________CHECK_____________________________ //
	// ************************************************************ //
	if (input.size() != layers[0]) {
		std::cout << "Wrong input vector to learn\n";
		return 0;
	}
	if (aim.size() != layers[layers_num - 1]) {
		std::cout << "Wrong aim vector to learn\n";
		return 0;
	}

	// ************************************************************ //
	// _____________________START_OF_LEARNING______________________ //
	// ************************************************************ //
	std::vector <std::vector <double>> values;
	std::vector <std::vector <double>> delta_values;

	for (int i = 0; i < layers_num; ++i) {
		std::vector <double> temp_to_values (layers[i]);
		values.push_back(temp_to_values);
	}

	for (int i = 1; i < layers_num; ++i) {
		std::vector <double> temp_to_values (layers[i]);
		delta_values.push_back(temp_to_values);
	}

	values[0] = input;

	for (int i = 0; i < layers_num - 1; ++i) {
		values[i+1] = multiply(weights[i], values[i]);
		values[i+1] = active(values[i+1]);

	}

	// ************************************************************ //
	// _______________________CALC_MSE_ERROR_______________________ //
	// ************************************************************ //
	std::vector <double> & result = values[layers_num - 1];

	double MSE_error = calc_error(result, aim);

	std::cout << "MSE Error " << MSE_error << std::endl;

	// ************************************************************ //
	// ______________________BACK_PROPOGATION______________________ //
	// ************************************************************ //

	std::vector <double> diff_values = diff_active(result);
	// std::cout << "DELTAS CALC CHECK\n";
	for (int i = 0; i < layers[layers_num-1]; i++) {
		// std::cout << "Sub " << (result[i] - aim[i]) << " and diff " << diff_values[i] << "\n";
		delta_values[delta_values.size() - 1][i] = (result[i] - aim[i]) * diff_values[i];
	}

	// std::cout << "Start delta\n";
	// print_container(delta_values[delta_values.size() - 1]);
	// std::cout << "with result\n";
	// print_container(result);
	// std::cout << "and aim\n";
	// print_container(aim);

	for (int i = 0; i < weights.size() - 1; i++) {
		delta_values[delta_values.size() - 2 - i] = multiply(delta_values[delta_values.size() - 1 - i], weights[weights.size() - 1 - i]);
	}

	recalc_weights(values, delta_values);

	// std::cout << "EEEE" << std::endl;
	return MSE_error;
}

void neural_network::recalc_weights(std::vector <std::vector <double>> & values, std::vector <std::vector <double>> & deltas) {

	// std::cout << "Sizes " << values.size() << " " << deltas.size() << " " << weights.size() << "\n";

	for (int i = 0; i < weights.size(); i++){
		// std::cout << "i = " << i << "\n";
		// std::cout << "Weight " << i << " is " << weights[i].getRows() << " " << weights[i].getCols() << "\n";
		// std::cout << "Value " << i << " is " << values[i].size() << "\n";
		// std::cout << "Delta " << i << " is " << deltas[i].size() << "\n";
		for (int j = 0; j < weights[i].getRows(); j++) {
			for (int k = 0; k < weights[i].getCols(); k++) {

				weights[i](j,k) = weights[i](j,k) - learn_factor * values[i][j] * deltas[i][k];
			}
		}
	}
}

void neural_network::init_weights() {
	for (int i = 0; i < layers.size() - 1; ++i) {
		weights.emplace_back(layers[i], layers[i + 1], 0.5); // инициализация весов как 1
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

std::vector<double> neural_network::multiply(std::vector<double>& v, matrix& m) {
	std::vector<double> result(m.getRows(), 0.0);
	for (int j = 0; j < m.getRows(); ++j) {
		for (int i = 0; i < m.getCols(); ++i) {
			result[j] += m(j, i) * v[i];
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
		case Act_Func::LReLU:
			return neural_network::lrelu(z);
		default:
			throw std::invalid_argument("Unknown activation function");
	}
}
std::vector<double> neural_network::diff_active(std::vector<double>& z) {
	switch (act) {
		case Act_Func::Sigmoid:
			return neural_network::diff_sigmoid(z);
		case Act_Func::ReLU:
			return neural_network::diff_relu(z);
		case Act_Func::LReLU:
			return neural_network::diff_lrelu(z);
		default:
			throw std::invalid_argument("Unknown activation function");
	}
}

std::vector<double> neural_network::sigmoid(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < z.size(); ++i) {
		result[i] = 1.0 / (1.0 + exp(-z[i]));
	}
	return result;
}

std::vector<double> neural_network::relu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < z.size(); ++i) {
		result[i] = (z[i] > 0) ? z[i] : 0; // ReLU
	}
	return result;
}

std::vector<double> neural_network::lrelu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < z.size(); ++i) {
		result[i] = (z[i] > 0) ? z[i] : z[i] / 100;
	}
	return result;
}

std::vector<double> neural_network::diff_sigmoid(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < z.size(); ++i) {
		result[i] = z[i] * (1 - z[i]);
	}
	return result;
}

std::vector<double> neural_network::diff_relu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < z.size(); ++i) {
		result[i] = (z[i] > 0) ? 1 : 0;
	}
	return result;
}

std::vector<double> neural_network::diff_lrelu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < z.size(); ++i) {
		result[i] = (z[i] > 0) ? 1 : 0.01;
	}
	return result;
}

void print_container(const std::vector<double>& c) {
    for (double i : c)
        std::cout << i << ' ';
	std::cout << '\n';
}

std::vector<double> get_rand_container(int size, int minimum, int maximum) {

	std::vector<double> rand_input(size);

	std::random_device rd; // используется для получения начального значения
    std::mt19937 gen(rd()); // инициализация генератора
    std::uniform_int_distribution<> dis(minimum, maximum); // диапазон

    for (int i = 0; i < size; ++i) {
        rand_input[i] = dis(gen);
    }
	return rand_input;
}

double neural_network::calc_error(std::vector <double> & result, std::vector <double> & aim) {
	double MSE_error = 0;
	for (int i = 0; i < result.size(); i++) {
		double addition = std::pow(result[i] - aim[i], 2);
		MSE_error += addition;
	}
	return std::sqrt(MSE_error);
}