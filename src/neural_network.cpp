
#include "neural_network.hpp"

neural_network::neural_network(std::vector<int>& layers, Act_Func act, double learn_factor) : act(act), learn_factor(learn_factor) {
		this->layers = layers;
		init_weights();
	}

std::vector<double> neural_network::forward(std::vector<double>& input) {
	if (static_cast<int>(input.size()) != layers[0]) {
		std::cout << "Wrong input vector to learn\n";
		return input;
	}

	std::vector<double> a = input;
	for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
		// std::cout << "a size " << a.size() << "\n";
		std::vector<double> r (weights[i].getCols());
		// std::cout << "w check\n";
		matrix_multiply_v_m(a.data(), weights[i].data, r.data(), weights[i].getRows(), weights[i].getCols());
		// std::cout << "r size " << r.size() << "\n";
		// std::cout << "r check " << r[0] << "\n";
		// std::cout << "Multiply " << i + 1 << std::endl;
		// print_container(a);
		// std::cout << "By " << i + 1 << std::endl;
		// weights[i].print();
		// std::cout << "Result " << i + 1 << std::endl;
		// print_container(r);
		a = active(r);
		// std::cout << "Activate " << i + 1 << std::endl;
		// print_container(a);
	}

	std::vector <double> aim = {0,1,2,3,4,5,6,7,8,9};
	double MSE_error = calc_error(a, aim);
	std::cout << "Result" << std::endl;
	print_container(a);
	std::cout << "MSE Error " << MSE_error << std::endl;

	return a;
}

double neural_network::learn(std::vector<double>& input, std::vector<double>& aim, int epoch) {

	// std::cout << "<WARNING>" << std::endl;
	int layers_num = layers.size();

	// ************************************************************ //
	// __________________________CHECK_____________________________ //
	// ************************************************************ //
	if (static_cast<int>(input.size()) != layers[0]) {
		std::cout << "Wrong input vector to learn\n";
		return 0;
	}
	if (static_cast<int>(aim.size()) != layers[layers_num - 1]) {
		std::cout << "Wrong aim vector to learn\n";
		return 0;
	}
	// std::cout << "<WARNING>" << std::endl;

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
	// std::cout << "<WARNING>" << std::endl;

	for (int i = 0; i < layers_num - 1; ++i) {
		std::vector<double> r (weights[i].getCols());
		matrix_multiply_v_m(values[i].data(), weights[i].data, r.data(), weights[i].getRows(), weights[i].getCols());
		values[i+1] = active(r);
	}

	// ************************************************************ //
	// _______________________CALC_MSE_ERROR_______________________ //
	// ************************************************************ //
	std::vector <double> & result = values[layers_num - 1];

	double MSE_error = calc_error(result, aim);

	// std::cout << "<WARNING>" << std::endl;
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


	// std::cout << "UPDATING WEIGHTS" << std::endl;
	for (int i = 0; i < static_cast<int>(weights.size()) - 1; i++) {

		// std::cout << "UPDATING WEIGHT " << delta_values.size() - 2 - i << std::endl;
		// std::cout << "Size of prev " << delta_values[delta_values.size() - 1 - i].size() << std::endl;
		// std::cout << "Size of next " << delta_values[delta_values.size() - 2 - i].size() << std::endl;

		matrix_multiply_m_v(
			weights[weights.size() - 1 - i].data,
			delta_values[delta_values.size() - 1 - i].data(),
			delta_values[delta_values.size() - 2 - i].data(),
			weights[weights.size() - 1 - i].getRows(),
			weights[weights.size() - 1 - i].getCols()
		);
	}

	recalc_weights(values, delta_values, epoch);

	// std::cout << "EEEE" << std::endl;
	return MSE_error;
}

void neural_network::recalc_weights(std::vector <std::vector <double>> & values, std::vector <std::vector <double>> & deltas, int epoch) {

	double correction_momentum = 1 - std::pow(BETA_MOMENTUM, epoch);
	double correction_gradient = 1 - std::pow(BETA_GRADIENT, epoch);
	// std::cout << "Sizes " << values.size() << " " << deltas.size() << " " << weights.size() << "\n";

	for (int i = 0; i < static_cast<int>(weights.size()); i++){
		// std::cout << "i = " << i << "\n";
		// std::cout << "Weight " << i << " is " << weights[i].getRows() << " " << weights[i].getCols() << "\n";
		// std::cout << "Value " << i << " is " << values[i].size() << "\n";
		// std::cout << "Delta " << i << " is " << deltas[i].size() << "\n";
		for (int j = 0; j < weights[i].getRows(); j++) {
			for (int k = 0; k < weights[i].getCols(); k++) {
				double grad = values[i][j] * deltas[i][k];

				momentum[i](j,k) = momentum[i](j,k) * BETA_MOMENTUM + (1 - BETA_MOMENTUM) * grad;
				sqgrad[i](j,k) = sqgrad[i](j,k) * BETA_GRADIENT + (1 - BETA_GRADIENT) * grad * grad;

				double new_momentum = momentum[i](j,k) / correction_momentum;
				double new_gradient = sqgrad[i](j,k) / correction_gradient;

				weights[i](j,k) = weights[i](j,k) - learn_factor * new_momentum / (std::sqrt(new_gradient + BORDER_FROM_ZERO));
			}
		}
	}
}

void neural_network::init_weights() {
	for (int i = 0; i < static_cast<int>(layers.size()) - 1; ++i) {
		// std::cout << "MAKE WEIGHT" << std::endl;
		weights.emplace_back(layers[i], layers[i+1], 0.5);
		// std::cout << "size " << weights.size() << " " << layers[i] << " " << layers[i+1] << std::endl;
		// std::cout << "CHECK NEW WEIGHT" << std::endl;
		// weights[i].print();


		momentum.emplace_back(layers[i], layers[i+1], 0.0);
		sqgrad.emplace_back(layers[i], layers[i+1], 0.0);
	}
	// std::cout << "DONE" << std::endl;
}

// std::vector<double> neural_network::multiply(matrix& m, std::vector<double>& v) {
// 	std::vector<double> result(m.getCols(), 0.0);
// 	for (int j = 0; j < m.getCols(); ++j) {
// 		for (int i = 0; i < m.getRows(); ++i) {
// 			result[j] += m(i, j) * v[i];
// 		}
// 	}
// 	return result;
// }

// std::vector<double> neural_network::multiply(std::vector<double>& v, matrix& m) {
// 	std::vector<double> result(m.getRows(), 0.0);
// 	for (int j = 0; j < m.getRows(); ++j) {
// 		for (int i = 0; i < m.getCols(); ++i) {
// 			result[j] += m(j, i) * v[i];
// 		}
// 	}
// 	return result;
// }

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
	for (int i = 0; i < static_cast<int>(z.size()); ++i) {
		result[i] = 1.0 / (1.0 + exp(-z[i]));
	}
	return result;
}

std::vector<double> neural_network::relu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < static_cast<int>(z.size()); ++i) {
		result[i] = (z[i] > 0) ? z[i] : 0; // ReLU
	}
	return result;
}

std::vector<double> neural_network::lrelu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < static_cast<int>(z.size()); ++i) {
		result[i] = (z[i] > 0) ? z[i] : z[i] / 100;
	}
	return result;
}

std::vector<double> neural_network::diff_sigmoid(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < static_cast<int>(z.size()); ++i) {
		result[i] = z[i] * (1 - z[i]);
	}
	return result;
}

std::vector<double> neural_network::diff_relu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < static_cast<int>(z.size()); ++i) {
		result[i] = (z[i] > 0) ? 1 : 0;
	}
	return result;
}

std::vector<double> neural_network::diff_lrelu(std::vector<double>& z) {
	std::vector<double> result(z.size());
	for (int i = 0; i < static_cast<int>(z.size()); ++i) {
		result[i] = (z[i] > 0) ? 1 : 0.01;
	}
	return result;
}

void print_container(const std::vector<double>& c) {
    for (double i : c) {
        std::cout << i << ' ';
	}
	std::cout << std::endl;
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
	// std::cout << "TEST " << MSE_error << "\n";
	// print_container(result);
	// print_container(aim);
	for (int i = 0; i < static_cast<int>(result.size()); i++) {
		double addition = std::pow(result[i] - aim[i], 2);
		MSE_error += addition;
	}
	// std::cout << "TEST " << MSE_error << "\n";
	return std::sqrt(MSE_error);
}


void neural_network::update_learning_factor (double new_learn_factor) {
	learn_factor = new_learn_factor;
}