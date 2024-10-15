#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
// #include <x86intrin.h>  // Для использования rdtsc
#include "neural_network.hpp"

int main () {

    // unsigned long long start, end;

	std::vector <int> layers = {100, 1000, 10};
	std::vector <double> aim = {0,1,2,3,4,5,6,7,8,9};

	double l_factor = 0.001;

	neural_network stupid_net(layers, Act_Func::LReLU, l_factor);

	std::cout << "Training vector" << std::endl;
    // Инициализируем генератор случайных чисел
	std::vector <double> rand_input = get_rand_container(layers[0], 0, 9);
	// std::fill(rand_input.data(), rand_input.data() + 100, 5);

	// std::vector <double> rand_input = {10,20,30};

	print_container(rand_input);
	std::vector <double> res = stupid_net.forward(rand_input);

	// return 0;
	// std::cout << "<WARNING>" << std::endl;

	// start = __rdtsc();  // Получение начального значения тактов

	int count_of_iteration = 0;

	double got_error, new_got_error = 0.0;
	do {
		count_of_iteration++;
		got_error = new_got_error;
		// std::cout << "<WARNING>" << std::endl;
		new_got_error = stupid_net.learn(rand_input, aim, count_of_iteration);

		rand_input = get_rand_container(layers[0], 0, 9);

		if (count_of_iteration == 2000) {
			l_factor /= 1000000;
			stupid_net.update_learning_factor(l_factor);
			std::cout << "~~~~~~~~~~ UPDATE ~~~~~~~~~~" << std::endl;
		}
		if (count_of_iteration == 4000) {
			l_factor /= 1000000;
			stupid_net.update_learning_factor(l_factor);
			std::cout << "~~~~~~~~~~ UPDATE ~~~~~~~~~~" << std::endl;
		}

	} while (new_got_error > 0.0001 && got_error != new_got_error && count_of_iteration != 6000);
	// } while (new_got_error > 0.001 && got_error != new_got_error);

    // end = __rdtsc();  // Получение конечного значения тактов

    // std::cout << "Количество тактов процессора: " << (end - start) << " тактов\n";


	res = stupid_net.forward(rand_input);
	print_container(res);
	std::cout << "Number of ephos " << count_of_iteration << std::endl;

	std::cout << "Check\n";
	for (int i = 0; i < 5; i++) {
		stupid_net.forward(rand_input);
		rand_input = get_rand_container(layers[0], 0, 9);
	}

	return 0;
}