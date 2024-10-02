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

	neural_network stupid_net(layers, Act_Func::LReLU , 0.000000000001);

    // Инициализируем генератор случайных чисел
	std::vector rand_input = get_rand_container(layers[0]);

	std::vector <double> res = stupid_net.forward(rand_input);
	std::cout << "Training vector\n";
	print_container(rand_input);



	// start = __rdtsc();  // Получение начального значения тактов

	int count_of_iteration = 0;

	double got_error, new_got_error;
	do {
		count_of_iteration++;
		got_error = new_got_error;
		new_got_error = stupid_net.learn(rand_input, aim);

		rand_input = get_rand_container(layers[0]);

	} while (new_got_error > 0.001 and got_error != new_got_error);

    // end = __rdtsc();  // Получение конечного значения тактов

    // std::cout << "Количество тактов процессора: " << (end - start) << " тактов\n";


	res = stupid_net.forward(rand_input);
	print_container(res);
	std::cout << "Number of ephos " << count_of_iteration << "\n";

	std::cout << "Check\n";
	for (int i = 0; i < 5; i++) {
		stupid_net.forward(rand_input);
		rand_input = get_rand_container(layers[0]);
	}

	return 0;
}