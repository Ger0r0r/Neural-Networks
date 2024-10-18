#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include "neural_network.hpp"
#include <windows.h>
#include <chrono>

// Функция для получения времени, затраченного процессом на CPU
double getCpuTime() {
	FILETIME createTime, exitTime, kernelTime, userTime;
	if (GetProcessTimes(GetCurrentProcess(), &createTime, &exitTime, &kernelTime, &userTime)) {
		// Преобразуем FILETIME в 64-битное значение
		ULARGE_INTEGER uKernelTime;
		uKernelTime.LowPart = kernelTime.dwLowDateTime;
		uKernelTime.HighPart = kernelTime.dwHighDateTime;

		ULARGE_INTEGER uUserTime;
		uUserTime.LowPart = userTime.dwLowDateTime;
		uUserTime.HighPart = userTime.dwHighDateTime;

		// Суммируем время в пользовательском и системном режимах и переводим в секунды
		return (uKernelTime.QuadPart + uUserTime.QuadPart) / 1e7; // 1e7 - для перевода в секунды
	}
	return 0.0;
}

int main () {

	SetConsoleOutputCP(CP_UTF8);

	std::vector <int> layers = {100, 1000, 10};
	std::vector <double> aim = {0,1,2,3,4,5,6,7,8,9};

	double l_factor = 0.001;

	neural_network stupid_net(layers, Act_Func::LReLU, l_factor);

	std::cout << "Training vector" << std::endl;
	std::vector <double> rand_input = get_rand_container(layers[0]);
	// std::fill(rand_input.data(), rand_input.data() + 100, 5);

	print_container(rand_input);
	std::vector <double> res = stupid_net.forward(rand_input, aim);

	// return 0;

	int count_of_iteration = 0;
	double sum_errors = 0;
	std::vector <double> errors (100, 0);
	double got_error, new_got_error = 0.0;
	bool condition_to_stop = 1;


	SYSTEM_INFO sysInfo;
	GetSystemInfo(&sysInfo);
	unsigned int numCores = sysInfo.dwNumberOfProcessors;
	auto startRealTime = std::chrono::high_resolution_clock::now();
	double startCpuTime = getCpuTime();

	do {
		count_of_iteration++;
		got_error = new_got_error;
		new_got_error = stupid_net.learn(rand_input, aim, count_of_iteration);

		if (count_of_iteration > 100) {
			sum_errors -= errors.data()[(count_of_iteration - 1) % 100];
		}
		errors.data()[(count_of_iteration - 1) % 100] = new_got_error;
		sum_errors += errors.data()[(count_of_iteration - 1) % 100];

		// std::cout << "MSE Error " << new_got_error << " mean " << sum_errors / 100 << std::endl;

		rand_input = get_rand_container(layers[0]);

		condition_to_stop = !(count_of_iteration > 100 && std::abs((sum_errors / 100) - new_got_error) < 0.1);

	} while (new_got_error > 0.0001 && got_error != new_got_error && count_of_iteration != 5000 && condition_to_stop);
	// } while (new_got_error > 0.001 && got_error != new_got_error);

	auto endRealTime = std::chrono::high_resolution_clock::now();
	double endCpuTime = getCpuTime();
	double totalCpuTime = endCpuTime - startCpuTime;
	std::chrono::duration<double> realTime = endRealTime - startRealTime;
	double cpuUsagePercent = (totalCpuTime / (realTime.count() * numCores)) * 100;

	std::cout << "Общее CPU время: " << totalCpuTime << " секунд" << std::endl;
	std::cout << "Реальное время выполнения: " << realTime.count() << " секунд" << std::endl;
	std::cout << "Количество логических ядер: " << numCores << std::endl;
	std::cout << "Процент загрузки CPU программой (учитывая все ядра): " << cpuUsagePercent << "%" << std::endl;


	res = stupid_net.forward(rand_input, aim);
	print_container(res);
	std::cout << "Number of ephos " << count_of_iteration << std::endl;

	std::cout << "Check\n";
	for (int i = 0; i < 5; i++) {
		stupid_net.forward(rand_input, aim);
		rand_input = get_rand_container(layers[0]);
	}

	return 0;
}