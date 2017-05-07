#include <iostream>
#include <random>
#include <cmath>
#include <fstream>

int main()
{
	unsigned num_patterns = 300;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::default_random_engine generator;

	// values near the mean are the most likely
	// standard deviation affects the dispersion of generated values from the mean
	std::normal_distribution<> normal(0, 0.2); // params: mean - std dev
	std::uniform_real_distribution<double> uniform(-3.0, 3.0); // params: min - max

	std::ofstream fun_out("funout.csv");
	std::ofstream fun_out_test("funout_test.csv");

	double y1, y2, x;
	for (int n = 0; n < num_patterns; ++n) {
		x = uniform(generator);
		y1 = x + 2 * sin(1.5*x);
		y2 = x + 2 * cos(0.5*x);
		fun_out_test << x << "," << y1 << "," << y2 << std::endl;

		x = uniform(generator);
		y1 = x + 2 * sin(1.5*x);
		y2 = x + 2 * cos(0.5*x);
		y1 += normal(generator);
		y2 += normal(generator);
		std::cout << x << "," << y1 << "," << y2 << std::endl;
		fun_out << x << "," << y1 << "," << y2 << std::endl;
	}

	fun_out.close();
	fun_out_test.close();
}