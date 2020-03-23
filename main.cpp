#include <iostream>
#include "Network.hpp"

int main() {
	Network network(2);
	network.AddLayer(5, "sigmoid");
	network.AddLayer(3, "tanh");

	vector<double> x = { 0.5, -0.3 };
	vector<double> y = network.Forward(x);

	cout << "x: " << x[0] << " " << x[1] << endl;
	cout << "y: " << y[0] << " " << y[1] << " " << y[2] << endl;

	cout << "Network weights: " << endl;
	network.Print();
}