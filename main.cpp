#include <iostream>
#include "Layer.hpp"

int main() {
	Layer layer(2, 3, "sigmoid");
	vector<double> x = { 0.5, -0.3 };
	vector<double> y = layer.Forward(x);

	cout << "layer weights: " << endl;
	layer.PrintWeights();
	cout << "x: " << x[0] << " " << x[1] << endl;
	cout << "y: " << y[0] << " " << y[1] << " " << y[2] << endl;
}