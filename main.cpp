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

	vector<double> dout = { 0.1, 0.2, -0.5 };
	vector<double> dx = layer.Backward(x, dout);

	cout << "dout: " << dout[0] << " " << dout[1] << " " << dout[2] << endl;
	cout << "dx: " << dx[0] << " " << dx[1] << endl;

}