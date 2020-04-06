#include <iostream>
#include "Layers/SoftmaxLayer.hpp"

void PrintVector(const vector<double> &x, const char* msg) {
    cout << msg << "[";

    for (int i = 0; i < x.size(); i++)
        cout << setw(4) << x[i] << " ";

    cout << "]" << endl;
}

int main() {
    SoftmaxLayer layer(3);
    vector<double> x = { 0, 2, 1 };
    layer.Forward(x);
    vector<double> y = layer.GetOutput();
    vector<double> dout = { 0.5, 1, -0.5 };
    layer.Backward(x, dout, true);
    vector<double> dx = layer.GetDx();
    PrintVector(x, "x: ");
    PrintVector(y, "y: ");
    PrintVector(dx, "dx: ");
}