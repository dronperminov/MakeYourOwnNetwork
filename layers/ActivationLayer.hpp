#pragma once

#include "Layer.hpp"

using namespace std;

class ActivationLayer : public Layer {
	string function; // тип активационной функции

public:	
	ActivationLayer(int inputs, int outputs, const string &function); // создание слоя

	vector<double> Forward(const vector<double> &x); // прямое распространение
	vector<double> Backward(const vector<double> &x, const vector<double> &dout); // обратное распространение

	void Summary() const; // вывод информации
};

ActivationLayer::ActivationLayer(int inputs, int outputs, const string &function) : Layer(inputs, outputs) {
	this->function = function; // запоминаем функцию активации
}

// прямое распространение
vector<double> ActivationLayer::Forward(const vector<double> &x) {	
	for (int i = 0; i < outputs; i++) {
		// применяем функцию активации
		if (function == "sigmoid") {
			output[i] = 1.0 / (exp(-x[i]) + 1);
			dx[i] = output[i] * (1 - output[i]);
		}
		else if (function == "tanh") {
			output[i] = tanh(x[i]);
			dx[i] = 1 - output[i] * output[i];
		}
		else if (function == "relu") {
			output[i] = x[i] > 0 ? x[i] : 0;
			dx[i] = x[i] > 0 ? 1 : 0;
		}
	}

	return output; // возвращаем результирующий вектор
}

// обратное распространение
vector<double> ActivationLayer::Backward(const vector<double> &x, const vector<double> &dout) {
	for (int i = 0; i < outputs; i++)
		dx[i] *= dout[i]; // умножаем градиенты активации на выходные градиенты

	return dx; // возвращаем градиенты по входам
}

// вывод информации
void ActivationLayer::Summary() const {
	string name = "activation '" + function + "'";
	cout << "| " << setw(20) << name << " | "  << setw(12) << inputs << " | " << setw(13) << outputs << " | " << setw(13) << (0) << " |" << endl;
}
