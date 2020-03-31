#pragma once

#include "Layer.hpp"

using namespace std;

class ActivationLayer : public Layer {
	string function; // тип активационной функции

public:	
	ActivationLayer(int outputs, const string &function); // создание слоя

	void Forward(const vector<double> &x); // прямое распространение
	void Backward(const vector<double> &x, const vector<double> &dout, bool needDx); // обратное распространение

	void Summary() const; // вывод информации
};

ActivationLayer::ActivationLayer(int outputs, const string &function) : Layer(outputs, outputs) {
	this->function = function; // запоминаем функцию активации

	if (function != "sigmoid" && function != "tanh" && function != "relu")
		throw runtime_error("unknown function '" + function + "'");
}

// прямое распространение
void ActivationLayer::Forward(const vector<double> &x) {
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
}

// обратное распространение
void ActivationLayer::Backward(const vector<double> &x, const vector<double> &dout, bool needDx) {
	if (!needDx)
		return;

	for (int i = 0; i < outputs; i++)
		dx[i] *= dout[i]; // умножаем градиенты активации на выходные градиенты
}

// вывод информации
void ActivationLayer::Summary() const {
	string name = "activation '" + function + "'";
	cout << "| " << setw(20) << name << " | "  << setw(12) << inputs << " | " << setw(13) << outputs << " | " << setw(13) << (0) << " |" << endl;
}
