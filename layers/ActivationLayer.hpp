#pragma once

#include "Layer.hpp"

using namespace std;

class ActivationLayer : public Layer {
	string function; // тип активационной функции
	int total;

public:	
	ActivationLayer(TensorSize outputSize, const string &function); // создание слоя

	void Forward(const Tensor &x); // прямое распространение
	void Backward(const Tensor &x, const Tensor &dout, bool needDx); // обратное распространение

	void Summary() const; // вывод информации
};

ActivationLayer::ActivationLayer(TensorSize outputSize, const string &function) : Layer(outputSize, outputSize) {
	this->function = function; // запоминаем функцию активации
	total = outputSize.width * outputSize.height * outputSize.depth;

	if (function != "sigmoid" && function != "tanh" && function != "relu")
		throw runtime_error("unknown function '" + function + "'");
}

// прямое распространение
void ActivationLayer::Forward(const Tensor &x) {
	for (int i = 0; i < total; i++) {
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
void ActivationLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	if (!needDx)
		return;

	for (int i = 0; i < total; i++)
		dx[i] *= dout[i]; // умножаем градиенты активации на выходные градиенты
}

// вывод информации
void ActivationLayer::Summary() const {
	string name = "activation '" + function + "'";
	cout << "| " << setw(20) << name << " | "  << setw(12) << inputSize << " | " << setw(13) << outputSize << " | " << setw(13) << (0) << " |" << endl;
}
