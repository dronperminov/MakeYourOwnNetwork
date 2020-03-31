#pragma once

#include "Layer.hpp"

using namespace std;

class SoftmaxLayer : public Layer {
public:	
	SoftmaxLayer(int outputs); // создание слоя

	void Forward(const vector<double> &x); // прямое распространение
	void Backward(const vector<double> &x, const vector<double> &dout, bool needDx); // обратное распространение

	void Summary() const; // вывод информации	
};

SoftmaxLayer::SoftmaxLayer(int outputs) : Layer(outputs, outputs) {
	
}

// прямое распространение
void SoftmaxLayer::Forward(const vector<double> &x) {
	double sum = 0;

	for (int i = 0; i < outputs; i++) {
		output[i] = exp(x[i]);
		sum += output[i];
	}

	for (int i = 0; i < outputs; i++)
		output[i] /= sum;
}

// обратное распространение
void SoftmaxLayer::Backward(const vector<double> &x, const vector<double> &dout, bool needDx) {
	if (!needDx)
		return;

	for (int i = 0; i < outputs; i++) {
		dx[i] = 0;

		for (int j = 0; j < outputs; j++)
			dx[i] += dout[j] * output[i] * ((i == j) - output[j]);
	}
}

// вывод информации
void SoftmaxLayer::Summary() const {
	cout << "|              softmax | "  << setw(12) << inputs << " | " << setw(13) << outputs << " | " << setw(13) << (0) << " |" << endl;
}
