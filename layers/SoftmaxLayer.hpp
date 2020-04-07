#pragma once

#include "Layer.hpp"

using namespace std;

class SoftmaxLayer : public Layer {
	int total;
public:	
	SoftmaxLayer(TensorSize outputSize); // создание слоя

	void Forward(const Tensor &x); // прямое распространение
	void Backward(const Tensor &x, const Tensor &dout, bool needDx); // обратное распространение

	void Summary() const; // вывод информации	
};

SoftmaxLayer::SoftmaxLayer(TensorSize outputSize) : Layer(outputSize, outputSize) {
	total = outputSize.width * outputSize.height * outputSize.depth;
}

// прямое распространение
void SoftmaxLayer::Forward(const Tensor &x) {
	double sum = 0;

	for (int i = 0; i < total; i++) {
		output[i] = exp(x[i]);
		sum += output[i];
	}

	for (int i = 0; i < total; i++)
		output[i] /= sum;
}

// обратное распространение
void SoftmaxLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	if (!needDx)
		return;

	for (int i = 0; i < total; i++) {
		dx[i] = 0;

		for (int j = 0; j < total; j++)
			dx[i] += dout[j] * output[i] * ((i == j) - output[j]);
	}
}

// вывод информации
void SoftmaxLayer::Summary() const {
	cout << "|              softmax | "  << setw(12) << inputSize << " | " << setw(13) << outputSize << " | " << setw(13) << (0) << " |" << endl;
}
