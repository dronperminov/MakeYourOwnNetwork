#pragma once

#include <random>
#include "Layer.hpp"

using namespace std;

class DropoutLayer : public Layer {
	int total;
	double p, q;
	std::default_random_engine generator;
	std::binomial_distribution<int> distribution;
public:	
	DropoutLayer(TensorSize outputSize, double p); // создание слоя

	void ForwardTrain(const Tensor &x); // прямое распространение
	void Forward(const Tensor &x); // прямое распространение
	void Backward(const Tensor &x, const Tensor &dout, bool needDx); // обратное распространение

	void Summary() const; // вывод информации	
};

DropoutLayer::DropoutLayer(TensorSize outputSize, double p) : Layer(outputSize, outputSize), distribution(1, 1 - p) {
	total = outputSize.width * outputSize.height * outputSize.depth;
	this->p = p;
	this->q = 1 - p;
}

// прямое распространение
void DropoutLayer::ForwardTrain(const Tensor &x) {
	for (int i = 0; i < total; i++) {
		if (distribution(generator)) {
			output[i] = x[i] / q;
			dx[i] = 1;
		}
		else {
			output[i] = 0;
			dx[i] = 0;
		}
	}
}

// прямое распространение
void DropoutLayer::Forward(const Tensor &x) {
	for (int i = 0; i < total; i++) {
		output[i] = x[i];
		dx[i] = 1;
	}
}

// обратное распространение
void DropoutLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	if (!needDx)
		return;

	for (int i = 0; i < total; i++)
		dx[i] *= dout[i];
}

// вывод информации
void DropoutLayer::Summary() const {
	cout << "|              dropout | "  << setw(12) << inputSize << " | " << setw(13) << outputSize << " | " << setw(13) << (0) << " | p: " << p  << endl;
}
