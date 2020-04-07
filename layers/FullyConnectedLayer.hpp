#pragma once

#include "Layer.hpp"

using namespace std;

class FullyConnectedLayer : public Layer {
	int inputs;
	int outputs;
	vector<vector<double>> w; // матрица весовых коэффициентов
	vector<vector<double>> dw; // градиенты весовых коэффициентов

	void InitializeWeights(); // инициализация весовых коэффициентов
public:	
	FullyConnectedLayer(TensorSize inputSize, int outputs); // создание слоя

	void Forward(const Tensor &x); // прямое распространение
	void Backward(const Tensor &x, const Tensor &dout, bool needDx); // обратное распространение
	void UpdateWeights(double learningRate); // обновление весовых коэффициентов

	void PrintWeights() const; // вывод весовых коэффициентов
	void Summary() const; // вывод информации
};

FullyConnectedLayer::FullyConnectedLayer(TensorSize inputSize, int outputs) : Layer(inputSize, { 1, 1, outputs }) {
	this->inputs = inputSize.width * inputSize.height * inputSize.depth;
	this->outputs = outputs;
	w = vector<vector<double>>(outputs, vector<double>(inputs + 1)); // выделяем память под весовые коэффициенты и выходной вектор
	dw = vector<vector<double>>(outputs, vector<double>(inputs + 1)); // выделяем место под градиенты

	InitializeWeights();
}

// инициализация весовых коэффициентов
void FullyConnectedLayer::InitializeWeights() {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			w[i][j] = GetRnd(-0.5, 0.5);

		w[i][inputs] = GetRnd(-0.5, 0.5);
	}
}

// прямое распространение
void FullyConnectedLayer::Forward(const Tensor &x) {
	for (int i = 0; i < outputs; i++) {
		double y = w[i][inputs];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

		output[i] = y;
	}
}

// обратное распространение
void FullyConnectedLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	// считаем градиенты весовых коэффициентов
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			dw[i][j] += dout[i] * x[j];

		dw[i][inputs] += dout[i];
	}

	if (!needDx)
		return;

	// считаем градиенты по входам
	for (int i = 0; i < inputs; i++) {
		dx[i] = 0;

		for (int j = 0; j < outputs; j++)
			dx[i] += w[j][i] * dout[j];
	}
}

// обновление весовых коэффициентов
void FullyConnectedLayer::UpdateWeights(double learningRate) {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++) {
			w[i][j] -= learningRate * dw[i][j]; // выполняем шаг градиентного спуска для веса
			dw[i][j] = 0;
		}

		w[i][inputs] -= learningRate * dw[i][inputs]; // выполняем шаг градиентного спуска для смещения
		dw[i][inputs] = 0;
	}
}

// вывод весовых коэффициентов
void FullyConnectedLayer::PrintWeights() const {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			cout << w[i][j] << " ";

		cout << w[i][inputs] << endl;
	}
}

// вывод информации
void FullyConnectedLayer::Summary() const {
	cout << "|      fully connected | " << setw(12) << inputSize << " | " << setw(13) << outputSize << " | " << setw(13) << ((inputs + 1) * outputs) << " |" << endl;
}