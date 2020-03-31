#pragma once

#include "Layer.hpp"

using namespace std;

class FullyConnectedLayer : public Layer {
	vector<vector<double>> w; // матрица весовых коэффициентов
	vector<vector<double>> dw; // градиенты весовых коэффициентов

	void InitializeWeights(); // инициализация весовых коэффициентов
public:	
	FullyConnectedLayer(int inputs, int outputs); // создание слоя

	vector<double> Forward(const vector<double> &x); // прямое распространение
	vector<double> Backward(const vector<double> &x, const vector<double> &dout); // обратное распространение
	void UpdateWeights(double learningRate); // обновление весовых коэффициентов

	void PrintWeights() const; // вывод весовых коэффициентов
	void Summary() const; // вывод информации
};

FullyConnectedLayer::FullyConnectedLayer(int inputs, int outputs) : Layer(inputs, outputs) {
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
vector<double> FullyConnectedLayer::Forward(const vector<double> &x) {	
	for (int i = 0; i < outputs; i++) {
		double y = w[i][inputs];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

		output[i] = y;
	}

	return output; // возвращаем результирующий вектор
}

// обратное распространение
vector<double> FullyConnectedLayer::Backward(const vector<double> &x, const vector<double> &dout) {
	// считаем градиенты весовых коэффициентов
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			dw[i][j] += dout[i] * x[j];

		dw[i][inputs] += dout[i];
	}

	// считаем градиенты по входам
	for (int i = 0; i < inputs; i++) {
		dx[i] = 0;

		for (int j = 0; j < outputs; j++)
			dx[i] += w[j][i] * dout[j];
	}

	return dx; // возвращаем градиенты по входам
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
	cout << "|      fully connected | " << setw(12) << inputs << " | " << setw(13) << outputs << " | " << setw(13) << ((inputs + 1) * outputs) << " |" << endl;
}