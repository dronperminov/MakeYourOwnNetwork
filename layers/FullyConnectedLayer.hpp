#pragma once

#include "Layer.hpp"

using namespace std;

class FullyConnectedLayer : public Layer {
	string function; // тип активационной функции
	vector<vector<double>> w; // матрица весовых коэффициентов
	vector<double> b; // вектор весов смещения
	
	vector<vector<double>> dw; // градиенты весовых коэффициентов
	vector<double> db; // градиенты весов смещения
	vector<double> df; // градиенты функции активации

	void InitializeWeights(); // инициализация весовых коэффициентов
public:	
	FullyConnectedLayer(int inputs, int outputs, const string &function); // создание слоя

	vector<double> Forward(const vector<double> &x); // прямое распространение
	vector<double> Backward(const vector<double> &x, const vector<double> &dout); // обратное распространение
	void UpdateWeights(double learningRate); // обновление весовых коэффициентов

	void PrintWeights() const; // вывод весовых коэффициентов
};

FullyConnectedLayer::FullyConnectedLayer(int inputs, int outputs, const string &function) : Layer(inputs, outputs) {
	this->function = function; // запоминаем функцию активации

	if (function != "sigmoid" && function != "tanh" && function != "relu")
		throw runtime_error("unknown function");

	// выделяем память под весовые коэффициенты и выходной вектор
	w = vector<vector<double>>(outputs, vector<double>(inputs));
	b = vector<double>(outputs);

	// выделяем место под градиенты
	dw = vector<vector<double>>(outputs, vector<double>(inputs));
	db = vector<double>(outputs);
	df = vector<double>(outputs, 0);

	InitializeWeights();
}

// инициализация весовых коэффициентов
void FullyConnectedLayer::InitializeWeights() {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			w[i][j] = GetRnd(-0.5, 0.5);

		b[i] = GetRnd(-0.5, 0.5); 
	}
}

// прямое распространение
vector<double> FullyConnectedLayer::Forward(const vector<double> &x) {	
	for (int i = 0; i < outputs; i++) {
		double y = b[i];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

		// применяем функцию активации
		if (function == "sigmoid") {
			output[i] = 1.0 / (exp(-y) + 1);
			df[i] = output[i] * (1 - output[i]);
		}
		else if (function == "tanh") {
			output[i] = tanh(y);
			df[i] = 1 - output[i] * output[i];
		}
		else if (function == "relu") {
			output[i] = y > 0 ? y : 0;
			df[i] = output[i] > 0 ? 1 : 0;
		}
	}

	return output; // возвращаем результирующий вектор
}

// обратное распространение
vector<double> FullyConnectedLayer::Backward(const vector<double> &x, const vector<double> &dout) {
	for (int i = 0; i < outputs; i++)
		df[i] *= dout[i]; // умножаем градиенты активации на выходные градиенты

	// считаем градиенты весовых коэффициентов
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			dw[i][j] = df[i] * x[j];

		db[i] = df[i];
	}

	// считаем градиенты по входам
	for (int i = 0; i < inputs; i++) {
		dx[i] = 0;

		for (int j = 0; j < outputs; j++)
			dx[i] += w[j][i] * df[j];
	}

	return dx; // возвращаем градиенты по входам
}

// обновление весовых коэффициентов
void FullyConnectedLayer::UpdateWeights(double learningRate) {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			w[i][j] -= learningRate * dw[i][j]; // выполняем шаг градиентного спуска для веса

		b[i] -= learningRate * db[i]; // выполняем шаг градиентного спуска для смещения
	}
}

// вывод весовых коэффициентов
void FullyConnectedLayer::PrintWeights() const {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			cout << w[i][j] << " ";

		cout << b[i] << endl;
	}
}