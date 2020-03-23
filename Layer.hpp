#pragma once
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Layer {
	int inputs; // количество входов
	int outputs; // количество выходов
	string function; // тип активационной функции
	vector<vector<double>> w; // матрица весовых коэффициентов
	vector<double> b; // вектор весов смещения
	vector<double> output; // выходной вектор

	vector<vector<double>> dw; // градиенты весовых коэффициентов
	vector<double> db; // градиенты весов смещения
	vector<double> df; // градиенты функции активации
	vector<double> dx; // градиенты входов

	double GetRnd(double a, double b); // получение случайного числа
	void InitializeWeights(); // инициализация весовых коэффициентов
public:	
	Layer(int inputs, int outputs, const string &function); // создание слоя

	vector<double> GetOutput() const; // получение выходов
	vector<double> GetDx() const; // получение градиентов входов

	vector<double> Forward(const vector<double> &x); // прямое распространение
	vector<double> Backward(const vector<double> &x, const vector<double> &dout); // обратное распространение
	void UpdateWeights(double learningRate); // обновление весовых коэффициентов

	void PrintWeights() const; // вывод весовых коэффициентов
};

Layer::Layer(int inputs, int outputs, const string &function) {
	this->inputs = inputs; // запоминаем число входов
	this->outputs = outputs; // запоминаем число выходов
	this->function = function; // запоминаем функцию активации

	if (function != "sigmoid" && function != "tanh" && function != "relu")
		throw runtime_error("unknown function");

	// выделяем память под весовые коэффициенты и выходной вектор
	w = vector<vector<double>>(outputs, vector<double>(inputs));
	b = vector<double>(outputs);
	output = vector<double>(outputs, 0);

	// выделяем место под градиенты
	dw = vector<vector<double>>(outputs, vector<double>(inputs));
	db = vector<double>(outputs);
	df = vector<double>(outputs, 0);
	dx = vector<double>(inputs, 0);

	InitializeWeights();
}

// получение случайного числа
double Layer::GetRnd(double a, double b) {
	return a + ((b - a) * rand()) / RAND_MAX;
}

// инициализация весовых коэффициентов
void Layer::InitializeWeights() {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			w[i][j] = GetRnd(-0.5, 0.5);

		b[i] = GetRnd(-0.5, 0.5); 
	}
}

// получение выходов
vector<double> Layer::GetOutput() const {
	return output;
}

// получение градиентов входов
vector<double> Layer::GetDx() const {
	return dx;
}

// прямое распространение
vector<double> Layer::Forward(const vector<double> &x) {	
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
vector<double> Layer::Backward(const vector<double> &x, const vector<double> &dout) {
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
void Layer::UpdateWeights(double learningRate) {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			w[i][j] -= learningRate * dw[i][j]; // выполняем шаг градиентного спуска для веса

		b[i] -= learningRate * db[i]; // выполняем шаг градиентного спуска для смещения
	}
}

// вывод весовых коэффициентов
void Layer::PrintWeights() const {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			cout << w[i][j] << " ";

		cout << b[i] << endl;
	}
}