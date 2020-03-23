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

	double GetRnd(double a, double b); // получение случайного числа
	void InitializeWeights(); // инициализация весовых коэффициентов
public:	
	Layer(int inputs, int outputs, const string &function); // создание слоя

	vector<double> Forward(const vector<double> &x);
	void PrintWeights() const; // вывод весовых коэффициентов
};

Layer::Layer(int inputs, int outputs, const string &function) {
	this->inputs = inputs;
	this->outputs = outputs;
	this->function = function;

	if (function != "sigmoid" && function != "tanh" && function != "relu")
		throw runtime_error("unknown function");

	w = vector<vector<double>>(outputs, vector<double>(inputs));
	b = vector<double>(outputs);
	output = vector<double>(outputs, 0);

	InitializeWeights();
}

double Layer::GetRnd(double a, double b) {
	return a + ((b - a) * rand()) / RAND_MAX;
}

void Layer::InitializeWeights() {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			w[i][j] = GetRnd(-0.5, 0.5);

		b[i] = GetRnd(-0.5, 0.5); 
	}
}

vector<double> Layer::Forward(const vector<double> &x) {	
	for (int i = 0; i < outputs; i++) {
		double y = b[i];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

		if (function == "sigmoid") {
			output[i] = 1.0 / (exp(-y) + 1);
		}
		else if (function == "tanh") {
			output[i] = tanh(y);
		}
		else if (function == "relu") {
			output[i] = y > 0 ? y : 0;
		}
	}

	return output;
}

// вывод весовых коэффициентов
void Layer::PrintWeights() const {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			cout << w[i][j] << " ";

		cout << b[i] << endl;
	}
}