#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

class Layer {
protected:
	int inputs; // количество входов
	int outputs; // количество выходов
	
	vector<double> output; // выходной вектор
	vector<double> dx; // градиенты входов

	double GetRnd(double a, double b); // получение случайного числа
public:
	Layer(int inputs, int outputs);

	vector<double> GetOutput() const; // получение выходов
	vector<double> GetDx() const; // получение градиентов входов

	virtual vector<double> ForwardTrain(const vector<double> &x); // прямое распространение (обучающий этап)
	virtual vector<double> Forward(const vector<double> &x) = 0; // прямое распространение
	virtual vector<double> Backward(const vector<double> &x, const vector<double> &dout) = 0; // обратное распространение
	virtual void UpdateWeights(double learningRate); // обновление весовых коэффициентов

	virtual void PrintWeights() const; // вывод весовых коэффициентов
	virtual void Summary() const = 0; // вывод информации
};

Layer::Layer(int inputs, int outputs) {
	this->inputs = inputs; // запоминаем число входов
	this->outputs = outputs; // запоминаем число выходов

	output = vector<double>(outputs, 0); // выделяем память под выходной вектор
	dx = vector<double>(inputs, 0); // выделяем место под градиенты
}

// получение случайного числа
double Layer::GetRnd(double a, double b) {
	return a + ((b - a) * rand()) / RAND_MAX;
}

// получение выходов
vector<double> Layer::GetOutput() const {
	return output;
}

// получение градиентов входов
vector<double> Layer::GetDx() const {
	return dx;
}

vector<double> Layer::ForwardTrain(const vector<double> &x) {
	return Forward(x);
}

// обновление весовых коэффициентов
void Layer::UpdateWeights(double learningRate) {
}

// вывод весовых коэффициентов
void Layer::PrintWeights() const {
}