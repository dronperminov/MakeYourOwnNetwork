#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../utils/Tensor.hpp"

using namespace std;

class Layer {
protected:
	TensorSize inputSize; // количество входов
	TensorSize outputSize; // количество выходов
	
	Tensor output; // выходной вектор
	Tensor dx; // градиенты входов

	double GetRnd(double a, double b); // получение случайного числа
public:
	Layer(TensorSize inputSize, TensorSize outputSize);

	Tensor GetOutput() const; // получение выходов
	Tensor GetDx() const; // получение градиентов входов

	virtual void ForwardTrain(const Tensor &x); // прямое распространение (обучающий этап)
	virtual void Forward(const Tensor &x) = 0; // прямое распространение
	virtual void Backward(const Tensor &x, const Tensor &dout, bool needDx) = 0; // обратное распространение
	virtual void UpdateWeights(double learningRate); // обновление весовых коэффициентов

	virtual void PrintWeights() const; // вывод весовых коэффициентов
	virtual void Summary() const = 0; // вывод информации
};

Layer::Layer(TensorSize inputSize, TensorSize outputSize) : output(outputSize), dx(inputSize) {
	this->inputSize = inputSize; // запоминаем число входов
	this->outputSize = outputSize; // запоминаем число выходов
}

// получение случайного числа
double Layer::GetRnd(double a, double b) {
	return a + ((b - a) * rand()) / RAND_MAX;
}

// получение выходов
Tensor Layer::GetOutput() const {
	return output;
}

// получение градиентов входов
Tensor Layer::GetDx() const {
	return dx;
}

void Layer::ForwardTrain(const Tensor &x) {
	Forward(x);
}

// обновление весовых коэффициентов
void Layer::UpdateWeights(double learningRate) {
}

// вывод весовых коэффициентов
void Layer::PrintWeights() const {
}