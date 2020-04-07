#pragma once

#include <iostream>
#include <vector>
#include "Tensor.hpp"

using namespace std;

typedef double (*LossFunction)(const Tensor &y, const Tensor &t, Tensor &dout); // указатель на функцию потерь

// средне квадратичное отклонение
double MSE(const Tensor &y, const Tensor &t, Tensor &dout) {
	double loss = 0;

	for (int i = 0; i < y.Total(); i++) {
		double e = y[i] - t[i]; // находим разность между элементами
		dout[i] = 2 * e; // вычисялем производную функции потерь
		loss += e * e; // добавляем значение функции потерь
	}

	return loss; // возвращаем значение функции потерь
}

// перекрёстная энтропия
double CrossEntropy(const Tensor &y, const Tensor &t, Tensor &dout) {
	double loss = 0;
	
	for (int i = 0; i < y.Total(); i++) {
		dout[i] = -t[i] / y[i]; // вычисялем производную функции потерь
		loss -= t[i] * log(y[i]); // добавляем значение функции потерь
	}

	return loss; // возвращаем значение функции потерь
}

// бинарная перекрёстная энтропия
double BinaryCrossEntropy(const Tensor &y, const Tensor &t, Tensor &dout) {
	double loss = 0;
	
	for (int i = 0; i < y.Total(); i++) {
		dout[i] = (y[i] - t[i]) / (y[i] * (1 - y[i])); // вычисялем производную функции потерь
		loss -= t[i] * log(y[i]) + (1 - t[i]) * log(1 - y[i]); // добавляем значение функции потерь
	}

	return loss; // возвращаем значение функции потерь
}