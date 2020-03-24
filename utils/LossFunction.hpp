#pragma once

#include <iostream>
#include <vector>

using namespace std;

typedef double (*LossFunction)(const vector<double> &y, const vector<double> &t, vector<double> &dout); // указатель на функцию потерь

// средне квадратичное отклонение
double MSE(const vector<double> &y, const vector<double> &t, vector<double> &dout) {
	double loss = 0;

	for (int i = 0; i < y.size(); i++) {
		double e = y[i] - t[i]; // находим разность между элементами
		dout[i] = 2 * e; // вычисялем производную функции потерь
		loss += e * e; // добавляем значение функции потерь
	}

	return loss; // возвращаем значение функции потерь
}

// перекрёстная энтропия
double CrossEntropy(const vector<double> &y, const vector<double> &t, vector<double> &dout) {
	double loss = 0;
	
	for (int i = 0; i < y.size(); i++) {
		dout[i] = -t[i] / y[i]; // вычисялем производную функции потерь
		loss -= t[i] * log(y[i]); // добавляем значение функции потерь
	}

	return loss; // возвращаем значение функции потерь
}

// бинарная перекрёстная энтропия
double BinaryCrossEntropy(const vector<double> &y, const vector<double> &t, vector<double> &dout) {
	double loss = 0;
	
	for (int i = 0; i < y.size(); i++) {
		dout[i] = (y[i] - t[i]) / (y[i] * (1 - y[i])); // вычисялем производную функции потерь
		loss -= t[i] * log(y[i]) + (1 - t[i]) * log(1 - y[i]); // добавляем значение функции потерь
	}

	return loss; // возвращаем значение функции потерь
}