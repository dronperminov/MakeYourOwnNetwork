#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "Layers/Layer.hpp"
#include "Layers/FullyConnectedLayer.hpp"
#include "Layers/ActivationLayer.hpp"

using namespace std;

struct NetworkData {
	vector<vector<double>> x;
	vector<vector<double>> y;
};

class Network {
	int inputs; // число входов
	int outputs; // число выходов
	int last; // индекс последнего слоя
	vector<Layer*> layers; // слои

	void Backward(const vector<double> &x, const vector<double> &dout); // обратное распространение ошибки
	double CalculateLoss(const vector<double> &y, const vector<double> &t, vector<double> &dout); // вычисление ошибки
	void UpdateWeights(double learningRate); // обновление весовых коэффициентов

public:
	Network(int inputs); // создание сети
	
	void AddLayer(const string &config); // добавление слоя
	void Print() const; // вывод коэффициентов
	void Train(const NetworkData &data, double learningRate, int epochs, int log_period); // обучение
	
	vector<double> Forward(const vector<double> &x); // прямое распространение
};

// создание сети
Network::Network(int inputs) {
	this->inputs = inputs; // запоминаем число входов
	this->outputs = inputs; // нет слоёв
	this->last = -1; // ещё нет слоёв
}

// обратное распространение ошибки
void Network::Backward(const vector<double> &x, const vector<double> &dout) {
	if (last == 0) {
		layers[last]->Backward(x, dout);
		return;
	}

	layers[last]->Backward(layers[last - 1]->GetOutput(), dout); // обрабатываем последний слой

	for (int i = last - 1; i >= 1; i--)
		layers[i]->Backward(layers[i - 1]->GetOutput(), layers[i + 1]->GetDx()); // обрабатываем промежуточные слои

	layers[0]->Backward(x, layers[1]->GetDx()); // обрабатываем первый слой
}

// вычисление ошибки
double Network::CalculateLoss(const vector<double> &y, const vector<double> &t, vector<double> &dout) {
	double loss = 0;

	for (int i = 0; i < outputs; i++) {
		double e = y[i] - t[i]; // находим разность между элементами
		dout[i] = 2 * e; // записываем производную функции ошибки
		loss += e * e; // добавляем ошибку
	}

	return loss; // возвращаем ошибку
}

// обновление весовых коэффициентов
void Network::UpdateWeights(double learningRate) {
	for (int i = 0; i < layers.size(); i++)
		layers[i]->UpdateWeights(learningRate); // обновляем веса у каждого из слоёв
}

// добавление слоя
void Network::AddLayer(const string &config) {
	stringstream ss(config);
	string type;
	ss >> type;

	if (type == "activation") {
		string function;
		ss >> function;
		layers.push_back(new ActivationLayer(this->outputs, this->outputs, function)); // добавляем слой
	}
	else if (type == "fc") {
		int outputs;
		ss >> outputs;

		layers.push_back(new FullyConnectedLayer(this->outputs, outputs)); // добавляем слой
		this->outputs = outputs; // обновляем число выходов сети
	}

	last++; // увеличиваем индекс последнего слоя
}

// вывод коэффициентов
void Network::Print() const {
	for (int i = 0; i < layers.size(); i++) {
		cout << "layer " << i << ": " << endl;
		layers[i]->PrintWeights();
	}
}

// обучение сети
void Network::Train(const NetworkData& data, double learningRate, int epochs, int log_period) {

	for (int epoch = 0; epoch < epochs; epoch++) {
		double loss = 0;

		for (int i = 0; i < data.x.size(); i++) {
			vector<double> out = Forward(data.x[i]); // выполняем прямое распространение
			vector<double> dout(outputs); // создаём вектор производных функции потерь
			loss += CalculateLoss(out, data.y[i], dout); // вычисляем функцию потерь
			Backward(data.x[i], dout); // выполняем обратное распространение
			UpdateWeights(learningRate); // обновляем весовые коэффициенты
		}

		// если нужно
		if (epoch % log_period == 0 || epoch == epochs - 1)
			cout << "Epoch: " << epoch << ", loss: " << loss << endl; // выводим текущую эпоху и ошибку
	}
}

// прямое распространение
vector<double> Network::Forward(const vector<double> &x) {
	layers[0]->Forward(x); // выполняем распространение в первом слое

	// распространяем сигналы по остальным слоям
	for (int i = 1; i < layers.size(); i++)
		layers[i]->Forward(layers[i - 1]->GetOutput());

	return layers[last]->GetOutput(); // возвращаем выход последнего слоя
}