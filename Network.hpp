#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "utils/LossFunction.hpp"
#include "Layers/Layer.hpp"
#include "Layers/FullyConnectedLayer.hpp"
#include "Layers/ActivationLayer.hpp"
#include "Layers/SoftmaxLayer.hpp"

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
	void UpdateWeights(double learningRate); // обновление весовых коэффициентов

public:
	Network(int inputs); // создание сети
	
	void AddLayer(const string &config); // добавление слоя
	void Print() const; // вывод коэффициентов
	void Train(const NetworkData &data, LossFunction L, double learningRate, int batchSize, int epochs, int log_period); // обучение
	
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
	else if (type == "softmax") {
		layers.push_back(new SoftmaxLayer(this->outputs, this->outputs));
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
void Network::Train(const NetworkData& data, LossFunction L, double learningRate, int batchSize, int epochs, int log_period) {

	for (int epoch = 0; epoch < epochs; epoch++) {
		double loss = 0;

		for (int i = 0; i < data.x.size(); i += batchSize) {
			for (int j = 0; j < batchSize && i + j < data.x.size(); j++) {
				vector<double> out = Forward(data.x[i + j]); // выполняем прямое распространение
				vector<double> dout(outputs); // создаём вектор производных функции потерь
				loss += L(out, data.y[i + j], dout); // вычисляем функцию потерь
				Backward(data.x[i + j], dout); // выполняем обратное распространение
			}

			UpdateWeights(learningRate / batchSize); // обновляем весовые коэффициенты
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