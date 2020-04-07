#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#include "utils/LossFunction.hpp"
#include "Layers/Layer.hpp"
#include "Layers/FullyConnectedLayer.hpp"
#include "Layers/DropoutLayer.hpp"
#include "Layers/ActivationLayer.hpp"
#include "Layers/SoftmaxLayer.hpp"

using namespace std;

typedef chrono::high_resolution_clock Time;
typedef chrono::time_point<Time> TimePoint;
typedef chrono::milliseconds ms;

struct NetworkData {
	vector<Tensor> x;
	vector<Tensor> y;
};

class Network {
	TensorSize inputSize; // входной размер
	TensorSize outputSize; // выходной размер
	int last; // индекс последнего слоя
	vector<Layer*> layers; // слои

	Tensor ForwardTrain(const Tensor &x); // прямое распространение
	void Backward(const Tensor &x, const Tensor &dout); // обратное распространение ошибки
	void UpdateWeights(double learningRate); // обновление весовых коэффициентов

public:
	Network(int width, int height, int depth); // создание сети
	
	void AddLayer(const string &config); // добавление слоя
	void Print() const; // вывод коэффициентов
	void Train(const NetworkData &data, LossFunction L, double learningRate, int batchSize, int epochs, int log_period); // обучение
	
	Tensor Forward(const Tensor &x); // прямое распространение
	void Summary() const;
};

// создание сети
Network::Network(int width, int height, int depth) {
	this->inputSize.width = width;
	this->inputSize.height = height;
	this->inputSize.depth = depth;
	this->outputSize = inputSize; // нет слоёв
	this->last = -1; // ещё нет слоёв
}

// прямое распространение
Tensor Network::ForwardTrain(const Tensor &x) {
	layers[0]->ForwardTrain(x); // выполняем распространение в первом слое

	// распространяем сигналы по остальным слоям
	for (int i = 1; i < layers.size(); i++)
		layers[i]->ForwardTrain(layers[i - 1]->GetOutput());

	return layers[last]->GetOutput(); // возвращаем выход последнего слоя
}

// обратное распространение ошибки
void Network::Backward(const Tensor &x, const Tensor &dout) {
	if (last == 0) {
		layers[last]->Backward(x, dout, false);
		return;
	}

	layers[last]->Backward(layers[last - 1]->GetOutput(), dout, true); // обрабатываем последний слой

	for (int i = last - 1; i >= 1; i--)
		layers[i]->Backward(layers[i - 1]->GetOutput(), layers[i + 1]->GetDx(), true); // обрабатываем промежуточные слои

	layers[0]->Backward(x, layers[1]->GetDx(), false); // обрабатываем первый слой
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
		layers.push_back(new ActivationLayer(this->outputSize, function)); // добавляем слой
	}
	else if (type == "fc") {
		int outputs;
		ss >> outputs;

		layers.push_back(new FullyConnectedLayer(outputSize, outputs)); // добавляем слой
		outputSize.width = 1;
		outputSize.height = 1;
		outputSize.depth = outputs; // обновляем число выходов сети
	}
	else if (type == "softmax") {
		layers.push_back(new SoftmaxLayer(this->outputSize));
	}
	else if (type == "dropout") {
		double p;
		ss >> p;
		layers.push_back(new DropoutLayer(this->outputSize, p));
	}
	else
		throw runtime_error("unknown layer '" + type + "'");

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
		TimePoint t0 = Time::now();

		for (int i = 0; i < data.x.size(); i += batchSize) {
			for (int j = 0; j < batchSize && i + j < data.x.size(); j++) {
				Tensor out = ForwardTrain(data.x[i + j]); // выполняем прямое распространение
				Tensor dout(outputSize); // создаём вектор производных функции потерь
				loss += L(out, data.y[i + j], dout); // вычисляем функцию потерь
				Backward(data.x[i + j], dout); // выполняем обратное распространение
			}

			UpdateWeights(learningRate / batchSize); // обновляем весовые коэффициенты
		}

		ms d = std::chrono::duration_cast<ms>(Time::now() - t0);

		// если нужно
		if (epoch % log_period == 0 || epoch == epochs - 1)
			cout << "Epoch: " << epoch << ", loss: " << loss << ", epoch time: " << (d.count() / 1000.0) << " s" << endl; // выводим текущую эпоху и ошибку
	}
}

// прямое распространение
Tensor Network::Forward(const Tensor &x) {
	layers[0]->Forward(x); // выполняем распространение в первом слое

	// распространяем сигналы по остальным слоям
	for (int i = 1; i < layers.size(); i++)
		layers[i]->Forward(layers[i - 1]->GetOutput());

	return layers[last]->GetOutput(); // возвращаем выход последнего слоя
}

void Network::Summary() const {
	cout << "+----------------------+--------------+---------------+---------------+" << endl;
	cout << "|      layer name      | inputs count | outputs count | weights count |" << endl;
	cout << "+----------------------+--------------+---------------+---------------+" << endl;

	for (int i = 0; i < layers.size(); i++)
		layers[i]->Summary();

	cout << "+----------------------+--------------+---------------+---------------+" << endl;
}