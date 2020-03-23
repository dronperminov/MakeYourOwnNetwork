#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class Network {
	int inputs; // число входов
	int outputs; // число выходов
	int last; // индекс последнего слоя
	vector<Layer> layers; // слои

public:
	Network(int inputs); // создание сети
	
	void AddLayer(int outputs, const string& function); // добавление слоя
	void Print() const; // вывод коэффициентов
	
	vector<double> Forward(const vector<double> &x); // прямое распространение
};

// создание сети
Network::Network(int inputs) {
	this->inputs = inputs; // запоминаем число входов
	this->outputs = inputs; // нет слоёв
	this->last = -1; // ещё нет слоёв
}

// добавление слоя
void Network::AddLayer(int outputs, const string& function) {
	layers.push_back(Layer(this->outputs, outputs, function)); // добавляем слой
	this->outputs = outputs; // обновляем число выходов сети
	last++; // увеличиваем индекс последнего слоя
}

// вывод коэффициентов
void Network::Print() const {
	for (int i = 0; i < layers.size(); i++) {
		cout << "layer " << i << ": " << endl;
		layers[i].PrintWeights();
	}
}

// прямое распространение
vector<double> Network::Forward(const vector<double> &x) {
	layers[0].Forward(x); // выполняем распространение в первом слое

	// распространяем сигналы по остальным слоям
	for (int i = 1; i < layers.size(); i++)
		layers[i].Forward(layers[i - 1].GetOutput());

	return layers[last].GetOutput(); // возвращаем выход последнего слоя
}