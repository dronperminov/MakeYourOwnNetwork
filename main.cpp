#include <iostream>
#include "Network.hpp"

using namespace std;

// вывод вектора
void PrintVector(const vector<double> &v) {
	cout << "[ ";
	for (int i = 0; i < v.size(); i++)
		cout << v[i] << " ";
	cout << "]";
}

int main() {
	NetworkData trainData; // задаём данные для XOR проблемы
	trainData.x = {
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
	};

	trainData.y = {
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 },
	};

	double learningRate = 0.5; // скорость обучения
	int epochs = 10000; // число эпох

	Network network(2); // создаём сеть из двух входов
	network.AddLayer(5, "tanh"); // добавляем слой из 5 нейронов
	network.AddLayer(1, "sigmoid"); // добавляем слой из одного нейрона

	network.Train(trainData, learningRate, epochs, 1000); // обучаем

	// выводим результат обучения
	for (int i = 0; i < trainData.x.size(); i++) {
		vector<double> y = network.Forward(trainData.x[i]);

		cout << "x: ";
		PrintVector(trainData.x[i]);
		cout << "\tcorrect:";
		PrintVector(trainData.y[i]);
		cout << "\tnetwork: ";
		PrintVector(y);
		cout << endl;
	}
}