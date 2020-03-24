#include <iostream>
#include <cmath>
#include "../Network.hpp"

using namespace std;

// вывод вектора
void PrintVector(const vector<double> &v) {
	cout << "[ ";
	for (int i = 0; i < v.size(); i++)
		cout << v[i] << " ";
	cout << "]";
}

// создание данных - точки внутри окружности радиуса 1
NetworkData InitData(int n) {
	NetworkData data;

	for (int i = 0; i < n; i++) {
		vector<double> point(2);
		vector<double> label(4, 0);

		double phi = rand() * M_PI * 2 / RAND_MAX;
		double r = rand() * 1.0 / RAND_MAX;

		point[0] = r * cos(phi);
		point[1] = r * sin(phi);

		if (phi < M_PI / 2)
			label[0] = 1;
		else if (phi < M_PI)
			label[1] = 1;
		else if (phi < 3 * M_PI / 2)
			label[2] = 1;
		else
			label[3] = 1;

		data.x.push_back(point);
		data.y.push_back(label);
	}

	return data;
}

// поиск аргумента с максимальным значением
int Argmax(const vector<double> &v) {
	int imax = 0;

	for (int i = 1; i < v.size(); i++)
		if (v[i] > v[imax])
			imax = i;

	return imax;
}

int main() {
	double learningRate = 0.5; // скорость обучения
	int batchSize = 1; // размер батча
	int epochs = 1000; // число эпох
	int n = 100;
	NetworkData trainData = InitData(n); // задаём обучающие данные для классификации
	NetworkData testData = InitData(n); // задаём проверочные данные для классификации

	Network network(2); // создаём сеть из двух входов
	network.AddLayer("fc 5"); // добавляем слой из 5 нейронов
	network.AddLayer("activation tanh"); // добавляем слой активации
	network.AddLayer("fc 4"); // добавляем слой из четырёх нейронов
	// network.AddLayer("activation sigmoid"); // добавляем слой активации
	network.AddLayer("softmax"); // добавляем softmax слой

	network.Train(trainData, MSE, learningRate, batchSize, epochs, 100); // обучаем

	double correctTest = 0;
	double correctTrain = 0;

	// измеряем точность классификации
	for (int i = 0; i < n; i++) {
		vector<double> y = network.Forward(testData.x[i]);

		int imax1 = Argmax(y);
		int imax2 = Argmax(testData.y[i]);

		if (imax1 == imax2) {
			correctTest++;
		}
		else {
			cout << "x: ";
			PrintVector(testData.x[i]);
			cout << "y: ";
			PrintVector(y);
			cout << "\tt:";
			PrintVector(testData.y[i]);
			cout << endl;
		}

		y = network.Forward(trainData.x[i]);

		imax1 = Argmax(y);
		imax2 = Argmax(trainData.y[i]);

		if (imax1 == imax2)
			correctTrain++;
	}

	cout << "train accuracy: " << correctTrain / n << endl;
	cout << " test accuracy: " << correctTest / n << endl;
}