#include <iostream>
#include <vector>

#include "../Network.hpp"
#include "../utils/DataLoader.hpp"

int Argmax(const vector<double> &x) {
	int imax = 0;

	for (int i = 1; i < x.size(); i++)
		if (x[i] > x[imax])
			imax = i;

	return imax;
}

double Test(Network &network, const NetworkData& data) {
	int correct = 0;
	double total = data.x.size();

	for (int i = 0; i < data.x.size(); i++) {
		vector<double> y = network.Forward(data.x[i]);

		int imax1 = Argmax(y);
		int imax2 = Argmax(data.y[i]);

		correct += imax1 == imax2;
	}
	
	return correct / total;
}

int main() {
	DataLoader loader("../datasets/mnist/mnist.txt");
	NetworkData trainData = loader.LoadData("../datasets/mnist/mnist_train.csv");
	NetworkData testData = loader.LoadData("../datasets/mnist/mnist_test.csv");

	Network network(784);
	network.AddLayer("fc 100");
	network.AddLayer("activation sigmoid");
	network.AddLayer("fc 10");
	network.AddLayer("softmax");

	double learningRate = 0.08;
	int epochs = 20;
	int testPeriod = 5;
	LossFunction loss = CrossEntropy;

	cout << "Init accuracy: " << Test(network, testData) << endl;

	for (int i = 0; i < epochs / testPeriod; i++) {
		network.Train(trainData, loss, learningRate, testPeriod, 1);
		
		cout << "Train accuracy: " << Test(network, trainData) << endl;
		cout << " Test accuracy: " << Test(network, testData) << endl;
	}
}