#include <iostream>
#include <vector>
#include <chrono>

#include "../Network.hpp"
#include "../utils/DataLoader.hpp"

double Test(Network &network, const NetworkData& data) {
	int correct = 0;
	double total = data.x.size();

	for (int i = 0; i < data.x.size(); i++) {
		Tensor y = network.Forward(data.x[i]);
		correct += y.Argmax() == data.y[i].Argmax();
	}
	
	return correct / total;
}

int main() {
	DataLoader loader("../datasets/mnist/mnist.txt");
	NetworkData trainData = loader.LoadData("../datasets/mnist/mnist_train.csv");
	NetworkData testData = loader.LoadData("../datasets/mnist/mnist_test.csv");

	Network network(28, 28, 1);
	network.AddLayer("fc 100");
	network.AddLayer("activation sigmoid");
	network.AddLayer("dropout 0.2");
	network.AddLayer("fc 10");
	network.AddLayer("softmax");

	cout << endl << "Configuration of network:" << endl;
	network.Summary();

	double learningRate = 0.08;
	int batchSize = 8;
	int epochs = 100;
	int testPeriod = 5;
	LossFunction loss = CrossEntropy;

	for (int i = 0; i < epochs / testPeriod; i++) {
		network.Train(trainData, loss, learningRate, batchSize, testPeriod, 1);

		double trainAcc = Test(network, trainData);
		double testAcc = Test(network, testData);

		cout << "train acc: " << trainAcc << ", test acc: " << testAcc << endl;
	}
}