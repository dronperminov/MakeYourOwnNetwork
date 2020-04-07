#pragma once

#include <vector>

using namespace std;

struct TensorSize {
    int width;
    int height;
    int depth;
};

class Tensor {
    TensorSize size;
    vector<double> values;
public:
    Tensor(int size);
    Tensor(int width, int height, int depth);

    double& operator[](int index);
    const double& operator[](int index) const;

    double& operator()(int i, int j, int d);
    const double& operator()(int i, int j, int d) const;

    int Total() const;
    int Argmax() const;
};

Tensor::Tensor(int size) {
    this->size.width = 1;
    this->size.height = 1;
    this->size.depth = size;
    
    values = vector<double>(size, 0);
}

Tensor::Tensor(int width, int height, int depth) {
    size.width = width;
    size.height = height;
    size.depth = depth;

    values = vector<double>(width * height * depth, 0);
}

double& Tensor::operator[](int index) {
    return values[index];
}

const double& Tensor::operator[](int index) const {
    return values[index];
}

double& Tensor::operator()(int i, int j, int d) {
    return values[(i * size.width + j) * size.depth + d];
}

const double& Tensor::operator()(int i, int j, int d) const {
    return values[(i * size.width + j) * size.depth + d];
}

int Tensor::Argmax() const {
    int imax = 0;

    for (int i = 1; i < values.size(); i++)
        if (values[i] > values[imax])
            imax = i;

    return imax;
}

int Tensor::Total() const {
    return values.size();
}