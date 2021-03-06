#pragma once

#include <vector>
#include "Image.hpp"

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
    Tensor(TensorSize size);
    Tensor(int width, int height, int depth);

    double& operator[](int index);
    const double& operator[](int index) const;

    double& operator()(int i, int j, int d);
    const double& operator()(int i, int j, int d) const;

    int Total() const;
    int Argmax() const;

    void SaveAsImage(const string& name);
};

Tensor::Tensor(int size) {
    this->size.width = 1;
    this->size.height = 1;
    this->size.depth = size;
    
    values = vector<double>(size, 0);
}

Tensor::Tensor(TensorSize size) {
    this->size.width = size.width;
    this->size.height = size.height;
    this->size.depth = size.depth;

    values = vector<double>(size.width * size.height * size.depth, 0);
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

ostream& operator<<(ostream& os, const TensorSize& size) {
    return os << (to_string(size.width) + "x" + to_string(size.height) + "x" + to_string(size.depth));
}

void Tensor::SaveAsImage(const string& name) {
    if (size.depth != 1)
        throw runtime_error("only wb images now");

    Image image(size.width, size.height);

    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            double value = (*this)(i, j, 0);
            uint8_t v = min(255, max(0, (int) (value * 255)));
            image.SetPixel(j, i, v, v, v);
        }
    }

    image.Save(name);
}