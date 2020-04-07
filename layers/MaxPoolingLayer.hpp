#pragma once

#include "Layer.hpp"

using namespace std;

class MaxPoolingLayer : public Layer {
    int scale;
public: 
    MaxPoolingLayer(TensorSize inputSize, int scale); // создание слоя

    void Forward(const Tensor &x); // прямое распространение
    void Backward(const Tensor &x, const Tensor &dout, bool needDx); // обратное распространение

    void Summary() const; // вывод информации   
};

MaxPoolingLayer::MaxPoolingLayer(TensorSize inputSize, int scale) : Layer(inputSize, { inputSize.width / scale, inputSize.height / scale, inputSize.depth }) {
    this->scale = scale;
}

void MaxPoolingLayer::Forward(const Tensor& x) {
    for (int d = 0; d < inputSize.depth; d++) {
        for (int i = 0; i < inputSize.height; i += scale) {
            for (int j = 0; j < inputSize.width; j += scale) {
                int imax = i;
                int jmax = j;
                double max = x(i, j, d);

                for (int ii = i; ii < i + scale; ii++) {
                    for (int jj = j; jj < j + scale; jj++) {
                        double value = x(ii, jj, d);
                        dx(ii, jj, d) = 0;

                        if (value > max) {
                            max = value;
                            imax = ii;
                            jmax = jj;
                        }
                    }
                }

                output(i / scale, j / scale, d) = max;
                dx(imax, jmax, d) = 1;
            }
        }
    }
}

void MaxPoolingLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
    if (!needDx)
        return;

    for (int d = 0; d < inputSize.depth; d++)
        for (int i = 0; i < inputSize.height; i++)
            for (int j = 0; j < inputSize.width; j++)
                dx(i, j, d) *= dout(i / scale, j / scale, d);
}

// вывод информации
void MaxPoolingLayer::Summary() const {
    cout << "|           maxpooling | "  << setw(12) << inputSize << " | " << setw(13) << outputSize << " | " << setw(13) << (0) << " | scale: " << scale << endl;
}
