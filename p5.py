import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X =       [[1, 2, 3, 2.5],
           [2.0, 5.0, -1.0, 2.0],     
           [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

class layer_dense:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weights = 0.1*np.random.randn(number_of_inputs, number_of_neurons)
        self.biases = np.zeros((1, number_of_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = layer_dense(2,5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(layer1.output)
print(activation1.output)
