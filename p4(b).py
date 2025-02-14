import numpy as np
np.random.seed(0)

X =       [[1, 2, 3, 2.5],
           [2.0, 5.0, -1.0, 2.0],
           [-1.5, 2.7, 3.3, -0.8]]


class layer_dense:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weights = 0.1*np.random.randn(number_of_inputs, number_of_neurons)
        self.biases = np.zeros((1, number_of_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = layer_dense(4,5)
layer2 = layer_dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)