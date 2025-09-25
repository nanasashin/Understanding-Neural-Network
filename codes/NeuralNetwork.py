import numpy as np

test_inputs = [
    [10, 20, 6, 5],
    [6, 8, 9, 10]
]

class LayerDense:
    def __init__(self, n_input, n_neurons):
        self.weights = 0.10 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class ActivationReLu:
    def forward(self, inputs):
        return np.maximum(0, inputs)

print(ActivationReLu().forward(LayerDense(4, 8).forward(test_inputs)))

