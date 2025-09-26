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
    

class AticationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clippped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_pred.shape) == 1:
            correct_confidence = y_pred_clippped[range(samples), y_true]

        elif len(y_pred.shape) == 2:
            correct_confidence = np.sum(y_pred_clippped*y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood


first_layer = ActivationReLu().forward(LayerDense(4, 8).forward(test_inputs))
second_layer = AticationSoftmax().forward(first_layer)

calculation_loss = Loss_CategoricalCrossEntropy().calculate(second_layer, [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0]])

print(calculation_loss)