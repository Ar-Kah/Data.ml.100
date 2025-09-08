import numpy as np

# Me trying to implement a neural network from scratch
# this neural network will be for regression problems

if __name__ == '__main__':

    class NeuralNetwork:

        def __init__(self, learning_rate):
            self.bias = -1
            self.weights = np.array([np.random.randn(), np.random.randn()])
            self.learning_rate = learning_rate

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def _sigmoid_deriv(self, x):
            return self.sigmoid(x) * (1 - self.sigmoid(x))

        def predict(self, input_vector):
            layer1 = np.dot(input_vector, self.weights) + self.bias
            layer2 = self.sigmoid(layer1)
            prediction = layer2
            return prediction

        def compute_gradinats(self, input_vector, target):
            layer1 = np.dot(input_vector, self.weights) + self.bias
            layer2 = self.sigmoid(layer1)
            prediction = layer2

            derror_dprediction = 2 * (prediction - target)
            dprediction_dlayer1 = self._sigmoid_deriv(layer1)
            dlayer1_bias = 1
            dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

            derror_bias = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_bias
            )
            derror_dweights = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
            )

            return derror_bias, derror_dweights

        def _update_parameters(self, derror_dbias, derror_dweights):
            self.bias = self.bias - (derror_dbias * self.learning_rate)
            self.weights = self.weights - (derror_dweights * self.learning_rate)

    learning_rate = 0.1

    neural_network = NeuralNetwork(learning_rate)

    print(neural_network.predict([1.72, 1.23]))
