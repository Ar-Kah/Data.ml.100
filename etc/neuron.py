import numpy as np
import matplotlib.pyplot as plt



class NeuralNetwork:
    def __init__(self, learning_rate, dimentions):
        self.weights = np.empty(dimentions)
        for dimention in range(dimentions):
            self.weights[dimention] = np.random.randn()
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._relu(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._relu_deriv(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._relu_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )


    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):

            input_vector = input_vectors[current_iteration]
            target = targets[current_iteration]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

# input_vectors = np.array(
#     [
#         [3, 1.5],
#         [2, 1],
#         [4, 1.5],
#         [3, 4],
#         [3.5, 0.5],
#         [2, 0.5],
#         [5.5, 1],
#         [1, 1],
#     ]
# )

# targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

# learning_rate = 0.1

# neural_network = NeuralNetwork(learning_rate)

# training_error = neural_network.train(input_vectors, targets, 10000)

# plt.plot(training_error)
# plt.xlabel("Iterations")
# plt.ylabel("Error for all training instances")
# plt.show()
