import numpy as np
import pygame


def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_error = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= weights_gradient * learning_rate
        self.bias -= output_gradient * learning_rate
        return input_error


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def reLu(x):
    return np.maximum(x, 0)


def reLu_prime(x):
    return x > 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


class NeuralNetwork:
    def __init__(self, sizes, activation_function, activation_function_prime):
        self.layers = []
        for index, size in enumerate(sizes):
            if index != len(sizes) - 1:
                self.layers.append(Dense(sizes[index], sizes[index + 1]))
                self.layers.append(Activation(activation_function, activation_function_prime))

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, input, actual, learning_rate):
        output = self.predict(input)
        weights_gradient = mse_prime(actual, output)
        for layer in reversed(self.layers):
            weights_gradient = layer.backward(weights_gradient, learning_rate)
        return mse(actual, output)


def draw_network(network, s, position, green, red, spacing_x=50, spacing_y=20):
    dense_layers = [lay for lay in network.layers if type(lay) == Dense]

    for x, lay in enumerate(dense_layers):
        out_size, in_size = lay.weights.shape
        x_start = x * spacing_x + position[0]
        x_end = x_start + spacing_x
        # Node y-positions depend only on the node index, so compute them once per layer
        ys_in = ((np.arange(in_size) - in_size / 2) * spacing_y + position[1]).tolist()
        ys_out = ((np.arange(out_size) - out_size / 2) * spacing_y + position[1]).tolist()

        # Line widths for the whole weight matrix at once
        positive = (lay.weights > 0).tolist()
        green_widths = np.ceil(lay.weights).astype(int).tolist()
        red_widths = (-np.floor(lay.weights)).astype(int).tolist()

        for y in range(in_size):
            start_pos = (x_start, ys_in[y])
            for z in range(out_size):
                end_pos = (x_end, ys_out[z])
                if positive[z][y]:
                    pygame.draw.line(s, green, start_pos, end_pos, green_widths[z][y])
                else:
                    pygame.draw.line(s, red, start_pos, end_pos, red_widths[z][y])

        for w, bias in enumerate(lay.bias[:, 0].tolist()):
            center = (x_end, ys_out[w])
            if bias > 0:
                pygame.draw.circle(s, green, center, int(np.ceil(bias)) * 2)
            else:
                pygame.draw.circle(s, red, center, int(np.floor(bias)) * -2)


nn = NeuralNetwork([2, 3, 3, 1], tanh, tanh_prime)
