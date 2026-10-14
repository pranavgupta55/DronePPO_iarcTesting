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


LIVE_GREEN, LIVE_RED, LIVE_NODE_EDGE = (96, 224, 118), (244, 84, 96), (110, 114, 130)
LIVE_REF = 1.5  # a connection signal (weight times the source node's activation) of this size gets the thickest, brightest line
LIVE_MAX_WIDTH = 7


def live_layout(sizes, spacing_x, spacing_y, margin=3):
    """(width, height, x of each layer's nodes, center y) of the surface a live network with these layer sizes needs. The surface is cropped tight: the first layer's nodes and
    the tallest layer's top node have their outer pixels on the surface's left and top edge (margin = node radius), so placing the surface at (m, m) puts the drawing exactly
    m px from the screen's left and top. Thick lines that start on those edges are clipped there."""
    height = int(round((max(sizes) - 1) * spacing_y + 2 * margin))
    xs = [margin + k * spacing_x for k in range(len(sizes))]
    return xs[-1] + margin + 2, height, xs, height / 2


def draw_live_network(surface, layers, activations, spacing_x=66, spacing_y=5.5, margin=3):
    """The network as it computes right now. layers: [(weight [out, in], bias), ...]; activations: the vectors that went through it this frame (the normalized input,
    each hidden layer's output, the action), as NetworkPolicy.trace returns them. A connection is drawn with the signal it carries, weight x the source node's activation:
    green for positive, red for negative, thicker and brighter the stronger it is. Nodes are filled by their own activation. surface should be a cleared SRCALPHA surface."""
    sizes = [len(activations[0])] + [len(a) for a in activations[1:]]
    _, _, xs, center_y = live_layout(sizes, spacing_x, spacing_y, margin)
    ys = [[center_y + (i - (n - 1) / 2) * spacing_y for i in range(n)] for n in sizes]
    for k, (weight, _) in enumerate(layers):
        signal = np.asarray(weight) * np.asarray(activations[k])[None, :]  # [out, in]: what each connection carries
        strength = np.minimum(np.abs(signal) / LIVE_REF, 1.0)
        for z in range(signal.shape[0]):
            for y in range(signal.shape[1]):
                s = float(strength[z, y])
                if s < 0.03:
                    continue
                base = LIVE_GREEN if signal[z, y] > 0 else LIVE_RED
                pygame.draw.line(surface, (*base, int(70 + 185 * s)), (xs[k], ys[k][y]), (xs[k + 1], ys[k + 1][z]), 1 + int(round((LIVE_MAX_WIDTH - 1) * s)))
    for k, act in enumerate(activations):
        scale = 1.0 if k < len(activations) - 1 else max(1.0, float(np.max(np.abs(act))))  # the action layer is linear, so it is scaled to its own largest value
        for i, a in enumerate(np.asarray(act).tolist()):
            level = min(1.0, abs(a) / (scale if k else 2.0))
            fill = (*(LIVE_GREEN if a > 0 else LIVE_RED), int(40 + 215 * level))
            pygame.draw.circle(surface, fill, (xs[k], ys[k][i]), 3 + (1 if k == len(activations) - 1 else 0))
            pygame.draw.circle(surface, LIVE_NODE_EDGE, (xs[k], ys[k][i]), 3 + (1 if k == len(activations) - 1 else 0), 1)
