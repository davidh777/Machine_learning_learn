import tensorflow as tf
import random as random
import math as math

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

input_data = ["" for _ in range(28 * 28)]
a = 0
for i in range(len(x_train[0])):
    for j in range(len(x_train[0][0])):
        input_data[a] = x_train[0][i][j]
        a += 1
input_labels = ["" for _ in range(10)]
for i in range(10):
    if i == y_train[0] - 1:
        input_labels[i] = 1
    else:
        input_labels[i] = 0


class Network(object):
    def __init__(self, number_layers):
        self.weights = ["" for _ in range(number_layers)]
        self.biases = ["" for _ in range(number_layers)]
        self.layer_info = ["" for _ in range(number_layers)]
        self.activations = ["" for _ in range(number_layers)]


    def relu(self, x):
        if x > 0:
            return x
        if x <= 0:
            return 0

    def sigmoid(self, x):
        if math.fabs(x) <= 700:
            return 1 / (1 + math.e ** (-1 * x))
        if x >= 700:
            return 1
        if x <= -700:
            return 0

    def layer_dense(self, input_values, neurons, activation_fun, layer_number):
        w = [[random.random() * random.randint(-1, 1) for _ in range(len(input_values))] for _ in range(neurons)]
        b = [random.random() * random.randint(-1, 1) for _ in range(neurons)]
        self.weights[layer_number] = w
        self.biases[layer_number] = b
        self.layer_info[layer_number] = [activation_fun]

        a = ["" for _ in range(neurons)]
        c = ["" for _ in range(neurons)]

        for j in range(neurons):
            a[j] = b[j]
            for i in range(len(input_values)):
                g = w[j][i] * input_values[i]
                a[j] += g
        for x in range(len(a)):
            if activation_fun == "relu":
                c[x] = Network.relu(self, a[x])
            if activation_fun == "sigmoid":
                c[x] = float(Network.sigmoid(self, a[x]))
        self.activations[layer_number] = c
        return c

    def derrivative_sig(self, x):
        return math.exp(x) / ((math.exp(x) + 1) ^ 2)

    def derrivative_relu(self, x):
        if x < 0:
            return 0
        if x >= 0:
            return 1

    def grad(self, loss):
        if loss == "mse":




    """def backprop(self, learning_rate, sgd_weights, sgd_biases):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = self.weights[i][j][k] - learning_rate * sgd_weights[i][j][k]
        for i in range(len(self.biases)):
            for k in range(len(self.biases[i])):
                self.biases[i][k] = self.biases[i][k] - learning_rate * sgd_biases[i][k]"""


net = Network(3)
l1 = net.layer_dense(input_data, 64, "relu", 0)
l2 = net.layer_dense(l1, 64, "relu", 1)
l3 = net.layer_dense(l2, 10, "sigmoid", 2)
print(l3)
