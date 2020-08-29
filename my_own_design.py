import math
import random


"""class Network(object):
    def __init__(self, number_layers):
        self.weights = ["" for _ in range(number_layers)]
        self.biases = ["" for _ in range(number_layers)]


    def relu(self, x):
        if x > 0:
            return x
        if x <= 0:
            return 0

    def sigmoid(self, x):
        return 1 / (1 + math.e ** - x)

    def layer_dense(self, input_values, neurons, activation_fun, layer_number):
        w = [[random.randint(-1, 1) for _ in range(len(input_values))] for i in range(neurons)]
        b = [random.randint(-1, 1) for _ in range(neurons)]
        Network.weights[layer_number] = w
        Network.biases[layer_number] = b

        a = ["" for _ in range(neurons)]
        c = ["" for _ in range(neurons)]

        for j in range(neurons):
            a[j] = b[j]
            for i in range(len(input_values)):
                g = w[j][i] * input_values[i]
                a[j] += g
        for x in range(len(a)):
            if activation_fun == "relu":
                c[x] = Network.relu(a[x])
            if activation_fun == "sigmoid":
                c[x] = Network.sigmoid(a[x])
        return c"""
print(random.random() * random.randint(-1, 1))





