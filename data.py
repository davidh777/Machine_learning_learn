import tensorflow as tf
import matplotlib.pyplot as plt
import random as random
import math as math
def relu(x):
    if x <= 0:
        y = 0
    else:
        y = x
    return y


def sigmoid(x):
    y = 1 / math.exp(-1 * x)
    return y


mnist = tf.keras.datasets.mnist
(x_train, x_test), (y_train, y_test) = mnist.load_data()
A = [["" for i in range(784)] for _ in range(60000)]
for n in range(2):
    a = 0
    for i in range(28):
        for k in range(28):
            z = x_train[n][i][k]
            A[n][a] = z
            a += 1
z = ["", "", "", "", ""]
z[0] = A
z[1] = ["" for _ in range(32)]
z[2] = ["" for _ in range(32)]
z[3] = ["" for _ in range(32)]
z[4] = ["" for _ in range(11)]
w = ["", "", "", ""]
b = ["", "", "", ""]
w[0] = [[random.random() for _ in range(len(z[0][0]))] for _ in range(32)]
w[1] = [[random.random() for _ in range(len(z[1][0]))] for _ in range(32)]
w[2] = [[random.random() for _ in range(len(z[2][0]))] for _ in range(32)]
w[3] = [[random.random() for _ in range(len(z[3][0]))] for _ in range(11)]
b[0] = [random.random() for _ in range(len(z[1]))]
b[1] = [random.random() for _ in range(len(z[2]))]
b[2] = [random.random() for _ in range(len(z[3]))]
b[3] = [random.random() for _ in range(len(z[4]))]
# model
t = 4
c = ["" for _ in range(t)]
a = 0
for n in range(t):
    c[n] = 0
    for k in range(4):
        for i in range(len(z[k + 1])):
            z[k + 1][i] = a + b[k][i]
            if k >= 0:
                for j in range(len(z[0][0])):
                    a += z[0][n][i] * w[0][i][j]

        """for j in range(11):
            if y_train[n].any == j:
                c += (1 - float(z[4][j]) ** 2)
            else:
                c += float(z[4][j]) ** 2"""




