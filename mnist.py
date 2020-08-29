import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)
plt.imshow(x_train[0])
plt.show()
print(y_train[0])