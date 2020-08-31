import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
import numpy as np
# imports the tensorflow library 
# In this script I am using a CNN for the Mnist Handwritten digit database where previous attempts of mine used normal feedforward NN

data = tf.keras.datasets.mnist

(x_train,y_train), (x_test, y_test) = data.load_data() # This sets up both the trianing data and the testing data as the network will attept to the fit the x_train to the y_train



x = np.concatenate((x_train,x_test), axis=0)
y = np.concatenate((y_train,y_test), axis=0)
print(y.max())
y0 = np.zeros((len(y),10))
for i in range(len(y)):
    h = y[i]
    y0[i][h - 1] = 1
    
x = x / 255.0
x = np.expand_dims(x,-1)

input_shape = (28,28,1)

model = Sequential()

model.add(Conv2D(64,(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x, y0, batch_size=32, epochs=3, validation_split=.1)
model.summary()
