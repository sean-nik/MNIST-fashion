# Fashion MNIST analysis w/ Keras & Tensorflow using Sequential API
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
( X_train_full , y_train_full ), ( X_test , y_test ) = fashion_mnist.load_data() 

X_train_full.shape
X_train_full.dtype

#create validation set and normalize values 
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class_names[y_train[0]]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu")) # same as specifying activation = keras.activations.relu
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax")) # using softmax as it's multiclass classification & classes are mutually exclusive

model.summary()
model.layers
hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
weights.shape
biases.shape

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(X_test, y_test)

# using the model to make predictions
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

#alternatively just to get the classes
y_pred = np.argmax(model.predict(X_new), axis=-1)
np.array(class_names)[y_pred]
