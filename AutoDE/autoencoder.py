# -*- coding: utf-8 -*-
# @Author   : TAKR-Zz
# @Time     : 2022/7/7 17:13
# @FileName : autoencoder

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import regularizers
from keras.utils import np_utils

(X_train, _), (X_test, _) = mnist.load_data()
X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
print(type(X_train))
print(X_train.shape)
print(X_test.shape)

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

print(X_train.shape)
print(X_test.shape)

input_size = 28 * 28
hidden_size = 64
output_size = 28 * 28

x = Input(shape=(input_size,))
h = Dense(hidden_size, activation="relu")(x)
r = Dense(output_size, activation='sigmoid')(h)

autoEncoder = Model(x, r)
autoEncoder.compile(optimizer="adam", loss="mse")

epochs = 5
batch_size = 128

history = autoEncoder.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_test, X_test)
                          )

conv_encoder = Model(x, h)
encoded_imgs = conv_encoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape(4, -1).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# plt.show()


decoded_imgs = autoEncoder.predict(X_test)

plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc="upper right")
plt.show()
