# Necessary imports
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from keras.callbacks import History

#### Load Data and split into Testing/Training ####
data = np.load('training_data.npy', allow_pickle=True)
X, Y = list(), list()
for features, label in data: # Separate spectrogram from its label
    X.append(features)
    Y.append(label)
X, Y = np.array(X).reshape(-1, 320, 240, 3), np.array(Y) # Reshape to fit CNN input size
X = X / 255.0
x_tr, x_val, y_tr, y_val = train_test_split(X, Y, stratify=Y, test_size=0.20, shuffle=True)


#### Construct Neural Network and Train ####
history = History() # Initialize history for plotting later
def cnn_model():
    """
    Function representing creation of Neural Network
    """
    model = Sequential()
    model.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(320, 240, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    model.add(Conv2D(10, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(10, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(10, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(10, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

model = cnn_model()
model.summary()
model.fit(x_tr, y_tr, epochs=50, batch_size=20, validation_data=(x_val, y_val))

model.save('Models/Single Beat #' + Number, save_format='tf')

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.savefig('Models/Single Beat #' + Number + '/loss graph.png')
plt.show()
plt.plot(history.history['acc'], color='b', label='Train Accuracy')
plt.plot(history.history['val_acc'], color='r', label='Test Accuracy')
plt.legend()
plt.savefig('Models/Single Beat #' + Number + '/accuracy graph.png')
plt.show()