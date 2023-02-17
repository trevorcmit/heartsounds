# Python 3.8.1
# Tensorflow 2.4.0
# Keras 2.6.0
# SHAP 0.39.0

# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from sklearn.model_selection import train_test_split
import shap
import cv2
import librosa.display


def CNN_explainer(vals, training, model, plot=True):
    """
    Uses SHAP to create a DeepExplainer model to explain how
    input convolutional neural network predicts

    Parameters:
    vals - array of input images to predict
    training - array of images to train DeepExplainer
    model - input CNN to analyze
    plot - optional bool, plots visualizer if True

    Returns:
    shap_values - list of shap_values for each image in vals
    """
    background = training[np.random.choice(training.shape[0], 60, replace=True)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(vals)
    for val in vals:
        print(model.predict(val.reshape(1, 240, 320, 3)) > 0.5)
    if plot:
        shap.image_plot(shap_values, vals, show=False)
        # plt.title('Image AS_001 on 400 img Explainer.png', loc='left')
        # plt.savefig('SHAP plots/assortment.png')
        plt.show()
    return shap_values


# data, sr = librosa.load('MQ Wavs/patient 1271 cut.wav', sr=44100)
# mel = librosa.feature.melspectrogram(y=data, sr=44100)
# meldb = librosa.core.power_to_db(mel, ref=np.max) # Power scale to decibel

# librosa.display.specshow(meldb, y_axis='mel', x_axis='time')
# plt.axis('off')
# plt.xticks(color='w')
# plt.yticks(color='w') # Remove labels and formatting on image
# plt.xlabel(None)
# plt.ylabel(None)
# plt.ylim(0, 1.8e3) # Set vertical axis limits
# plt.tick_params(
#     axis='both', which="both", 
#     labelbottom=False, labelleft=False,
#     bottom=False, top=False, 
#     left=False, right=False
# )
# plt.show()


#### Pick Model and Load Dataset ####
LABELS = {'AS': 0, 'MR': 1, 'MS': 2, 'MVP': 3, 'N': 4}             # Numeric labels for heart conditions
TRAINING_PATH = 'training_data_cropped.npy'                        # Spectrogram training data
MODEL_NUM = '#68'                                                  # Which Model number to utilize
MODEL_PATH = 'PAPER Models/PAPER Model ' + MODEL_NUM               # Directory where CNN model is
NUM = 101                                                         # Multilabel Case Number to use

#### Load Model, Data, & Multilabel Image ####
data = np.load(TRAINING_PATH, allow_pickle=True)                   # Load training spectrogram dataset
X, Y = [n[0] for n in data], [n[1] for n in data]                  # Separate data and labels
X = np.array(X).reshape(-1, 240, 320, 3)                           # Certify 320x240 resolution
Y = np.array(Y)
# print(max(X))
# sys.exit()
X = X/255.0                                                        # Normalize range to interval [0,1]

x_tr, x_val, y_tr, y_val = train_test_split(X, Y, stratify=Y, test_size=0.15, shuffle=True)
input_shape = X.shape[1:]
model = tf.keras.models.load_model(MODEL_PATH)                     # Import CNN Model

# img = np.array(cv2.imread('TEST MQ/TEST MQ ' + str(NUM) + '.png')) # Read Multilabel Image
img = np.array(cv2.imread('TEST MQ/mq-' + str(NUM) + '.png'))
img = cv2.resize(np.float32(img), (320, 240))                      # Certify 320x240 resolution
img = img.reshape(-1, 240, 320, 3)                                 # Reshape for use as an input
img = img.astype(float)


#### SHAP Calculations ####
background = x_tr[np.random.choice(x_tr.shape[0], 100, replace=False)] # Create background training for SHAP algorithm

import sys
print(img.shape); sys.exit()
# import sys
# sys.exit()
e = shap.DeepExplainer(model, background)                              # Initialize SHAP Deep Explainer
shap_values = e.shap_values(img)                                       # Calculate the SHAP Values


shap.image_plot(shap_values, img, show=False)
plt.show()