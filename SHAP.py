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
import sys
import librosa
import librosa.display



# data, sr = librosa.load('MQ Wavs/patient 1296 cut.wav', sr=44100)
# mel = librosa.feature.melspectrogram(y=data, sr=44100)
# meldb = librosa.core.power_to_db(mel, ref=np.max) # Power scale to decibel

# librosa.display.specshow(meldb, y_axis='mel', x_axis='time')
# plt.axis('off')
# plt.xticks(color='w')
# plt.yticks(color='w') # Remove labels and formatting on image
# plt.xlabel(None)
# plt.ylabel(None)
# plt.ylim(0, 1.8e3)
# plt.tick_params(
#     axis='both', which="both", 
#     labelbottom=False, labelleft=False,
#     bottom=False, top=False, 
#     left=False, right=False
# )
# plt.show()
# sys.exit()


#### Pick Model and Load Dataset ####
LABELS = {'AS': 0, 'MR': 1, 'MS': 2, 'MVP': 3, 'N': 4}             # Numeric labels for heart conditions
TRAINING_PATH = 'training_data_cropped.npy'                        # Spectrogram training data
MODEL_NUM = '#68'                                                  # Which Model number to utilize
MODEL_PATH = 'PAPER Models/PAPER Model ' + MODEL_NUM               # Directory where CNN model is
NUM = 1296                                                         # Multilabel Case Number to use

#### Load Model, Data, & Multilabel Image ####
data = np.load(TRAINING_PATH, allow_pickle=True)                   # Load training spectrogram dataset
X, Y = [n[0] for n in data], [n[1] for n in data]                  # Separate data and labels
X = np.array(X).reshape(-1, 240, 320, 3)                           # Certify 320x240 resolution
Y = np.array(Y)
X = X/255.0                                                        # Normalize range to interval [0,1]

x_tr, x_val, y_tr, y_val = train_test_split(X, Y, stratify=Y, test_size=0.15, shuffle=True)
input_shape = X.shape[1:]
model = tf.keras.models.load_model(MODEL_PATH)                     # Import CNN Model

# img = np.array(cv2.imread('TEST MQ/TEST MQ ' + str(NUM) + '.png')) # Read Multilabel Image
img = np.array(cv2.imread('mq-1296.png'))
img = cv2.resize(np.float32(img), (320, 240))                                  # Certify 320x240 resolution
img = img.reshape(-1, 240, 320, 3)                                 # Reshape for use as an input
img = img.astype(float)


#### SHAP Calculations ####
background = x_tr[np.random.choice(x_tr.shape[0], 100, replace=False)] # Create background training for SHAP algorithm
e = shap.DeepExplainer(model, background)                              # Initialize SHAP Deep Explainer
shap_values = e.shap_values(img)                                       # Calculate the SHAP Values

shap.image_plot(shap_values, img, show=False)
# plt.title('N_001 Stretched')
plt.show()