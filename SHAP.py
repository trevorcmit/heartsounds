# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import shap
import tensorflow.keras.backend
import cv2


#####################################
#### Pick Model and Load Dataset ####
#####################################
LABELS = {'AS': 0, 'MR': 1, 'MS': 2, 'MVP': 3, 'N': 4}  # Numeric labels for heart conditions
TRAINING_PATH = 'training_data_cropped.npy'             # Spectrogram training data
MODEL_NUM = '#68'                                       # Which Model number to utilize
MODEL_PATH = 'PAPER Models/PAPER Model ' + MODEL_NUM    # Directory where CNN model is
NUM = 1048                                              # Multilabel Case Number to use


##############################################
#### Load Model, Data, & Multilabel Image ####
##############################################
data = np.load(TRAINING_PATH, allow_pickle=True)  # Load training spectrogram dataset
X, Y = [n[0] for n in data], [n[1] for n in data] # Separate data and labels
X = np.array(X).reshape(-1, 240, 320, 3)          # Certify 320x240 resolution
Y = np.array(Y)
X = X/255.0                                       # Normalize range to interval [0,1]

x_tr, x_val, y_tr, y_val = train_test_split(X, Y, stratify=Y, test_size=0.15, shuffle=True)
input_shape = X.shape[1:]
model = tf.keras.models.load_model(MODEL_PATH)                     # Import CNN Model
img = np.array(cv2.imread('TEST MQ/TEST MQ ' + str(NUM) + '.png')) # Read Multilabel Image
img = cv2.resize(img, (320, 240))                                  # Certify 320x240 resolution
img = img.reshape(-1, 240, 320, 3)                                 # Reshape for use as an input
img = img.astype(float)


###########################
#### SHAP Calculations ####
###########################
background = x_tr[np.random.choice(x_tr.shape[0], 105, replace=False)] # Create background training for SHAP algorithm
e = shap.DeepExplainer(model, background)                              # Initialize SHAP Deep Explainer
shap_values = e.shap_values(img)                                       # Calculate the SHAP Values