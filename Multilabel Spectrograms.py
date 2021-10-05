import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import librosa.display
from scipy.io.wavfile import read, write

dataset = list()
path = 'MQ Wavs'
for folder in tqdm(os.listdir(path)): # Folder type
    filepath = os.path.join(path, folder)
    for file in os.listdir(filepath): # File in specific folder
        counter = 0
        label = file # Save patient number for filename
        data, sr = librosa.load(filepath, sr=8000)
        mel = librosa.feature.melspectrogram(y=n, sr=sr)
        meldb = librosa.core.power_to_db(mel, ref=np.max) # Power scale to decibel
        librosa.display.specshow(meldb, y_axis='mel', x_axis='time')
        plt.axis('off')
        plt.xticks(color='w')
        plt.yticks(color='w') # Remove labels and formatting on image
        plt.xlabel(None)
        plt.ylabel(None)
        plt.tick_params(
            axis='both', which="both", 
            labelbottom=False, labelleft=False,
            bottom=False, top=False, 
            left=False, right=False
        )
        save = 'MQ Spect/' + label + '.png'
        plt.savefig(save, bbox_inches='tight', pad_inches=0)