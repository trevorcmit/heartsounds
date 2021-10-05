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
for folder in tqdm(os.listdir('New Wavs')): # Folder type
    path = 'MQ Wavs'
    for file in os.listdir(path): # File in specific folder
        counter = 0
        label = file # Save patient number for filename
        filepath = os.path.join(path, file)
        data, sr = librosa.load(filepath, sr=8000)
    
        mel = librosa.feature.melspectrogram(y=n, sr=sr)
        meldb = librosa.core.power_to_db(mel, ref=np.max) # Power scale to decibel
        librosa.display.specshow(meldb, y_axis='mel', x_axis='time')
        plt.axis('off')
        plt.xticks(color='w')
        plt.yticks(color='w') # Remove labels and formatting on image
        plt.xlabel("")
        plt.ylabel("")
        save = 'MQ Spect/' + label + '.png'
        plt.savefig(save, bbox_inches='tight', pad_inches=0)