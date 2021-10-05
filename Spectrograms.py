import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

dataset = list()
path = 'New Wavs' # Folder directory for current folder to convert
for folder in tqdm(os.listdir(path)): # Folder type
    filepath = os.path.join(path, folder)
    for file in os.listdir(filepath): # File in specific folder
        counter = 0
        label = file # Save condition (AS, MR, MS, MVP, N) as label
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

# mel = librosa.feature.melspectrogram(y=data.astype(float), sr=8000)
# meldb = librosa.core.power_to_db(mel, ref=np.max) # Power scale to decibel
# librosa.display.specshow(meldb, y_axis='mel', x_axis='time', sr=8000)

# plt.xlabel(None)
# plt.ylabel('Frequency (Hz)')
# plt.tick_params(axis='both', which="both", 
#         labelbottom=False, labelleft=True,
#         bottom=False, top=False, 
#         left=True, right=False)
# plt.show()
