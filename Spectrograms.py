import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

sr, data = read('MQ Wavs/patient 1132 cut.wav')
# sr, data = read('New Wavs/N/New_N_001.wav')
# print(len(data))



mel = librosa.feature.melspectrogram(y=data.astype(float), sr=44100)
meldb = librosa.core.power_to_db(mel, ref=np.max) # Power scale to decibel
librosa.display.specshow(meldb, y_axis='mel', x_axis='time', sr=44100)

plt.xlabel(None)
plt.ylabel('Frequency (Hz)')
plt.ylim(0, 2000)
# plt.title('Case #1296')
plt.tick_params(axis='both', which="both", 
        labelbottom=False, labelleft=True,
        bottom=False, top=False, 
        left=True, right=False)
plt.show()
