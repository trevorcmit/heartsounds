import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import sys


def AudioToMelspectro(file, graph=False, save=False):
    """
    Converts audio file to melspectrogram, then 
    converts power scale to decibel scale.

    Parameters:
    file - audio file in any common format
    graph - optional bool, graphs melspectrogram if True
    save - optional bool, saves melspectrogram image if True

    Returns:
    decibel-scale melspectrogram
    """
    y, sr = librosa.load(file)
    D = np.abs(librosa.stft(y)) #**2 # Power 2 scale
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    meldb = librosa.core.power_to_db(mel, ref=np.max) # Power scale to decibel
    if graph:
        librosa.display.specshow(meldb, y_axis='mel', x_axis='time') ####
        plt.xlabel('Time (Seconds)')
        # plt.ylabel('Frequency (Hz)')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
    if save:
        name = os.path.basename(file).split(".")[0]
        librosa.display.specshow(meldb)
        plt.savefig(str(name) + ".png")
    return meldb


def MelToTimeSeries(melspec, sampling, graph=False):
    """
    Converts melspectrogram to time series data 
    using Griffin-Lim Algorithm

    Parameters:
    melspec - melspectrogram to be converted
    sampling - sampling rate of original spectrogram audio
    graph - optional bool, graphs time series data

    Returns:
    time series data as numpy array
    """
    power = librosa.core.db_to_power(melspec)
    tseries = librosa.feature.inverse.mel_to_audio(
        power, sr=sampling, n_fft=2048, 
        center=True, power=2.0, n_iter=32
        )
    if graph:
        plt.plot(tseries)
        plt.show()
    return tseries