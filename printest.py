import os
import numpy as np
from matplotlib import pyplot as plt
import IPython.display as ipd
import librosa
import pandas as pd


def print_plot_play(x, Fs, text=''):
    """1. Prints information about an audio singal, 2. plots the waveform, and 3. Creates player

    Notebook: C1/B_PythonAudio.ipynb

    Args:
        x: Input signal
        Fs: Sampling rate of x
        text: Text to print
    """
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    ipd.display(ipd.Audio(data=x, rate=Fs))


# Read wav
fn_wav = os.path.join('./online_audio/', 'fine_neutral.wav')
x, Fs = librosa.load(fn_wav, sr=22050)
print_plot_play(x=x, Fs=Fs, text='WAV file: ')

fn_wav = os.path.join('./online_audio/', 'fine_negative.wav')
x, Fs = librosa.load(fn_wav, sr=22050)
print_plot_play(x=x, Fs=Fs, text='WAV file: ')

plt.show()