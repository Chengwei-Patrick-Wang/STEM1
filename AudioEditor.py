import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from strategy import *


#scale_file = "MusicFiles/21-22 WHS Chamber Orchestra Auditions (Feb 9, 2021 at 9_03 PM).wav"
scale_file = "MusicFiles/Csardas.wav"
#scale_file = "C:/Windows/Media/Alarm01.wav"

# Read wave file
scale, sr = librosa.load(scale_file, sr=None)
length = len(scale)
print(f"Sample rate: {sr} Length: {length}")

FRAME_SIZE = 2048
HOP_SIZE = 512

# Short-Time Fourier Transform (STFT)
y_pad = librosa.util.fix_length(scale, size=length + FRAME_SIZE // 2)
S_scale = librosa.stft(y_pad, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

# Apply strategies
Y_scale = increase_amplitude(S_scale, amplitude_factor=100)
#Y_scale = remove_noises(S_scale)
#Y_scale = shift_tone(S_scale)
#Y_scale = silent_period_test(S_scale)

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    fig = plt.figure(figsize=(25, 10))
    img = librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format='%+02.0f dB')
    plt.show()

# Plot spectrogram
Y_log_scale = librosa.power_to_db(np.abs(Y_scale) ** 2)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")

# Invert stft
R_scale = librosa.istft(Y_scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, length=length)

# De-normalize
nbits = 16
D_scale = R_scale * (2 ** (nbits - 1))

# Write wave file
from scipy.io.wavfile import write
write("MusicFiles/output.wav", sr, D_scale.astype(np.int16))