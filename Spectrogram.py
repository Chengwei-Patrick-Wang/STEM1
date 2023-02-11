import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


scale_file = "MusicFiles/21-22 WHS Chamber Orchestra Auditions (Feb 9, 2021 at 9_03 PM).wav"

scale, sr = librosa.load(scale_file)

FRAME_SIZE = 2048
HOP_SIZE = 512
S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
S_scale.shape
type(S_scale[0][0])
Y_scale = np.abs(S_scale) ** 2
Y_scale.shape
type(Y_scale[0][0])

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.show()
Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")