import sys
import numpy as np
from scipy.ndimage import shift

def increase_amplitude(sample_array, amplitude_factor=2):
    real_part = np.real(sample_array)
    scaled_real_part = amplitude_factor * real_part
    new_sample_array = scaled_real_part + 1j * np.imag(sample_array)
    return new_sample_array

def remove_noises(sample_array, noise_factor=0.2):
    new_sample_array = sample_array
    np.place(new_sample_array, np.abs(new_sample_array) ** 2 < noise_factor, 0j)
    return new_sample_array

def shift_tone(sample_array, shift_factor=10):
    return shift(sample_array, (shift_factor, 0), cval=0j)

def silent_period_test(sample_array, start=0, end=20):
    new_sample_array = sample_array
    for i in range(start, end):
        frame = new_sample_array[:,i]
        for j in range(len(frame)):
            frame[j] = 0j
    return new_sample_array