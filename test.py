from os.path import dirname, join as pjoin
from scipy.io import wavfile
import numpy as np
import scipy.io
import math

dtypes = {
    "float32": "32_bit_floating_point",
    "int32": "32-bit integer PCM",
}

samplerate, data = wavfile.read("test.wav")

print(f"number of channels = {data.shape[1]}")

length = data.shape[0] / samplerate
print(f"length = {length}s")

channel_1_info = data[:, 0]
channel_2_info = data[:, 1]

frames_per_second = 60

scanner_length = samplerate
scanner_begin = 0
scanner_end = scanner_begin + scanner_length

number_of_seconds = math.floor(len(channel_1_info) / scanner_length)
number_of_divisions = math.floor(samplerate / frames_per_second)

frames_data = []
