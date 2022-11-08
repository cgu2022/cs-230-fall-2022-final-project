#import convert_files from coughvid folder
import convert_files, segmentation
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys
sys.path.append(os.path.abspath('../src'))
import librosa.display

# #convert all audio files in coughvid_databse to wav files
# convert_files.convert_files('/Users/serenazhang/Documents/cs230/coughvid_20211012/')

# #segment all wav files in coughvid_database
#try with 3 files first
i = 0
for file in os.listdir('/Users/serenazhang/Documents/cs230/coughvid_20211012/'):
    if file.endswith('.wav'):
        x,fs = librosa.load('/Users/serenazhang/Documents/cs230/coughvid_20211012/' + file, sr=None)
        S = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=128, fmax=12000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, fmax=12000)

        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-spectrogram')
        plt.tight_layout()
        plt.show()

        i += 1

    if i > 1:
        break



