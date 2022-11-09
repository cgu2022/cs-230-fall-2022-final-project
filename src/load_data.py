#import convert_files from coughvid folder
from segmentation import segment_cough
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys
import librosa.display
import pandas as pd

# #convert all audio files in coughvid_databse to wav files
# convert_files.convert_files('/Users/serenazhang/Documents/cs230/coughvid_20211012/')

# #segment all wav files in coughvid_database
#try with 3 files first
# i = 0
# for file in os.listdir('/Users/serenazhang/Documents/cs230/coughvid_20211012/'):
#     if file.endswith('.wav'):
#         x,fs = librosa.load('/Users/serenazhang/Documents/cs230/coughvid_20211012/' + file, sr=None)
#         S = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=128, fmax=12000)
#         S_dB = librosa.power_to_db(S, ref=np.max)
#         librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=fs, fmax=12000)

#         plt.colorbar(format='%+2.0f dB')
#         plt.title('Mel-spectrogram')
#         plt.tight_layout()
#         plt.show()

#         i += 1

#     if i > 1:
#         break

# data formatting

#load metadata_compiled.csv
df = pd.read_csv('/Users/serenazhang/Documents/cs230/coughvid_20211012/metadata_compiled.csv', header=0)
#remove rows with cough_detected < .8
df = df[df['cough_detected'] >= .8]
#print('cough_detected removed: ', df)
#remove rows where status == NaN
df = df[df['status'].notna()]
#print('status removed: ', df)
#find all rows where quality_1 or quality_2 or quality_3 or quality_4 are not NaN and add a column to df called professionally_verified where the value is 1
df['professionally_verified'] = np.where((df['quality_1'].notna()) | (df['quality_2'].notna()) | (df['quality_3'].notna()) | (df['quality_4'].notna()), 1, 0)
#print('added pv: ', df)
#count all the samples where status is covid, healthy, or symptomatic and print
#print('covid: ', df[df['status'] == 'COVID-19'].count())
#print('healthy: ', df[df['status'] == 'healthy'].count())
#print('symptomatic: ', df[df['status'] == 'symptomatic'].count())

#iterate over all rows exluding row 0 in df
print(df)
for row in range(0, len(df)):
    print('PROGRESS: ', str(row) + '/13535')
    #get uuid from row index
    uuid = df.iloc[row, 1]
    print('uuid: ', uuid)
    #add wav and json file with uuid to a new folder called valid_data
    #if uuid.wav and uuid.json exist in coughvid_database, copy to valid_data
    if os.path.exists('/Users/serenazhang/Documents/cs230/coughvid_20211012/' + uuid + '.wav') and os.path.exists('/Users/serenazhang/Documents/cs230/coughvid_20211012/' + uuid + '.json'):
        #copy to valid_data or create valid_data if it doesn't exist
        if not os.path.exists('/Users/serenazhang/Documents/cs230/valid_data'):
            os.mkdir('/Users/serenazhang/Documents/cs230/valid_data')
        os.system('cp /Users/serenazhang/Documents/cs230/coughvid_20211012/' + uuid + '.wav /Users/serenazhang/Documents/cs230/valid_data/')
        os.system('cp /Users/serenazhang/Documents/cs230/coughvid_20211012/' + uuid + '.json /Users/serenazhang/Documents/cs230/valid_data/')

#convert df into csv and add to valid_data folder
df.to_csv('/Users/serenazhang/Documents/cs230/valid_data/metadata_compiled_valid.csv')











