#import convert_files from coughvid folder
from segmentation import segment_cough
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys
import librosa.display
import pandas as pd

AUDIO_DIR = '../../coughvid_20211012'
df = pd.read_csv(os.path.join(AUDIO_DIR, 'metadata_compiled.csv'), header=0)

df = df[df['cough_detected'] >= .98]
df = df[df['status'].notna()]
df['professionally_verified'] = np.where((df['quality_1'].notna()) | (df['quality_2'].notna()) | (df['quality_3'].notna()) | (df['quality_4'].notna()), 1, 0)

print(df)
for row in range(0, len(df)):
    print('PROGRESS: ', str(row) + '/13535')
    uuid = df.iloc[row, 1]
    print('uuid: ', uuid)
    #add wav and json file with uuid to a new folder called valid_data
    #if uuid.wav and uuid.json exist in coughvid_database, copy to valid_data
    if os.path.exists(AUDIO_DIR + uuid + '.wav') and os.path.exists(AUDIO_DIR + uuid + '.json'):
        #copy to valid_data or create valid_data if it doesn't exist
        if not os.path.exists('../valid_data'):
            os.mkdir('../valid_data')
        os.system('cp ' + AUDIO_DIR + uuid + '.wav ../valid_data')
        os.system('cp ' + AUDIO_DIR + uuid + '.json ../valid_data')

#convert df into csv and add to valid_data folder
df.to_csv('../valid_data/metadata_compiled_valid.csv')











