import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

AUDIO_DIR = "../valid_data/"
ANNOTATIONS_FILE = os.path.join(AUDIO_DIR, "metadata_compiled_valid_edited.parquet.gzip")

def make_dataset(ANNOTATIONS_FILE):
    df = pd.read_parquet(ANNOTATIONS_FILE)

    train_val, test = train_test_split(df, test_size=0.1, random_state=0, stratify=df['status'])
    train, val = train_test_split(train_val, test_size=0.1, random_state=0, stratify=train_val['status'])

    # for val and test make sure no status = symptomatic gets in
    val = val[val['status'] != 'symptomatic']
    test = test[test['status'] != 'symptomatic']
    val = val.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    test.to_parquet(os.path.join(AUDIO_DIR, "test_balanced_3500.parquet.gzip"))
    val.to_parquet(os.path.join(AUDIO_DIR, "val_balanced_3500.parquet.gzip"))


    # for the test set, we want to make sure that we have 50% healthy and 50% covid
    df = train
    df2 = pd.DataFrame(columns=df.columns)
    covid = 0
    healthy = 0
    for index, row in df.iterrows():
        if os.path.isfile(os.path.join(AUDIO_DIR, row['uuid'] + '.wav')):
            if row['status'] == 'healthy':
                if row['cough_detected'] > 0.98:
                    if healthy < 3500:
                        healthy += 1
                        df2 = df2.append(row, ignore_index=True)
            elif row['status'] == 'COVID-19':
                # add the row 5 times in random locations to the dataframe
                for i in range(5):
                    df2 = df2.append(row, ignore_index=True)
                    covid += 1
    df2 = df2.sample(frac=1).reset_index(drop=True)
    df2.to_parquet(os.path.join(AUDIO_DIR, "train_balanced_3500.parquet.gzip"))
    #save as csv
    df2.to_csv(os.path.join(AUDIO_DIR, "train_balanced_3500.csv"))

def print_dataset():
    # read in the parquet files
    test = pd.read_parquet(os.path.join(AUDIO_DIR, "test_balanced_3500.csv"))
    val = pd.read_parquet(os.path.join(AUDIO_DIR, "val_balanced_3500.csv"))
    train = pd.read_parquet(os.path.join(AUDIO_DIR, "train_balanced_3500.csv"))

    #print
    print('test: ', test)
    print('val: ', val)
    print('train: ', train)





