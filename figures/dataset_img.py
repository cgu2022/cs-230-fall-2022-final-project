# build graph of train, val, test dataset to show how many healthy and covid samples are in each set
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

AUDIO_DIR = "../valid_data/"

def oversampled_ds_image():
    healthy_train = 0
    covid_train = 0
    healthy_val = 0
    covid_val = 0
    healthy_test = 0
    covid_test = 0

    train_b = pd.read_parquet("train_balanced_3500.parquet.gzip")
    val_b = pd.read_parquet("val_balanced_3500.parquet.gzip")
    test_b = pd.read_parquet("test_balanced_3500.parquet.gzip")

    for index, row in train_b.iterrows():
        if row['status'] == 'healthy':
            healthy_train += 1
        else:
            covid_train += 1
    for index, row in val_b.iterrows():
        if row['status'] == 'healthy':
            healthy_val += 1
        else:
            covid_val += 1
    for index, row in test_b.iterrows():
        if row['status'] == 'healthy':
            healthy_test += 1
        else:
            covid_test += 1

    #use this to plot the bar graph
    healthy = [healthy_train, healthy_val, healthy_test]
    covid = [covid_train, covid_val, covid_test]
    labels = ['train', 'val', 'test']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    #plot the bar graph
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, healthy, width, label='healthy')
    rects2 = ax.bar(x + width/2, covid, width, label='covid')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of samples')
    ax.set_title('COVID-19 vs Healthy Balanced')
    # add labels on top of the bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    # change the color of the bars
    rects1[0].set_color('tab:red')
    rects1[1].set_color('tab:red')
    rects1[2].set_color('tab:red')
    rects2[0].set_color('tab:green')
    rects2[1].set_color('tab:green')
    rects2[2].set_color('tab:green')
    # add buffer to the y axis
    ax.set_ylim(top=4000)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

def orig_ds_image():
    # graph the number of healthy, covid, and symptomatic samples in each set
    healthy_train = 0
    symptomatic_train = 0
    covid_train = 0
    healthy_val = 0
    symptomatic_val = 0
    covid_val = 0
    healthy_test = 0
    symptomatic_test = 0
    covid_test = 0

    train = pd.read_parquet(os.path.join(AUDIO_DIR, "train_edited.parquet.gzip"))
    val = pd.read_parquet(os.path.join(AUDIO_DIR, "val_edited.parquet.gzip"))
    test = pd.read_parquet(os.path.join(AUDIO_DIR, "test_edited.parquet.gzip"))

    for index, row in train.iterrows():
        if row['status'] == 'healthy':
            healthy_train += 1
        elif row['status'] == 'symptomatic':
            symptomatic_train += 1
        else:
            covid_train += 1
    for index, row in val.iterrows():
        if row['status'] == 'healthy':
            healthy_val += 1
        elif row['status'] == 'symptomatic':
            symptomatic_val += 1
        else:
            covid_val += 1
    for index, row in test.iterrows():
        if row['status'] == 'healthy':
            healthy_test += 1
        elif row['status'] == 'symptomatic':
            symptomatic_test += 1
        else:
            covid_test += 1

    #use this to plot the bar graph
    healthy = [healthy_train, healthy_val, healthy_test]
    symptomatic = [symptomatic_train, symptomatic_val, symptomatic_test]
    covid = [covid_train, covid_val, covid_test]
    labels = ['train', 'val', 'test']
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    #plot the bar graph
    fig, ax = plt.subplots()
    #divide into 3 groups
    rects1 = ax.bar(x - width, healthy, width, label='healthy')
    rects2 = ax.bar(x, symptomatic, width, label='symptomatic')
    rects3 = ax.bar(x + width, covid, width, label='covid')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of samples')
    ax.set_title('Original Coughvid Dataset')
    # add labels on top of the bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in rects3:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    # change the color of the bars
    rects1[0].set_color('tab:red')
    rects1[1].set_color('tab:red')
    rects1[2].set_color('tab:red')
    rects2[0].set_color('tab:green')
    rects2[1].set_color('tab:green')
    rects2[2].set_color('tab:green')
    rects3[0].set_color('tab:blue')
    rects3[1].set_color('tab:blue')
    rects3[2].set_color('tab:blue')
    # add a little buffer to the y axis

    # add spacing between train, val, and test on the x axis
    ax.set_ylim(top=9000)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

