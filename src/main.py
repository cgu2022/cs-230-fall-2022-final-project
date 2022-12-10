import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import audiomentations
from torch_audiomentations import Compose, PitchShift, TimeInversion, AddBackgroundNoise, AddColoredNoise, PolarityInversion
import torchaudio
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorboard
from torch.utils.tensorboard import SummaryWriter
#import build_features.py from features folder in src
from features.build_features import CoughDataset
#import train_model.py from models folder in src
from models.cnn_large import CNNLarge
from models.cnn_small import CNNSmall
from models.crnn_att import CRNN_with_Attention
from models.crnn import CRNN
import run_models.utils


WRITER_PATH ="../logs/CRNNA"
AUDIO_DIR = "../valid_data/"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE*10
BATCH_SIZE = 256
EPOCHS = 50
MODEL_FOLDER = '../models/CRNN'


def train(batch_size, epochs, model, model_folder):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_df = pd.read_parquet(os.path.join(AUDIO_DIR, "train_balanced_3500.parquet.gzip"))
    val_df = pd.read_parquet(os.path.join(AUDIO_DIR, "val_balanced_3500.parquet.gzip"))


    train_data = CoughDataset(train_df,
                            AUDIO_DIR,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device,
                            )

    val_data = CoughDataset(val_df,
                            AUDIO_DIR,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,drop_last=True) 
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE,drop_last=True) 

    # construct model and assign it to device
    model = model

    # initialise loss funtion + optimiser
    #loss_fn = nn.CrossEntropyLoss(weight=train_data.label_weights)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-5)

    train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs, do_augment=True, writer_path=WRITER_PATH, model_folder)

def test(batch_size, epochs, model, model_folder):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    test_df = pd.read_parquet(os.path.join(AUDIO_DIR, "test_balanced_3500.parquet.gzip"))
    test_data = CoughDataset(test_df,
                            AUDIO_DIR,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,drop_last=True) 



#if name == main
if __name__ == "__main__":
    #train(batch_size, epochs, model, model_folder)
    train(BATCH_SIZE, EPOCHS, CRNN_with_Attention(), MODEL_FOLDER)