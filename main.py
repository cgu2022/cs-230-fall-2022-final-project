import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import audiomentations
from torch_audiomentations import (
    Compose,
    PitchShift,
    TimeInversion,
    AddBackgroundNoise,
    AddColoredNoise,
    PolarityInversion,
)
import torchaudio
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import argparse

# import build_features.py from features folder in src
from cough_dataset import CoughDataset

# import train_model.py from models folder in src
from models.cnn_large import CNN_Large
from models.cnn_small import CNN_Small
from models.crnn_att import CRNN_with_Attention
from models.crnn import CRNN

# import run_models.utils

# WRITER_PATH = "../logs/CRNNA"
# AUDIO_DIR = "../valid_data/"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 10
# MODEL_FOLDER = "../models_output/CRNN"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128
).to(DEVICE)

augmentations = Compose(
    transforms=[
        PitchShift(
            mode="per_example", p=0.5, sample_rate=SAMPLE_RATE, output_type="tensor"
        ),
        TimeInversion(mode="per_example", p=0.5, output_type="tensor"),
        AddColoredNoise(
            mode="per_example", p=0.5, sample_rate=SAMPLE_RATE, output_type="tensor"
        ),
        PolarityInversion(mode="per_example", p=0.5, output_type="tensor"),
    ],
    output_type="tensor",
)


def count_correct(logits, y_true):
    y_pred = torch.argmax(logits, axis=1)
    return torch.sum(y_pred == y_true)


def train_single_epoch(
    model,
    train_data_loader,
    val_data_loader,
    loss_fn,
    optimiser,
    device,
    do_augment=False,
):
    total_loss_train = 0.0
    correct_pred_train = 0.0
    total_pred_train = 0

    train_trues = []
    train_preds = []

    for x_batch, y_batch in tqdm(train_data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        if do_augment:
            x_batch = augmentations(x_batch, SAMPLE_RATE)

        x_batch = x_batch.reshape(-1, x_batch.shape[-1])
        x_batch = mel_spectrogram(x_batch)
        x_batch = x_batch.reshape(
            x_batch.shape[0], 1, x_batch.shape[-2], x_batch.shape[-1]
        )

        # calculate loss
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        # add to list for f1 score
        train_trues.append(y_batch.cpu())
        train_preds.append(y_pred.cpu())

        correct_pred_train += count_correct(y_pred, y_batch)
        total_pred_train += y_batch.shape[0]

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss_train += loss.item()

    print(
        f"Training loss: {total_loss_train}, Training accuracy : {correct_pred_train/total_pred_train}"
    )
    print(f"Training loss normalized: {total_loss_train/len(train_data_loader)}")

    total_loss_val = 0.0
    correct_pred_val = 0.0
    total_pred_val = 0

    val_trues = []
    val_preds = []

    for x_batch, y_batch in tqdm(val_data_loader):
        with torch.no_grad():
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            x_batch = x_batch.reshape(-1, x_batch.shape[-1])
            x_batch = mel_spectrogram(x_batch)
            x_batch = x_batch.reshape(
                x_batch.shape[0], 1, x_batch.shape[-2], x_batch.shape[-1]
            )

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss_val += loss.item()

            val_trues.append(y_batch.cpu())
            val_preds.append(y_pred.cpu())

        correct_pred_val += count_correct(y_pred, y_batch)
        total_pred_val += y_batch.shape[0]

    print(
        f"Validataion loss: {total_loss_val}, Validation accuracy : {correct_pred_val/total_pred_val}"
    )
    print(f"Validation loss normalized: {total_loss_val/len(val_data_loader)}")
    return (
        total_loss_train / len(train_data_loader),
        correct_pred_train / total_pred_train,
        total_loss_val / len(val_data_loader),
        correct_pred_val / total_pred_val,
    )


def train(
    model,
    train_data_loader,
    val_data_loader,
    loss_fn,
    optimiser,
    epochs,
    writer_path,
    model_dave_dir,
    do_augment=False,
    device="cpu",
):
    writer = SummaryWriter(writer_path)
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_loss, train_acc, val_loss, val_acc = train_single_epoch(
            model,
            train_data_loader,
            val_data_loader,
            loss_fn,
            optimiser,
            device,
            do_augment=do_augment,
        )
        writer.add_scalar("train/accuracy", train_acc, i)
        writer.add_scalar("train/loss", train_loss, i)
        writer.add_scalar("validation/accuracy", val_acc, i)
        writer.add_scalar("validation/loss", val_loss, i)

        path = os.path.join(model_dave_dir, f"epoch_{i}.pth")
        torch.save(model.state_dict(), path)
        print(f"Saved at {path}")
        print("---------------------------")
    print("Finished training")
    print("---------------------------")


def evaluate(model, eval_data_loader, loss_fn, device):
    print("Evaluating model")
    total_loss = 0.0
    correct_pred = 0.0
    total_pred = 0
    for x_batch, y_batch in tqdm(eval_data_loader):
        with torch.no_grad():
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            x_batch = x_batch.reshape(-1, x_batch.shape[-1])
            x_batch = mel_spectrogram(x_batch)
            x_batch = x_batch.reshape(
                x_batch.shape[0], 1, x_batch.shape[-2], x_batch.shape[-1]
            )

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            correct_pred += count_correct(y_pred, y_batch)
            total_pred += y_batch.shape[0]

            total_loss += loss.item()

    print(
        f"Evaluation loss: {total_loss}, Evaluation accuracy : {correct_pred/total_pred}"
    )
    print(f"Evaluation loss normalized: {total_loss/len(eval_data_loader)}")
    print("---------------------------")


def construct_datasets(train_path, val_path, test_path, audio_dir, batch_size):
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    train_data = CoughDataset(
        train_df,
        audio_dir,
        SAMPLE_RATE,
        NUM_SAMPLES,
        DEVICE,
    )

    val_data = CoughDataset(val_df, audio_dir, SAMPLE_RATE, NUM_SAMPLES, DEVICE)
    test_data = CoughDataset(test_df, audio_dir, SAMPLE_RATE, NUM_SAMPLES, DEVICE)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_model(name, dropout, batch_size):
    name = name.lower()
    if name == "cnn_small":
        return CNN_Small(drop_p=dropout)
    elif name == "cnn_large":
        return CNN_Large(drop_p=dropout)
    elif name == "crnn":
        return CRNN(drop_p=dropout, batch_size=batch_size)
    elif name == "crnna":
        return CRNN_with_Attention(drop_p=dropout, batch_size=batch_size)
    else:
        raise ValueError("Model type not recognized")


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--val_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--audio_dir", type=str, default="audio/")
    parser.add_argument("--model_save_dir", type=str, default="outputs/")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        default="crnna",
        help="Choose from cnn_small, cnn_large, crnn, crnna",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="logdir/temp_log")
    return parser.parse_args()


# if name == main
if __name__ == "__main__":
    config = get_config()
    # construct datasets
    train_path = config.train_file
    val_path = config.val_file
    test_path = config.test_file
    train_dataloader, val_dataloader, test_dataloader = construct_datasets(
        train_path, val_path, test_path, config.audio_dir, config.batch_size
    )

    # construct model and assign it to device
    model = get_model(config.model, config.dropout, config.batch_size)
    model.to(DEVICE)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-5)

    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    train(
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimiser,
        config.epochs,
        config.logdir,
        config.model_save_dir,
        do_augment=False,
        device=DEVICE,
    )

    evaluate(model, val_dataloader, loss_fn, DEVICE)
    evaluate(model, test_dataloader, loss_fn, DEVICE)
