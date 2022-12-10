import torch
from torch import nn


class CRNN_with_Attention(torch.nn.Module):
    def __init__(self, batch_size=16, drop_p=0.0, device="cpu"):
        super().__init__()
        channels = [1, 16, 32, 64, 128]
        # hidden_state = [2]
        # 4 conv blocks / flatten / linear / softmax

        self.num_layers = 1
        self.hidden_size = 256
        self.embed_size = self.hidden_size
        self.source_sequence_length = 31
        self.target_sequence_length = 1
        self.batch_size = batch_size
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(channels[1]),
            nn.Dropout(p=drop_p),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(channels[2]),
            nn.Dropout(p=drop_p),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[2],
                out_channels=channels[3],
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(channels[3]),
            nn.Dropout(p=drop_p),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[3],
                out_channels=channels[4],
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(channels[4]),
            nn.Dropout(p=drop_p),
        )

        # LSTM Section
        self.LSTM_model = nn.LSTM(
            input_size=1024, hidden_size=256, bidirectional=True, batch_first=True
        )
        self.Tanh = nn.Tanh()

        # Attention Section
        self.embed_size = (
            self.hidden_size * 2
        )  # embed size is 512 because our LSTM is bidirectional (need to multiply by two because hidden_size=256 and bidirectional)

        self.linear_query = nn.Linear(self.embed_size, self.embed_size)
        self.linear_key = nn.Linear(self.embed_size, self.embed_size)
        self.linear_value = nn.Linear(self.embed_size, self.embed_size)

        # Attention Section - Initializing Weights
        nn.init.xavier_normal_(self.linear_query.weight)
        nn.init.zeros_(self.linear_query.bias)

        nn.init.xavier_normal_(self.linear_key.weight)
        nn.init.zeros_(self.linear_key.bias)

        nn.init.xavier_normal_(self.linear_key.weight)
        nn.init.zeros_(self.linear_key.bias)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_size, num_heads=4, dropout=0.0
        )

        # Transition from attention to FC

        self.linear_temporal = nn.Sequential(
            nn.Linear(self.source_sequence_length, 1), nn.ReLU()
        )

        # For Fully Connected Layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.embed_size, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_data):
        # nomralization
        """=std = input_data.std()
        input_data -= input_data.mean()
        input_data /= std"""

        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # td = self.flatten(x)

        # Preprocessing for the LSTM
        td = torch.swapaxes(
            x, 1, 3
        )  # [batch_size, 128, 8, 31]  --> [batch_size, 31, 8, 128]
        td = torch.reshape(
            td, (td.shape[0], 31, -1)
        )  # 128 channels * 8 height of each channel = 1024

        # LSTM Section - Preparing for the LSTM
        self.h_0 = nn.Parameter(
            torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size).to(
                self.device
            )
        )  # hidden state
        self.c_0 = nn.Parameter(
            torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size).to(
                self.device
            )
        )  # internal state

        # LSTM
        LSTM_output, (hn, cn) = self.LSTM_model(
            td, (self.h_0, self.c_0)
        )  # lstm with input, hidden, and internal state
        LSTM_output = self.Tanh(LSTM_output)
        # LSTM_Output: (31, 512)

        # Attention
        Query = self.linear_query(LSTM_output)
        Key = self.linear_key(LSTM_output)
        Value = self.linear_value(LSTM_output)
        attn_output, attn_output_weights = self.multihead_attn(Query, Key, Value)

        # LSTM_output --> (512, 1)
        attn_output = torch.swapaxes(attn_output, 1, 2)
        crunch = self.linear_temporal(attn_output)

        # Final Fully Connnected Layer
        x = self.flatten(crunch)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
