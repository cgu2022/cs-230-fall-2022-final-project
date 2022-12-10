# cs-230-fall-2022-final-project

## Classifying Cough Audio with CNN, CRNN, and CRNN with Attention
This project aims to classify cough audio using three different deep learning models: a convolutional neural network (CNN), a convolutional recurrent neural network (CRNN), and a CRNN with attention.

# Data
The dataset used in this project consists of cough recordings from Coughvid. 

# Models
The following three models have been trained and evaluated on the dataset:

CNN: The CNN model consists of several convolutional and pooling layers, followed by a dense layer for classification.

CRNN: The CRNN model is similar to the CNN model, but includes a recurrent layer (e.g. LSTM or GRU) to capture temporal information in the audio data.

CRNN with Attention: The CRNN with attention model is similar to the CRNN model, but includes an attention layer to weight different parts of the input sequence and improve the model's performance.

# Usage
To use the trained models, follow these steps:

Clone this repository and navigate to the project directory.
``` git clone https://github.com/cgu2022/cs-230-fall-2022-final-project.git ```

Install the required packages.
``` pip install -r requirements.txt ```

Run the main.py script, specifying the audio file and the model to use. For example, to use the CRNN model on the cough.wav audio file:

```
main.py --train_file='path/train.parquet' \
        --val_file='path/val.parquet' \
        --test_file='path/test.parquet' \
        --audio_dir='path_to_wav_files' \
```

## Python Notebooks

Data_exp
- This section will contain explolatory & data processing Ipython notebooks.
- Run the audio_processing, bad_file_remover, and Data_splitting notebooks inr order to format the COUGHVID dataset for our mode

Model_exp
- Model experiment Ipython notebooks. Leftover from development.
