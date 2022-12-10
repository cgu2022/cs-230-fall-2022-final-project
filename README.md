# cs-230-fall-2022-final-project

## Project structure 

### notebooks

Data_exp
- This section will contain explolatory & data processing Ipython notebooks.
- Run the audio_processing, bad_file_remover, and Data_splitting notebooks inr order to format the COUGHVID dataset for our mode

Model_exp
- Model experiment Ipython notebooks. Leftover from development.

## How to use

main.py --train_file='path/train.parquet' \
        --val_file='path/val.parquet' \
        --test_file='path/test.parquet' \
        --audio_dir='path_to_wav_files' \
