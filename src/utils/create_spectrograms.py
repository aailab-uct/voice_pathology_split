from pathlib import Path
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

from src.utils.converters import wav2spectrogram, txt2wav



USED_SVD_LIST_PATH = Path("misc", "used_svd_recordings.csv")
USED_VOICED_LIST_PATH = Path("misc", "used_voiced_recordings.csv")
print(USED_VOICED_LIST_PATH.absolute())
SVD_PATH = Path("svd_db")
VOICED_PATH = Path("voiced_db")
PATH_SVD_SPECTROGRAMS = Path("spectrograms", "svd")
PATH_VOICED_SPECTROGRAMS = Path("spectrograms", "voiced")
VOICED_SAMPLERATE = 8000
# scaler = MinMaxScaler((-1, 1))

fft_len = 256  # length of fft window (Hamming is used by default)
fft_overlap = 128  # overlap of ffts windows
spectrogram_resolution = (480, 480)  # desired spectrogram resolution as tuple
octaves = []  # specify octave filters if needed, see utilities.octave_filter_bank.py for details
chunks = 11  # each wav file is split into multiple chunks, set the number of chunks
sample_rate = 50000

def create_spectrograms_from_SVD():
    print("Transforming files SVD wav files to txt files...")
    data = pd.read_csv(USED_SVD_LIST_PATH).sort_values(["patient_id", "recording_id"])
    data["order"] = data.groupby(["patient_id", "state"]).cumcount().apply(lambda x: f"{x:0>4}")
    print(data.head())
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        # Defining the source path and the destination path stem
        source_file_path = SVD_PATH.joinpath(f'{row["recording_id"]}-a_n.wav')
        destination_file_path_stem = (f'svd_patient_{row["patient_id"]:0>4}_{row["state"]}_record_'
                                      f'{row["recording_id"]:0>4}_order_{row["order"]}')

        # Loading the file
        if source_file_path.exists():
            sample_rate, samples = wavfile.read(source_file_path)
        else:
            print(f"File {str(source_file_path)} not found")
            raise FileNotFoundError

        # Segmenting the file to chunks
        if chunks > 1:
            wav_chunks = np.array_split(samples, chunks)
            wav_chunks.pop(0)  # to remove bad data at start
        else:
            wav_chunks = np.array_split(samples, chunks)

        # Generating spectrograms from chunks and saving them in the sepctrograms\svd folder
        for chunk_idx, wav_chunk in enumerate(wav_chunks):
            destination_file_path = PATH_SVD_SPECTROGRAMS.joinpath(f"{destination_file_path_stem}_segment_{chunk_idx:04d}.png")
            wav2spectrogram(wav_chunk, destination_file_path, fft_len, fft_overlap,
                            spectrogram_resolution, samplerate=sample_rate)


def create_spectrograms_from_VOICED():
    data = pd.read_csv(USED_VOICED_LIST_PATH).sort_values(["patient_id", "recording_id"])
    data["order"] = data.groupby(["patient_id", "state"]).cumcount().apply(lambda x: f"{x:0>4}")
    print(data.head())
    sample_rate = VOICED_SAMPLERATE
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        # Defining the source path and the destination path stem
        source_file_path = VOICED_PATH.joinpath(f'voice{row["recording_id"]:0>3}.txt')
        destination_file_path_stem = (f'voiced_patient_{row["patient_id"]:0>4}_{row["state"]}_record_'
                                      f'{row["recording_id"]:0>4}_order_{row["order"]}')

        # Loading the file

        if source_file_path.exists():
            samples = np.loadtxt(source_file_path)
        else:
            print(f"File {str(source_file_path)} not found")
            raise FileNotFoundError

        # Segmenting the file to chunks
        if chunks > 1:
            wav_chunks = np.array_split(samples, chunks)
            wav_chunks.pop(0)  # to remove bad data at start
        else:
            wav_chunks = np.array_split(samples, chunks)

        # Generating spectrograms from chunks and saving them in the sepctrograms\svd folder
        for chunk_idx, wav_chunk in enumerate(wav_chunks):
            destination_file_path = PATH_VOICED_SPECTROGRAMS.joinpath(
                f"{destination_file_path_stem}_segment_{chunk_idx:04d}.png")
            wav2spectrogram(wav_chunk, destination_file_path, fft_len, fft_overlap,
                            spectrogram_resolution, samplerate=sample_rate)