"""
This module generates spectrogram images from audio recordings in the SVD and VOICED datasets.

It reads metadata about which recordings to use, loads the corresponding audio files,
splits them into smaller chunks, and converts each chunk into a spectrogram image with a standardized name.
The generated spectrograms are stored in dataset-specific folders.
"""
from pathlib import Path
from scipy.io import wavfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.converters import wav2spectrogram

# Paths
USED_SVD_LIST_PATH = Path("misc", "used_svd_recordings.csv")
USED_VOICED_LIST_PATH = Path("misc", "used_voiced_recordings.csv")
SVD_PATH = Path("svd_db")
VOICED_PATH = Path("voiced_db")
PATH_SVD_SPECTROGRAMS = Path("spectrograms", "svd")
PATH_VOICED_SPECTROGRAMS = Path("spectrograms", "voiced")

FFT_LEN = 256  # length of fft window (Hamming is used by default)
FFT_OVERLAP = 128  # overlap of ffts windows
SPECTROGRAM_RESOLUTION = (480, 480)  # desired spectrogram resolution as tuple
CHUNKS = 11  # one chunk in the beginning is eliminated, leaving 10 segments -> easier for 9:1 split ratio
VOICED_SAMPLERATE = 8000 # Since VOICED audio is saved in a txt format and cannot be read when loading the file, its
                         # sample rate is defined by a constant

def create_spectrograms_from_SVD():
    """
    Generate spectrograms for the SVD dataset recordings.

    This function:
    - Reads the list of used SVD recordings from a CSV file.
    - Sorts them by patient and recording IDs.
    - Loads each WAV file.
    - Splits it into predefined number of chunks (removing the first chunk if `CHUNKS > 1`).
    - Converts each chunk into a spectrogram and saves it as a PNG file.

    Output files are saved in `PATH_SVD_SPECTROGRAMS` with filenames that include:
    patient ID, state, recording ID, order, and chunk index.

    Raises
    ------
    FileNotFoundError
       If an expected WAV file does not exist in the dataset folder.
    """
    print("Transforming files SVD wav files to txt files...")
    data = pd.read_csv(USED_SVD_LIST_PATH).sort_values(["patient_id", "recording_id"])
    data["order"] = data.groupby(["patient_id", "state"]).cumcount().apply(lambda x: f"{x:0>4}")
    print(data.head())
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        # Defining the source path and the destination path stem
        # SVD samples are named 'XXXX-a_n.wav' where 'XXXX' represents the session (recording) ID, 'a' the /a:/ sound,
        # and 'n' the recording in a neutral pitch
        source_file_path = SVD_PATH.joinpath(f'{row["recording_id"]}-a_n.wav')
        # The spectrogram name encodes the following:
        #  - patient ID:    unique identifier of a patient in the DB
        #  - health state:  healthy if no pathology or "Sängerstimme" appears in the pathology description, otherwise
        #                   pathological
        #  - record ID:     session ID -> unique identifier of a recording
        #  - order:         information showing the chronology of recordings of the same patient and health state
        #                   e.g.: order 0001 states that this is the second recording for this specific combination of
        #                   a patient-health state
        destination_file_path_stem = (f'svd_patient_{row["patient_id"]:0>4}_{row["state"]}_record_'
                                      f'{row["recording_id"]:0>4}_order_{row["order"]}')

        # Loading the file
        if source_file_path.exists():
            sample_rate, samples = wavfile.read(source_file_path)
        else:
            # Raising an error if a file is missing for some reason (should not happen though as the list of files is
            # checked prior calling this function
            print(f"File {str(source_file_path)} not found")
            raise FileNotFoundError

        # Segmenting the file to chunks
        if CHUNKS > 1:
            wav_chunks = np.array_split(samples, CHUNKS)
            wav_chunks.pop(0)  # to remove silence that can occur in the beginning of the audio
        else:
            wav_chunks = np.array_split(samples, CHUNKS)

        # Generating spectrograms from chunks and saving them in the spectrograms\svd folder
        for chunk_idx, wav_chunk in enumerate(wav_chunks):
            destination_file_path = PATH_SVD_SPECTROGRAMS.joinpath(f"{destination_file_path_stem}_segment_{chunk_idx:04d}.png")
            wav2spectrogram(wav_chunk, destination_file_path, FFT_LEN, FFT_OVERLAP,
                            SPECTROGRAM_RESOLUTION, samplerate=sample_rate)


def create_spectrograms_from_VOICED():
    """
    Generate spectrograms for the VOICED dataset recordings.

    This function:
    - Reads the list of used VOICED recordings from a CSV file.
    - Sorts them by patient and recording IDs.
    - Loads each recording from a `.txt` file (raw audio samples).
    - Splits it into predefined number of chunks (removing the first chunk if `CHUNKS > 1`).
    - Converts each chunk into a spectrogram and saves it as a PNG file.

    VOICED recordings use a fixed sample rate (`VOICED_SAMPLERATE`).

    Output files are saved in `PATH_VOICED_SPECTROGRAMS` with filenames that include:
    patient ID, state, recording ID, order, and chunk index.

    Raises
    ------
    FileNotFoundError
       If an expected text-based audio file does not exist in the dataset folder.
    """
    data = pd.read_csv(USED_VOICED_LIST_PATH).sort_values(["patient_id", "recording_id"])
    data["order"] = data.groupby(["patient_id", "state"]).cumcount().apply(lambda x: f"{x:0>4}")
    print(data.head())
    sample_rate = VOICED_SAMPLERATE
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        # Defining the source path and the destination path stem
        # VOICED samples are named 'voiceXXX.txt' where 'XXX' represents the recording ID
        source_file_path = VOICED_PATH.joinpath(f'voice{row["recording_id"]:0>3}.txt')
        # The spectrogram name encodes the following:
        #  - patient ID:    unique identifier of a patient in the DB -> since VOICED contains one recording per patient,
        #                   the recording ID is used
        #  - health state:  healthy if healthy appears in the pathology description, otherwise
        #                   pathological
        #  - record ID:     unique identifier of a recording
        #  - order:         information showing the chronology of recordings of the same patient and health state
        #                   it is set to 0000 for VOICED recordings as VOICED contains one recording per patient
        destination_file_path_stem = (f'voiced_patient_{row["patient_id"]:0>4}_{row["state"]}_record_'
                                      f'{row["recording_id"]:0>4}_order_{row["order"]}')

        # Loading the file
        if source_file_path.exists():
            samples = np.loadtxt(source_file_path)
        else:
            # Raising an error if a file is missing for some reason (should not happen though as the list of files is
            # checked prior calling this function
            print(f"File {str(source_file_path)} not found")
            raise FileNotFoundError

        # Segmenting the file to chunks
        if CHUNKS > 1:
            wav_chunks = np.array_split(samples, CHUNKS)
            wav_chunks.pop(0)  # to remove silence that can occur in the beginning of the audio
        else:
            wav_chunks = np.array_split(samples, CHUNKS)

        # Generating spectrograms from chunks and saving them in the spectrograms\voiced folder
        for chunk_idx, wav_chunk in enumerate(wav_chunks):
            destination_file_path = PATH_VOICED_SPECTROGRAMS.joinpath(
                f"{destination_file_path_stem}_segment_{chunk_idx:04d}.png")
            wav2spectrogram(wav_chunk, destination_file_path, FFT_LEN, FFT_OVERLAP,
                            SPECTROGRAM_RESOLUTION, samplerate=sample_rate)