
# Utility functions for generating spectrograms from SVD recordings, if missing, afor preparation of datasets for each
# scenario, and for running the classification on all datasets
from src.utils.create_spectrograms import create_spectrograms_from_SVD, create_spectrograms_from_VOICED
# from src import split_all, classification_runner
from pathlib import Path
import pandas as pd


PATH_TO_SVD_AUDIO = Path("svd_db")
PATH_TO_VOICED_AUDIO = Path("voiced_db")
PATH_TO_SPECTROGRAMS = Path("spectrograms")
PATH_USED_SVD_FILES = Path("misc", "used_svd_recordings.csv")
PATH_USED_VOICED_FILES = Path("misc", "used_voiced_recordings.csv")

# Generate spectrograms from SVD recordings if not existing
if not PATH_TO_SPECTROGRAMS.joinpath("svd").exists():
    if PATH_TO_SVD_AUDIO.exists():
        print("Folder with SVD spectrograms not found, generating spectrograms from SVD audio...")
        used_recordings = set(pd.read_csv(PATH_USED_SVD_FILES).recording_id.tolist())
        found_recordings = set(map(lambda x: int(x.name.split("-")[0]), PATH_TO_SVD_AUDIO.glob("*.wav")))
        missing_recordings = found_recordings - used_recordings
        if len(missing_recordings) == 0:
            create_spectrograms_from_SVD()
        else:
            print("The following recordings were not found in the database:",
                  ", ".join(map(lambda x: f"{x}-a_n.wav", missing_recordings)))
            raise FileNotFoundError
    else:
        print("Folder with SVD audio files not found. Please, place all SVD audio recordings with /a:/ sound in "
              "neutral pitch in", PATH_TO_SVD_AUDIO)
        raise FileNotFoundError

# Generate spectrograms from VOICED recordings if not existing
if not PATH_TO_SPECTROGRAMS.joinpath("voiced").exists():
    if PATH_TO_VOICED_AUDIO.exists():
        print("Folder with VOICED spectrograms not found, generating spectrograms from VOICED audio...")
        used_recordings = set(pd.read_csv(PATH_USED_VOICED_FILES).recording_id.tolist())
        found_recordings = set(map(lambda x: int(x.stem[-3:]), PATH_TO_VOICED_AUDIO.glob("*.dat")))
        missing_recordings = found_recordings - used_recordings
        if len(missing_recordings) == 0:
            create_spectrograms_from_VOICED()
        else:
            print("The following recordings were not found in the database:",
                  ", ".join(map(lambda x: f"voice{x}.txt", missing_recordings)))
            raise FileNotFoundError
    else:
        print("Folder with VOICED audio files not found. Please, place all VOICED audio recordings in",
              PATH_TO_VOICED_AUDIO)
        raise FileNotFoundError