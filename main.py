
# Utility functions for generating spectrograms from SVD recordings, if missing, afor preparation of datasets for each
# scenario, and for running the classification on all datasets
from src.utils.create_spectrograms import create_spectrograms
# from src import split_all, classification_runner
from pathlib import Path
import pandas as pd
used_recordings = set(pd.read_csv(Path("misc", "used_recordings.csv")).recording_id.tolist())
found_recordings = set(map(lambda x: int(x.name.split("-")[0]), Path("svd_db").glob("*.wav")))
found_recordings.remove(1555)
found_recordings.add(111111111111111)
len(used_recordings - found_recordings) == 0

PATH_TO_SVD_AUDIO = Path("svd_db")
PATH_TO_SPECTROGRAMS = Path("spectrograms")
PATH_USED_SVD_FILES = Path("misc", "used_recordings.csv")

# Generate spectrograms from SVD recordings if not existing
if not PATH_TO_SPECTROGRAMS.joinpath("svd").exists():
    if PATH_TO_SVD_AUDIO.exists():
        print("Folder with SVD spectrograms not found, generating spectrograms from SVD audio...")
        used_recordings = set(pd.read_csv(PATH_USED_SVD_FILES).recording_id.tolist())
        found_recordings = set(map(lambda x: int(x.name.split("-")[0]), PATH_TO_SVD_AUDIO.glob("*.wav")))
        missing_recordings = found_recordings - used_recordings
        if len(missing_recordings) == 0:
            create_spectrograms()
        else:
            print("The following recordings were not found in the database:",
                  ", ".join(map(lambda x: f"{x}-a_n.wav", missing_recordings)))
            raise FileNotFoundError
    else:
        print("Folder with SVD audio files not found. Please, place all SVD audio recordings with /a:/ sound in "
              "neutral pitch in the ")
        raise FileNotFoundError