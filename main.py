"""
Main experiment runner for segmentation leakage study.

This script manages the **entire experimental pipeline**:
  1. Ensures spectrograms are available for both SVD and VOICED datasets.
     - If missing, generates them from raw database files.
  2. Randomly samples external validation recordings.
  3. Creates multiple dataset splits (scenarios 1–6) simulating different
     segmentation leakage conditions.
  4. Trains YOLOv8 models on each dataset scenario and evaluates
     them on validation and test (external validation) subsets.
  5. Saves all results to CSV for further analysis.

Workflow Summary
----------------
- **Spectrogram generation**:
    * Checks if spectrogram folders exist for SVD and VOICED.
    * If missing, generates PNG spectrograms from WAV/TXT recordings.
- **External validation sampling**:
    * Randomly selects 10 healthy + 10 pathological recordings for each dataset.
    * These are held out as *true* test sets.
- **Scenario dataset preparation**:
    * Calls splitting functions to create scenarios 1-4 (both datasets) and scenarios 5-6 (SVD).
- **Training & evaluation**:
    * Runs YOLOv8 classification for all prepared datasets.
    * Evaluates on validation plus test (external validation) splits, storing metrics.

Constants
---------
PATH_TO_SVD_AUDIO : Path to the SVD dataset audio folder containing WAV files.
PATH_TO_VOICED_AUDIO : Path to the VOICED dataset audio folder containing TXT/DAT files.
PATH_TO_SPECTROGRAMS : Root folder where spectrograms will be stored.
PATH_USED_SVD_FILES : CSV file listing selected SVD recordings for the experiment.
PATH_USED_VOICED_FILES : CSV file listing selected VOICED recordings for the experiment.

Output
------
- Spectrograms in `spectrograms/svd` and `spectrograms/voiced`
- Dataset scenarios in `datasets/`
- CSV lists of dataset mappings in `dataset_lists/`
- Model training results saved by YOLOv8 and final CSV metrics
"""


# Utility functions for generating spectrograms, preparing datasets, and running classification
from src.utils.create_spectrograms import create_spectrograms_from_SVD, create_spectrograms_from_VOICED
from src.split_all import (sample_ext_validation, split_scenario_1, split_scenario_2, split_scenario_3,
                           split_scenario_4_voiced_5_svd, split_scenario_4_svd_6
                           )
from src.classification_runner import run_experiments
# Additional libraries
from pathlib import Path
import pandas as pd

# === CONSTANT PATHS ===
PATH_TO_SVD_AUDIO = Path("svd_db")             # Location of SVD audio WAVs
PATH_TO_VOICED_AUDIO = Path("voiced_db")       # Location of VOICED audio files
PATH_TO_SPECTROGRAMS = Path("spectrograms")    # Where generated spectrogram PNGs are stored
PATH_USED_SVD_FILES = Path("misc", "used_svd_recordings.csv")     # Metadata of selected SVD recordings
PATH_USED_VOICED_FILES = Path("misc", "used_voiced_recordings.csv") # Metadata of selected VOICED recordings


if __name__ == "__main__":
    # Generate spectrograms for SVD recordings if missing
    if not PATH_TO_SPECTROGRAMS.joinpath("svd").exists():
        if PATH_TO_SVD_AUDIO.exists():
            print("Folder with SVD spectrograms not found, generating spectrograms from SVD audio...")
            # Load expected recording IDs from CSV
            used_recordings = set(pd.read_csv(PATH_USED_SVD_FILES).recording_id.tolist())
            # Find available recordings in SVD folder
            found_recordings = set(map(lambda x: int(x.name.split("-")[0]), PATH_TO_SVD_AUDIO.glob("*.wav")))
            # Determine which required recordings are missing
            missing_recordings = used_recordings - found_recordings

            if len(missing_recordings) == 0:
                # All required recordings found -> create spectrogram folder & generate PNGs
                PATH_TO_SPECTROGRAMS.joinpath("svd").mkdir(parents=True)
                create_spectrograms_from_SVD()
            else:
                # Missing recordings detected -> stop
                print("The following recordings were not found in the database:",
                      ", ".join(map(lambda x: f"{x}-a_n.wav", missing_recordings)))
                raise FileNotFoundError
        else:
            # SVD audio folder itself missing -> cannot proceed, stop
            print("Folder with SVD audio files not found. Please, place all SVD audio recordings with /a:/ sound in "
                  "neutral pitch in", PATH_TO_SVD_AUDIO)
            raise FileNotFoundError

    # Generate spectrograms for VOICED recordings if missing (they should not as they are a part of the repository)
    if not PATH_TO_SPECTROGRAMS.joinpath("voiced").exists():
        if PATH_TO_VOICED_AUDIO.exists():
            print("Folder with VOICED spectrograms not found, generating spectrograms from VOICED audio...")
            # Load expected recording IDs from CSV
            used_recordings = set(pd.read_csv(PATH_USED_VOICED_FILES).recording_id.tolist())
            # Find available recordings in SVD folder
            found_recordings = set(map(lambda x: int(x.stem[-3:]), PATH_TO_VOICED_AUDIO.glob("*.dat")))
            # Determine which required recordings are missing
            missing_recordings = found_recordings - used_recordings

            if len(missing_recordings) == 0:
                # All required recordings found -> create spectrogram folder & generate PNGs
                PATH_TO_SPECTROGRAMS.joinpath("voiced").mkdir(parents=True)
                create_spectrograms_from_VOICED()
            else:
                # Missing recordings detected -> stop
                print("The following recordings were not found in the database:",
                      ", ".join(map(lambda x: f"voice{x}.txt", missing_recordings)))
                raise FileNotFoundError
        else:
            # SVD audio folder itself missing -> cannot proceed, stop
            print("Folder with VOICED audio files not found. Please, place all VOICED audio recordings in",
                  PATH_TO_VOICED_AUDIO)
            raise FileNotFoundError

    # Sample "external validation" recordings
    #  - select 10 healthy and 10 pathological recordings per dataset which will be placed in the test set only
    ext_val_samples = sample_ext_validation()

    # Prepare datasets for scenarios 1-3, 4 (VOICED), and 5 (SVD)
    # for i in range(10):
    #     for db in ["svd", "voiced"]:
    #         # Scenario 1 -> fixed 5th segment (0004) for val/test
    #         # split_scenario_1(db, ext_val_samples)
    #         # Scenario 2 -> one random segment per recording for val
    #         # split_scenario_2(db, ext_val_samples)
    #         # Scenario 3 -> fully random segment split
    #         # split_scenario_3(db, ext_val_samples)
    #         # Scenario 4 (VOICED) / 5 (SVD) -> patient/record-oriented split
    #         split_scenario_4_voiced_5_svd(db, ext_val_samples, i)

    # Prepare datasets for scenarios 4 (SVD) and 6
    #  - exclude_duplicates = False -> patient-level with all available recordings (Scenario 4)
    #  - exclude_duplicates = True -> only oldest healthy & pathological recording per patient (Scenario 6)
    # for boolean in [True, False]:
    #     split_scenario_4_svd_6(ext_val_samples, exclude_duplicates=boolean)

    # Run YOLOv8 classification experiments on all scenarios
    run_experiments()