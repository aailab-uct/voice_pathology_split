"""
Dataset generation script for segmentation leakage experiments.

This script organizes spectrogram images into multiple dataset splits to simulate
different segmentation leakage scenarios. It generates controlled, random, and patient-oriented
splits for both SVD and VOICED datasets, with optional external validation samples.

External validation:
--------------------
- Randomly selects 10 healthy and 10 pathological recordings for each dataset
  to serve as an *external* test set.
- For these, only the 5th segment (0004) is retained, others are excluded.

Scenarios
---------
1. Controlled partitioning with the chosen segment:
   - Always selects the 5th segment (0004) for validation.
2. Controlled partitioning with a random segment:
   - Randomly selects one segment per recording for validation.
3. Fully random partitioning:
   - Randomly splits all segments into a 9:1 train/val ratio.
4. Patient-oriented random partitioning:
   - All segments from a patient are placed either in train or val.
5. Record-oriented random partitioning (SVD only):
   - Entire recordings (with all segments) are randomly assigned to train or val.
6. Patient-oriented with single healthy and pathological recording (SVD only):
   - For each patient, only their oldest healthy and pathological recordings are used, then patient-level split
     is applied.

Constants
---------
PATH_USED_SVD_FILES: CSV file listing which SVD recordings are used.
PATH_USED_VOICED_FILES: CSV file listing which VOICED recordings are used.
PATH_DATASET_LISTS: Folder where source -> destination mappings for each scenario are stored.
PATH_DATASETS: Main output folder containing organized dataset scenarios.

Outputs
-------
- Organized dataset folders under `datasets/`
- CSV mapping files in `dataset_lists/` for reproducibility
"""

import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np

# Paths to metadata files listing which recordings are used
PATH_USED_SVD_FILES = Path("misc", "used_svd_recordings.csv")
PATH_USED_VOICED_FILES = Path("misc", "used_voiced_recordings.csv")

# Where the resulting dataset lists (source -> destination mappings) are stored
PATH_DATASET_LISTS = Path("dataset_lists")
PATH_DATASET_LISTS.mkdir(exist_ok=True)

# Prepare main dataset output folder (clean start)
PATH_DATASETS = Path("datasets")
if PATH_DATASETS.exists():
    shutil.rmtree(PATH_DATASETS)
PATH_DATASETS.mkdir()

# Create basic subfolder structure for each scenario in both SVD and VOICED
dbs = ["svd", "voiced"]
for db in dbs:
    max_scenario = 7 if db == "svd" else 5
    for i in range(1, max_scenario):
        scenario_folder = PATH_DATASETS.joinpath(f"{db}_scenario_{i}")
        for subset in ["train", "test", "val"]:
            for health_state in ["healthy", "pathological"]:
                scenario_folder.joinpath(subset, health_state).mkdir(parents=True)

def sample_ext_validation():
    """
    Select external validation samples for each dataset.

    Strategy:
        - For each dataset (SVD, VOICED), randomly select 10 pathological + 10 healthy recordings.
        - For SVD, only patients with a single recording are eligible
          (avoids accidental leakage from patients with multiple recordings).

    Returns
    -------
    ext_val_samples : dict
        Dictionary:
            {
                "svd": [list of selected recording IDs],
                "voiced": [list of selected recording IDs]
            }
    """
    # Load metadata for both datasets
    used_svd_recordings = pd.read_csv(PATH_USED_SVD_FILES)
    used_voiced_recordings = pd.read_csv(PATH_USED_VOICED_FILES)

    # Identify SVD patients with multiple recordings and exclude them
    used_svd_recordings["duplicity"] = used_svd_recordings["patient_id"].apply(
        lambda x: used_svd_recordings[used_svd_recordings["patient_id"] == x].shape[0]
    )
    svd_recordings_for_val = used_svd_recordings[used_svd_recordings["duplicity"] == 1].copy()
    used_svd_recordings.drop(columns=["duplicity"], inplace=True)

    # Pick 10 healthy + 10 pathological recordings for external validation
    db_used_recordings = [svd_recordings_for_val, used_voiced_recordings]
    ext_val_samples = {"svd": [], "voiced": []}

    for name, db in zip(ext_val_samples.keys(), db_used_recordings):
        ext_val_samples[name] += db[db["state"] == "pathological"].sample(n=10, random_state=42).recording_id.tolist()
        ext_val_samples[name] += db[db["state"] == "healthy"].sample(n=10, random_state=42).recording_id.tolist()

    return ext_val_samples

def split_scenario_1(db: str, ext_val_samples: dict):
    """
    Scenario 1: Controlled split, always using the 5th segment (0004).

    - For every recording:
        * Segment 0004 -> validation set
        * Remaining segments -> training set
    - External validation handling:
        * Only the 5th segment (0004) from selected external recordings goes into the test set.
        * All other segments from those recordings are removed.

    Parameters
    ----------
    db : str
        Dataset name ("svd" or "voiced").
    ext_val_samples : dict
        Dictionary of external validation recording IDs per dataset.

    Notes
    -----
    - 90:10 Train/val ratio is fixed by sampling a specific single segment of each recording in the validation set.
    - The test set size is fixed by the external validation selection.
    """
    ext_val_recordings = ext_val_samples[db]

    # List all spectrograms for this dataset
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])

    # Extract metadata from filename
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])

    # By default, segment 4 -> validation, others -> train
    data["subset"] = data["segment"].apply(lambda x: "val" if x == 4 else "train")

    # External validation recordings, segment 4 -> test
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"

    # Remove all other segments from external validation recordings
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]

    # Build destination paths
    data["destination_path"] = data.apply(
        lambda row: Path("datasets", f"{db}_scenario_1", row["subset"], row["class"], row["name"]),
        axis=1
    )

    # Copy files from the source path to the destination path
    print(f"Copying data for {db}_scenario_1...", end="\n\n")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])

    # Save the mapping source -> destination
    data[["source_path", "destination_path"]].to_csv(PATH_DATASET_LISTS.joinpath(f"{db}_scenario_1.csv"),
                                                     index=False)


def split_scenario_2(db: str, ext_val_samples: dict):
    """
    Scenario 2: Controlled split, random segment per recording.

    - Randomly selects exactly one segment (0–9) per recording for validation.
    - External validation handling:
        * Only the 5th segment (0004) from selected external recordings goes into the test set.
        * All other segments from those recordings are removed.

    Parameters
    ----------
    db : str
        Dataset name ("svd" or "voiced").
    ext_val_samples : dict
        Dictionary of external validation recording IDs per dataset.

    Notes
    -----
    - 90:10 Train/val ratio is fixed by sampling a random single segment of each recording in the validation set
      with a reproducible seed (42).
    - The test set size is fixed by the external validation selection.
    """
    ext_val_recordings = ext_val_samples[db]

    # List all spectrograms for this dataset
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])

    # Extract metadata from filename
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])

    # Randomly assign one "validation segment" per recording
    unique_recording_ids = data.recording_id.unique().tolist()
    rng = np.random.default_rng(seed=42)

    # high is set to 10 -> random number from 0-9 -> corresponds with segment numbers
    random_segments = rng.integers(low=0, high=10, size=len(unique_recording_ids))
    segmentation_data = pd.DataFrame(data=random_segments, columns=["segment"], index=unique_recording_ids)

    # Add the assigned segment to the table with all segments
    data["segment_for_test"] = data["recording_id"].apply(lambda x: segmentation_data.loc[x, "segment"])

    # Assign subset based on equality with chosen random segment
    data["subset"] = data.apply(lambda row: "val" if row["segment"] == row["segment_for_test"] else "train", axis=1)

    # External validation recordings, segment 4 -> test
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"

    # Remove all other segments from external validation recordings
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]

    # Build destination paths
    data["destination_path"] = data.apply(
        lambda row: Path("datasets", f"{db}_scenario_2", row["subset"], row["class"], row["name"]), axis=1)

    # Copy files from the source path to the destination path
    print(f"Copying data for {db}_scenario_2...")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])

    # Save the mapping source -> destination
    data[["source_path", "destination_path"]].to_csv(PATH_DATASET_LISTS.joinpath(f"{db}_scenario_2.csv"),
                                                     index=False)

# Scenario 3 - Random split of all segments in 9:1 ratio
def split_scenario_3(db: str, ext_val_samples: dict):
    """
    Scenario 3: Fully random partitioning of all segments (9:1 ratio).

    - Randomly assigns ~90% of all segments to the training set
     and the remaining ~10% to the validation set.
    - External validation handling:
        * Only the 5th segment (0004) from selected external recordings goes into the test set.
        * All other segments from those recordings are removed.

    This scenario imitates the regular practice appearing in many articles.

    Parameters
    ----------
    db : str
       Dataset name ("svd" or "voiced").
    ext_val_samples : dict
       Dictionary of external validation recording IDs per dataset.

    Notes
    -----
    - Train/val ratio is fixed by sampling fraction `frac=0.9` with a reproducible seed (42).
    - The test set size is fixed by the external validation selection.
    """
    ext_val_recordings = ext_val_samples[db]

    # List all spectrograms for this dataset
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])

    # Extract metadata from filename
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])

    # Prepare an empty column for subset assignment
    data["subset"] = None

    # External validation override
    #  - external validation recordings, segment 4 -> test
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"

    #  - remove all other segments from external validation recordings
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]

    # Random train-val split
    #  - randomly sample 90% of the remaining segments for the training subset
    sampled_segments = data.loc[pd.isna(data["subset"]), "source_path"].sample(frac=0.9, random_state=42).tolist()

    #  - change the subset for the remaining segments based on random sampling
    data.loc[data["source_path"].isin(sampled_segments), "subset"] = "train"
    data.loc[pd.isna(data["subset"]), "subset"] = "val"

    # Build destination paths
    data["destination_path"] = data.apply(
        lambda row: Path("datasets", f"{db}_scenario_3", row["subset"], row["class"], row["name"]),
        axis=1
    )

    # Copy files from the source path to the destination path
    print(f"Copying data for {db}_scenario_3...")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])

    # Save the mapping source -> destination
    data[["source_path", "destination_path"]].to_csv(PATH_DATASET_LISTS.joinpath(f"{db}_scenario_3.csv"),
                                                     index=False)


def split_scenario_4_voiced_5_svd(db: str, ext_val_samples: dict):
    """
    Scenario 4 (VOICED) and Scenario 5 (SVD): Patient/record-oriented random split.

    - All segments from an entire recording are placed **entirely** in either the training or validation set.
    - For VOICED (Scenario 4), this enforces a **patient-level split** (since VOICED contains single recording
      per patient), ensuring no leakage between train and validation for the same patient.
    - For SVD (Scenario 5), only recordings are randomly assigned, meaning the same patient may appear
      in both train and val if they have multiple recordings -> **possible leakage**.
    - External validation handling:
        * Only the 5th segment (0004) from selected external recordings goes into the test set.
        * All other segments from those recordings are removed.

    Parameters
    ----------
    db : str
        Dataset name ("voiced" -> Scenario 4, "svd" -> Scenario 5).
    ext_val_samples : dict
        Dictionary of external validation recording IDs per dataset.

    Notes
    -----
    - A train/val ratio is fixed by sampling fraction `frac=0.9` to ~9:1 with a reproducible seed (42).
    - The test set size is fixed by the external validation selection.
    """
    ext_val_recordings = ext_val_samples[db]

    # List all spectrograms for this dataset
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])

    # Extract metadata from filename
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])

    # Prepare an empty column for subset assignment
    data["subset"] = None

    # External validation override
    #  - external validation recordings, segment 4 -> test
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"

    #  - remove all other segments from external validation recordings
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]

    # Random train-val split
    #  - randomly sample 90% of the remaining recordings for the training subset
    unique_recordings = pd.Series(data.loc[pd.isna(data["subset"]), "recording_id"].unique())
    sampled_recordings = unique_recordings.sample(frac=0.9, random_state=42).tolist()

    #  - change the subset for the remaining recordings based on random sampling
    data.loc[data["recording_id"].isin(sampled_recordings), "subset"] = "train"
    data.loc[pd.isna(data["subset"]), "subset"] = "val"

    # Build destination paths
    scenario_number = 4 if db == "voiced" else 5
    data["destination_path"] = data.apply(
        lambda row: Path("datasets", f"{db}_scenario_{scenario_number}", row["subset"], row["class"], row["name"]),
        axis=1
    )

    # Copy files from the source path to the destination path
    print(f"Copying data for {db}_scenario_{scenario_number}...")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])

    # Save the mapping source -> destination
    data[["source_path", "destination_path"]].to_csv(PATH_DATASET_LISTS.joinpath(f"{db}_scenario_{scenario_number}.csv"),
                                                     index=False)


def split_scenario_4_svd_6(ext_val_samples: dict, exclude_duplicates=False):
    """
    Scenario 4 (SVD patient-level) and Scenario 6 (SVD single-recording patient-level).

    - Both scenarios enforce a **patient-oriented split**:
        * All segments from a patient are assigned entirely to either train or validation.
    - External validation handling:
        * Only the 5th segment (0004) from selected external recordings goes into the test set.
        * All other segments from those recordings are removed.

    Scenario differences:
    ---------------------
    - Scenario 4 (default, exclude_duplicates=False):
        * Uses **all recordings** of a patient.
        * Patients with multiple recordings can still appear fully in either train or val,
          but leakage risk is minimized since patient-level splitting is applied.
    - Scenario 6 (exclude_duplicates=True):
        * Uses **only the oldest healthy and pathological recordings** for each patient
          (identified by order=0000).
        * Ensures **each patient contributes at most one healthy plus one pathological recording**.

    Parameters
    ----------
    ext_val_samples : dict
        Dictionary of external validation recording IDs.
    exclude_duplicates : bool, optional
        If True, filters out any segments from duplicate recordings,
        retaining only the first (oldest) recording for each patient plus state.
        Default = False → standard patient-level split (Scenario 4).

    Notes
    -----
    - The train/val ratio is ~0.9, but determined dynamically by adding patients
      to the validation set until ~10% of total segments is reached. To allow reproducibility, patient IDs are shuffled
      with a reproducible seed (42).
    - Scenario number is determined by `exclude_duplicates`:
        * False -> scenario 4
        * True -> scenario 6
    """
    ext_val_recordings = ext_val_samples["svd"]

    # List all spectrograms for this dataset
    data = pd.DataFrame(list(Path("spectrograms", "svd").glob("*.png")), columns=["source_path"])

    # (Scenario 6 only): Remove duplicates -> keep only first recording for each patient plus state
    if exclude_duplicates:
        # Extract recording "order" from filename (indicates repeated recordings for the same patient)
        data["order"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-3]))
        # Keep only those with order == 0 (oldest recording)
        mask = data["order"] == 0
        data = data[mask].drop(columns=["order"])
        scenario_number = 6
    else:
        scenario_number = 4

    # Extract metadata from filename
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])
    data["patient_id"] = data["source_path"].apply(lambda x: x.stem.split("_")[2])

    # Prepare an empty column for subset assignment
    data["subset"] = None

    # External validation override
    #  - external validation recordings, segment 4 -> test
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"

    #  - remove all other segments from external validation recordings
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]

    # Random train-val split
    #  - randomly shuffle the list of remaining unique patient IDs
    unique_patients = data.loc[pd.isna(data["subset"]), "patient_id"].unique()

    #  - shuffle patients reproducibly
    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_patients)

    #  - iteratively assign patients to validation set until reaching ~10% ratio
    for patient_id in unique_patients:
        # Assign ALL segments of this patient to "val"
        data.loc[data["patient_id"] == patient_id, "subset"] = "val"
        # Compute current test/total ratio dynamically
        test_ratio = data[data["subset"] == "val"].shape[0] / (
            data[(data["subset"] == "val") | (pd.isna(data["subset"]))].shape[0]
        )
        # Stop when just over 10% of total segments are assigned to val
        if test_ratio > 0.1:
            break

    #  - iteratively assign ALL segments of the remaining patients to training set
    #    (indicated by having an empty subset value)
    data.loc[pd.isna(data["subset"]), "subset"] = "train"

    # Build destination paths
    data["destination_path"] = data.apply(lambda row: Path("datasets", f"svd_scenario_{scenario_number}",
                                                           row["subset"], row["class"], row["name"]), axis=1)

    # Copy files from the source path to the destination path
    print(f"Copying data for svd_scenario_{scenario_number}...")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])

    # Save the mapping source -> destination
    data[["source_path", "destination_path"]].to_csv(PATH_DATASET_LISTS.joinpath(f"svd_scenario_{scenario_number}.csv"),
                                                     index=False)
