"""
Script for generating datasets for each scenario:
 - Scenario 1 (Controlled partitioning with chosen segment): The fifth segment (0004) from each recording is placed in the test set
 - Scenario 2 (Controlled partitioning with random segment): One random segment from each recording is placed in the test set
 - Scenario 3 (Fully random partitioning): Fully randomized split with train-test ratio = 0.9 (so 1/10 of segments is in the test set)
 - Scenario 4 (Patient-oriented random partitioning): All segments of all recordings belonging to a signle patient are placed either in the train or test set. The train-test ratio is kept close but not exactly equal to 0.9 as the recordings are shuffled randomly.
For the SVD dataset only, two more datasets are generated:
 - Scenario 5 (Record-oriented random partitioning): Each recording, with all its segments, is randomly allocated to either the train or test set. This differs from Scenario 4 as some patients in SVD were recorded multiple times -> potential introduction of data leakage
 - Scenario 6 (Patient-oriented random partitioning with a single healthy and pathological recording): Patients, along with their oldest healthy and pathological recordings, are randomly assigned to either the training -> checking if there is any influence of involving multiple recordings of the same patients during the training.
or test set.
"""

import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np

PATH_USED_SVD_FILES = Path("misc", "used_svd_recordings.csv")
PATH_USED_VOICED_FILES = Path("misc", "used_voiced_recordings.csv")
PATH_DATASET_LISTS = Path("dataset_lists")
PATH_DATASET_LISTS.mkdir(exist_ok=True)

# Creating the dataset folder
PATH_DATASETS = Path("datasets")
if PATH_DATASETS.exists():
    shutil.rmtree(PATH_DATASETS)
PATH_DATASETS.mkdir()

# Creating the subfolders for each scenario
dbs = ["svd", "voiced"]
for db in dbs:
    max_scenario = 7 if db == "svd" else 5
    for i in range(1, max_scenario):
        scenario_folder = PATH_DATASETS.joinpath(f"{db}_scenario_{i}")
        for subset in ["train", "test", "val"]:
            for health_state in ["healthy", "pathological"]:
                scenario_folder.joinpath(subset, health_state).mkdir(parents=True)

def sample_ext_validation():
    # Select 10 patients with a single recording for each health state and each dataset
    used_svd_recordings = pd.read_csv(PATH_USED_SVD_FILES)
    used_voiced_recordings = pd.read_csv(PATH_USED_VOICED_FILES)

    # SVD is problematic as we could accidentally select a recording of a patient with multiple recordings for validation
    # Keeping track of patients with multiple recordings
    used_svd_recordings["duplicity"] = used_svd_recordings["patient_id"].apply(lambda x: used_svd_recordings[used_svd_recordings["patient_id"] == x].shape[0])
    # Filtering out patients with multiple recordings
    svd_recordings_for_val = used_svd_recordings[used_svd_recordings["duplicity"] == 1].copy()
    used_svd_recordings.drop(columns=["duplicity"], inplace=True)

    # Picking 10 random healthy and pathological recordings for "external" validation -> these will use only the 5th segment (0004).
    db_used_recordings = [svd_recordings_for_val, used_voiced_recordings]
    ext_val_samples = {"svd": [], "voiced": []}

    for name, db in zip(ext_val_samples.keys(), db_used_recordings):
        ext_val_samples[name] += db[db["state"] == "pathological"].sample(n=10, random_state=42).recording_id.tolist()
        ext_val_samples[name] += db[db["state"] == "healthy"].sample(n=10, random_state=42).recording_id.tolist()

    return ext_val_samples

def split_scenario_1(db: str, ext_val_samples: dict):
    # Script for scenario 1
    # Load the list of recordings which are not used for
    ext_val_recordings = ext_val_samples[db]
    # Get the list of all spectrograms
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])
    # Extract the name, recording ID, segment number and class as these are necessary for distribution to the dataset subdirectories
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])
    # Decide the subset based on the segment number
    data["subset"] = data["segment"].apply(lambda x: "val" if x == 4 else "train")
    # Change the subset for recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"
    # Delete the other segments of recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]
    # Define the destination path based on the subset information
    data["destination_path"] = data.apply(lambda row: Path("datasets", f"{db}_scenario_1", row["subset"], row["class"], row["name"]), axis=1)
    # Loop through the rows and copy from source path to the destination path
    print(f"Copying data for {db}_scenario_1...", end="\n\n")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])

    # Save the information about the source and destination paths to the dataset_lists folder
    data[["source_path", "destination_path"]].to_csv(PATH_DATASET_LISTS.joinpath(f"{db}_scenario_1.csv"),
                                                     index=False)

# Scenario 2 - random segment from each recording goes to the test set
def split_scenario_2(db: str, ext_val_samples: dict):
    ext_val_recordings = ext_val_samples[db]
    # Get the list of all spectrograms
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])
    # Extract the name, recording ID, segment number and class as these are necessary for distribution to the dataset subdirectories
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])

    # Generate the random number from 0 to 9 (representing a segment) for each recording to decide which segment goes to the test set
    # Create a Generator instance with a specified seed
    unique_recording_ids = data.recording_id.unique().tolist()
    rng = np.random.default_rng(seed=42)
    # Generate random integers using the Generator instance
    random_segments = rng.integers(low=0, high=10, size=len(unique_recording_ids))
    segmentation_data = pd.DataFrame(data=random_segments, columns=["segment"], index=unique_recording_ids)
    # Add the assigned segment to the table with all segments
    data["segment_for_test"] = data["recording_id"].apply(lambda x: segmentation_data.loc[x, "segment"])
    # Decide the subset based on the equality of the chosen and actual segments
    data["subset"] = data.apply(lambda row: "val" if row["segment"] == row["segment_for_test"] else "train", axis=1)
    # Change the subset for recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"
    # Delete the other segments of recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]
    # Define the destination path based on the subset information
    data["destination_path"] = data.apply(
        lambda row: Path("datasets", f"{db}_scenario_2", row["subset"], row["class"], row["name"]), axis=1)

    # Loop through the rows and copy from source path to the destination path
    print(f"Copying data for {db}_scenario_2...")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])

    # Save the information about the source and destination paths to the dataset_lists folder
    data[["source_path", "destination_path"]].to_csv(PATH_DATASET_LISTS.joinpath(f"{db}_scenario_2.csv"),
                                                     index=False)

# Scenario 3 - Random split of all segments in 9:1 ratio
def split_scenario_3(db: str, ext_val_samples: dict):
    ext_val_recordings = ext_val_samples[db]
    # Get the list of all spectrograms
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])
    # Extract the name, recording ID, segment number and class as these are necessary for distribution to the dataset subdirectories
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])
    data["subset"] = None
    # Change the subset for recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"
    # Delete the other segments of recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]
    # Randomly sample 90% of the remaining segments for the training set
    sampled_segments = data.loc[pd.isna(data["subset"]), "source_path"].sample(frac=0.9, random_state=42).tolist()
    # Change the subset for the remaining segments based on whether they are sampled or not
    data.loc[data["source_path"].isin(sampled_segments), "subset"] = "train"
    data.loc[pd.isna(data["subset"]), "subset"] = "val"
    # Define the destination path based on the subset information
    data["destination_path"] = data.apply(lambda row: Path("datasets", f"{db}_scenario_3", row["subset"], row["class"], row["name"]), axis=1)

    # Loop through the rows and copy from source path to the destination path
    print(f"Copying data for {db}_scenario_3...")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])


# Scenarios 4 (VOICED) and 5 (SVD) -> since algorithmically, they are the same
def split_scenario_4_voiced_5_svd(db: str, ext_val_samples: dict):
    ext_val_recordings = ext_val_samples[db]
    # Get the list of all spectrograms
    data = pd.DataFrame(list(Path("spectrograms", db).glob("*.png")), columns=["source_path"])
    # Extract the name, recording ID, segment number and class as these are necessary for distribution to the dataset subdirectories
    data["name"] = data["source_path"].apply(lambda x: x.name)
    data["recording_id"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[5]))
    data["segment"] = data["source_path"].apply(lambda x: int(x.stem.split("_")[-1]))
    data["class"] = data["source_path"].apply(lambda x: x.stem.split("_")[3])
    data["subset"] = None
    # Change the subset for recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] == 4)
    data.loc[mask, "subset"] = "test"
    # Delete the other segments of recordings selected for external validation
    mask = data["recording_id"].isin(ext_val_recordings) & (data["segment"] != 4)
    data = data[~mask]
    # Randomly sample 90% of the remaining recordings for the training set
    unique_recordings = pd.Series(data.loc[pd.isna(data["subset"]), "recording_id"].unique())
    sampled_recordings = unique_recordings.sample(frac=0.9, random_state=42).tolist()
    # Change the subset for the remaining recordings based on whether they are sampled or not
    data.loc[data["recording_id"].isin(sampled_recordings), "subset"] = "train"
    data.loc[pd.isna(data["subset"]), "subset"] = "val"
    # Define the destination path based on the subset information
    scenario_number = 4 if db == "voiced" else 5
    data["destination_path"] = data.apply(lambda row: Path("datasets", f"{db}_scenario_{scenario_number}", row["subset"], row["class"], row["name"]), axis=1)

    # Loop through the rows and copy from source path to the destination path
    print(f"Copying data for {db}_scenario_{scenario_number}...")
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        shutil.copy(src=row["source_path"], dst=row["destination_path"])
