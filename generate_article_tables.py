"""
Script for generation of tables which are used in the journal article
"""


import pandas as pd
from pathlib import Path

# Path to datasets and dictionary for renaming dataset folder names to something more suitable for publishing
PATH_TO_DATASETS = Path("datasets")
DATASET_RENAMING = {
    "patients_wise_split_svd_no_duplicities": "patient-wise split without duplicite recordings",
    "patients_wise_split_svd_with_duplicities": "patient-wise split with duplicite recordings",
    "patients_wise_split_voiced": "patient-wise split",
    "segmentation_leakage_random_segment_of_each_recording_svd": "random segment of each recording in the test set",
    "segmentation_leakage_random_segment_of_each_recording_voiced": "random segment of each recording in the test set",
    "segmentation_leakage_random_split_svd": "random train-test split",
    "segmentation_leakage_random_split_voiced": "random train-test split",
    "segmentation_leakage_recording_wise_split_svd": "recording-wise split",
    "segmentation_leakage_same_segment_of_each_recording_svd": "fifth segment of each recording in the test set",
    "segmentation_leakage_same_segment_of_each_recording_voiced": "fifth segment of each recording in the test set"
}
# Empty dataframe for writing the counts of healthy and pathological segments
data = pd.DataFrame(columns=["dataset", "split", "pathological_train", "pathological_test", "healthy_train",
                             "healthy_test", "split_ratio"])

# Looping through all datasets to get the counts
for folder in PATH_TO_DATASETS.glob("*"):
    dataset_type = "VOICED" if "voiced" in str(folder) else "SVD"
    healthy_counts = []
    pathological_counts = []
    for subset in ["train", "test"]:
        paths = list(Path(folder, subset).glob("**/*.png"))
        # Healthy segments contain /healthy/ in their path
        healthy_counts.append(len([s for s in paths if "\\healthy\\" in str(s)]))
        # Unhealthy segments contain either /nonhealthy/, or /unhealthy/ in their paths
        pathological_counts.append(len([s for s in paths if ("\\unhealthy\\" in str(s) or "\\nonhealthy\\" in str(s))]))
    # Ratio of test segments to all used segments -> showing the train-test split ratio is very similar for each split
    # strategy
    ratio = (pathological_counts[1] + healthy_counts[1]) / (sum(healthy_counts) + sum(pathological_counts))
    # Adding a row into the dataframe
    data.loc[len(data)] = [dataset_type, folder.name] +  pathological_counts + healthy_counts + [ratio]

# Renaming the folder names to be more explanatory
data["split"] = data["split"].apply(lambda x: DATASET_RENAMING[x])

# Loading a template which will be used in the article
with open(Path("article_tables", "templates", "table_counts_header.tex"), "r") as template:
    article_table = template.read()
# Generating separate tables for SVD and VOICED
for dataset in ["SVD", "VOICED"]:
    data_to_write = data[data["dataset"] == dataset].drop(columns=["dataset"])
    # Using custom table, taking just the content (numbers) and formatting for better look in the LATEX code
    table_content = "\n\t\t\t".join(data_to_write.to_latex(header=False, index=False, float_format="%.4f").split("\n")[3:-3])
    table_content = article_table.format(table_content=table_content, dataset=dataset, dataset_lower=dataset.lower())
    # Saving the final table as LATEX file
    with open(Path("article_tables", f"table_counts_{dataset.lower()}.tex"), "w") as table:
        table.write(table_content)