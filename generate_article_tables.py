"""
Script for generation of tables which are used in the journal article
"""


import pandas as pd
from pathlib import Path

# Path to datasets and dictionary for renaming dataset folder names to something more suitable for publishing
PATH_TO_DATASETS = Path("datasets")
PATH_TO_TABLES = Path("article_tables")
PATH_TO_TEMPLATES = PATH_TO_TABLES.joinpath("templates")
PATH_TO_RESULTS = Path("segmentation_leakage_results.csv")

# Ordering of the model sizes
SIZE_ORDER = ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"]

# Data type management
COLS_DTYPE_TO_CHANGE = ["val_tp", "val_fp", "val_tn", "val_fn", "test_tp", "test_fp", "test_tn", "test_fn"]


##################### Count tables ######################
# Empty dataframe for writing the counts of healthy and pathological segments
data = pd.DataFrame(columns=["dataset", "split", "pathological_train", "pathological_test", "pathological_ext_val", "healthy_train",
                             "healthy_test", "healthy_ext_val", "split_ratio"])
# Looping through all datasets to get the counts
for folder in PATH_TO_DATASETS.glob("*"):
    dataset_type = "VOICED" if "voiced" in str(folder) else "SVD"
    healthy_counts = []
    pathological_counts = []
    for subset in ["train", "val", "test"]:
        paths = list(Path(folder, subset).glob("**/*.png"))
        # Healthy segments contain /healthy/ in their path
        healthy_counts.append(len([s for s in paths if "_healthy_" in str(s)]))
        # Unhealthy segments contain either /nonhealthy/, or /unhealthy/ in their paths
        pathological_counts.append(len([s for s in paths if "_pathological_" in str(s)]))
    # Ratio of test segments to all used segments -> showing the train-test split ratio is very similar for each split
    # strategy
    ratio = (pathological_counts[1] + healthy_counts[1]) / (sum(healthy_counts[:2]) + sum(pathological_counts[:2]))
    # Adding a row into the dataframe
    data.loc[data.shape[0]] = [dataset_type, folder.name] +  pathological_counts + healthy_counts + [ratio]

# Reformat the name of scenarios
data["split"] = data["split"].apply(lambda s: " ".join(s.split("_")[1:]))

# Loading a template which will be used in the article
with open(PATH_TO_TEMPLATES.joinpath("table_counts_header.tex"), "r") as template:
    article_table = template.read()
# Generating separate tables for SVD and VOICED
for db in ["SVD", "VOICED"]:
    data_to_write = data[data["dataset"] == db].drop(columns=["dataset"])
    # Using custom table, taking just the content (numbers) and formatting for better look in the LATEX code
    table_content = "\n\t\t\t".join(data_to_write.to_latex(header=False, index=False, float_format="%.4f").split("\n")[3:-3])
    table_content = article_table.format(table_content=table_content, dataset=db, dataset_lower=db.lower())
    # Saving the final table as LATEX file
    with open(PATH_TO_TABLES.joinpath(f"table_counts_{db.lower()}.tex"), "w") as table:
        table.write(table_content)

##################### Result tables ######################
data_results = pd.read_csv(PATH_TO_RESULTS).drop(columns=["Unnamed: 0"])
data_results["size"] = pd.Categorical(data_results["size"], categories=SIZE_ORDER, ordered=True)
data_results["db"] = data_results["scenario"].apply(lambda s: s.split("_")[0])
data_results["scenario"] = data_results["scenario"].apply(lambda s: " ".join(s.split("_")[1:]))
# Changing the data type to integer since these represent counts
data_results[COLS_DTYPE_TO_CHANGE] = data_results[COLS_DTYPE_TO_CHANGE].astype(int)

# Save SVD and VOICED data individually
for db in ["svd", "voiced"]:
    results_to_export = (data_results[data_results.db == db].drop(columns=["db"])
                                                              .groupby(["scenario", "size"]).sum())
    # Turning the results table to latex format
    table_content = results_to_export.to_latex(header=False, float_format="%.4f")
    # Getting rid of the declaration of tabular and outer rules as we need only the values
    table_content = "\n\t\t\t".join(table_content.split("\n")[4:-4])
    # Switching the clines for midrules
    table_content = table_content.replace("\\cline{1-14}", "\\midrule")
    # Changing the multirow vertical alignment and horizontal size to respect the size of the column
    table_content = table_content.replace("[t]", "")#.replace("{*}", "{10em}")
    # Template for result table
    with open(PATH_TO_TEMPLATES.joinpath("table_results_header.tex"), "r") as template:
        article_table = template.read()
    # Placing data in the template
    table_content = article_table.format(table_content=table_content, dataset=db.upper(), dataset_lower=db)
    # Saving the final table as LATEX file
    with open(PATH_TO_TABLES.joinpath(f"table_results_{db}.tex"), "w") as table:
        table.write(table_content)