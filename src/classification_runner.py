"""
This script evaluates the classification performance of YOLOv8 models
on different dataset splits (scenarios).

Constants:
    PATH_DATASETS: Path to the folder containing scenario subfolders.
    BASE_DIR: Root directory of the project (used for relative paths).
    MODELS: List of YOLOv8 model weights to test (nano → xlarge).
    EPOCHS: Number of training epochs for each experiment.

Output:
    - Trained YOLOv8 models (saved in runs/classify/...)
    - CSV summary of evaluation metrics across all scenarios and model sizes.
"""

from pathlib import Path
from ultralytics import YOLO
import pandas as pd

PATH_DATASETS = Path("datasets_variance_test")
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS = ["yolov8n-cls.pt",
          # "yolov8s-cls.pt",
          # "yolov8m-cls.pt",
          # "yolov8l-cls.pt",
          # "yolov8x-cls.pt"
          ]

EPOCHS = 300

def run_experiments():
    """
    Run classification experiments on all dataset scenarios using multiple YOLOv8 models.

    For each dataset scenario:
        - Train each YOLOv8 model variant for a fixed number of epochs.
        - Load the best model checkpoint.
        - Evaluate on both validation and test subsets.
        - Compute confusion-matrix-derived metrics:
            * Accuracy (ACC)
            * Unweighted Average Recall (UAR)
        - Save results in a cumulative CSV file.

    Metrics for each subset (val/test):
        ACC: (TP + TN) / total
        SEN: TP / (TP + FN)
        SPE: TN / (TN + FP)
        UAR: (SEN + SPE) / 2

    Results are appended into a pandas DataFrame and written to:
        `segmentation_leakage_results.csv`
    """

    # Create an empty results table to accumulate experiment results
    results_table = pd.DataFrame(columns=["scenario", "size",
                                    "val_acc", "val_uar", "val_tp", "val_fp", "val_tn", "val_fn",
                                    "test_acc", "test_uar", "test_tp", "test_fp", "test_tn", "test_fn"])

    # Identify all dataset scenarios (each folder inside datasets/)
    scenarios = [x.name for x in PATH_DATASETS.glob("*")]

    # Train and evaluate all YOLOv8 model sizes for each scenario
    for scenario in scenarios:
        for model_name in MODELS:
            # Dataset path for YOLO model
            dataset_path = BASE_DIR.joinpath("datasets", scenario)
            # Initialize pre-trained YOLO model
            model = YOLO(Path("models", model_name))
            # Train the model on the dataset scenario
            model.train(data=dataset_path, optimizer="SGD", epochs=EPOCHS,
                        name=f"{scenario}_{model_name.split('.')[0]}_{EPOCHS}_sgd")

            # Retrieve the best model checkpoint (from training run)
            best_model_path = BASE_DIR.joinpath("runs", "classify",
                                                f"{scenario}_{model_name.split('.')[0]}_{EPOCHS}_sgd",
                                                "weights", "best.pt")
            model = YOLO(best_model_path)
            # Validate on the validation split
            results_val = model.val(data=dataset_path, split="val")
            # External validation
            results_test = model.val(data=dataset_path, split="test")
            # Store results for both val and test subsets
            row = []
            for subset, result in zip(["val", "test"], [results_val, results_test]):
                # Extract confusion matrix values
                TP = result.confusion_matrix.matrix[1][1]
                TN = result.confusion_matrix.matrix[0][0]
                FP = result.confusion_matrix.matrix[0][1]
                FN = result.confusion_matrix.matrix[1][0]
                # Compute metrics
                ACC = (TP + TN) / (TP + TN + FN + FP)
                SEN = TP / (TP + FN)
                SPE = TN / (TN + FP)
                UAR = (SEN + SPE) / 2 # Unweighted Average Recall

                row += [ACC, UAR, TP, FP, TN, FN]

            # Append one row (scenario + model size + metrics)
            results_table.loc[len(results_table.index)] = [scenario, model_name] + row
            # Save cumulative results after each experiment
            results_table.to_csv("segmentation_leakage_results_variance_test.csv")
