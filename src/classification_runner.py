"""
This script is used to test the classification performance of the YOLOv8 model
on differently split datasets.
"""
#pylint: disable=use-maxsplit-arg,invalid-name
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

PATH_DATASETS = Path("datasets")
BASE_DIR = Path(__file__).resolve().parent.parent
PATH_YAML = Path("misc", "yaml_template.txt")
MODELS = ["yolov8n-cls.pt",
          "yolov8s-cls.pt",
          "yolov8m-cls.pt",
          "yolov8l-cls.pt",
          "yolov8x-cls.pt"]

EPOCHS = 3

def run_experiments():
    results_table = pd.DataFrame(columns=["scenario", "size",
                                    "val_acc", "val_uar", "val_tp", "val_fp", "val_tn", "val_fn",
                                    "test_acc", "test_uar", "test_tp", "test_fp", "test_tn", "test_fn"])



    scenarios = [x.name for x in PATH_DATASETS.glob("*")]

    # Train all models for each scenario
    for scenario in scenarios:
        for model_name in MODELS:
            # Define the path to the dataset for YOLO model
            dataset_path = BASE_DIR.joinpath("datasets", scenario)
            # Load the YOLO model
            model = YOLO(Path("models", model_name))
            # Train the model
            model.train(data=dataset_path, optimizer="SGD", epochs=EPOCHS,
                        name=f"{scenario}_{model_name.split('.')[0]}_{EPOCHS}_sgd")
            print("#"*30, "Validation", "#"*30)

            # Define the path to the best model settings
            best_model_path = BASE_DIR.joinpath("runs", "classify",
                                                f"{scenario}_{model_name.split('.')[0]}_{EPOCHS}_sgd", "weights",
                                                "best.pt")
            model = YOLO(best_model_path)

            results_val = model.val(data=dataset_path, split="val")
            print(results_val)

            results_test = model.val(data=dataset_path, split="test")
            print(results_test)

            row = []
            for subset, result in zip(["val", "test"], [results_val, results_test]):
                TP = result.confusion_matrix.matrix[1][1]
                TN = result.confusion_matrix.matrix[0][0]
                FP = result.confusion_matrix.matrix[0][1]
                FN = result.confusion_matrix.matrix[1][0]
                ACC = (TP + TN) / (TP + TN + FN + FP)
                SEN = TP / (TP + FN)
                SPE = TN / (TN + FP)
                UAR = (SEN + SPE) / 2
                row += [ACC, UAR, TP, FP, TN, FN]

            results_table.loc[len(results_table.index)] = [scenario, model_name] + row
            results_table.to_csv("segmentation_leakage_results.csv")
