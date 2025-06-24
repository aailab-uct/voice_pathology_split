"""
This script is used to test the classification performance of the YOLOv8 model
on differently split datasets.
"""
#pylint: disable=use-maxsplit-arg,invalid-name
import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd


PATH_DATASETS = Path("datasets")

if __name__ == "__main__":
    results_table = pd.DataFrame(columns=["Approach/dataset", "Architecture", "ACC", "SPE", "TN", "TP", "FN", "FP"])
    datasets = ['patients_wise_split_svd_no_duplicities',
                'patients_wise_split_svd_with_duplicities',
                'patients_wise_split_voiced',
                'segmentation_leakage_random_segment_of_each_recording_svd',
                'segmentation_leakage_random_segment_of_each_recording_voiced',
                'segmentation_leakage_random_split_svd',
                'segmentation_leakage_random_split_voiced',
                'segmentation_leakage_recording_wise_split_svd',
                'segmentation_leakage_same_segment_of_each_recording_svd',
                'segmentation_leakage_same_segment_of_each_recording_voiced'
                ]

    models = ["yolov8n-cls.pt",
              "yolov8s-cls.pt",
              "yolov8m-cls.pt",
              "yolov8l-cls.pt",
              "yolov8x-cls.pt"
              ]

    epochs = 1
    
    for folder_name in datasets:
        # folder_abs_path = os.path.join(ABS_PATH, folder_name)
        for model_name in models:
            # Load a model
            model = YOLO(os.path.join("src", "models", model_name))

            # Train the model
            model.train(data=PATH_DATASETS.joinpath(folder_name), optimizer="SGD", epochs=epochs,
                        name=f"{folder_name}_{model_name.split('.')[0]}_{epochs}_sgd")
            print("#"*30, "Validation", "#"*30)
            results = model.val("test.yaml")  # use your custom dataset YAML

            TN = results.confusion_matrix.matrix[1][1]
            TP = results.confusion_matrix.matrix[0][0]
            FN = results.confusion_matrix.matrix[0][1]
            FP = results.confusion_matrix.matrix[1][0]
            ACC = (TP + TN) / (TP + FP + FN + FP)
            SEN = TP / (TP + FN)
            SPE = TN / (TN + FP)

            results_table.loc[len(results_table.index)] = [folder_name, model_name, ACC, SPE, TN, TP, FN, FP]
            results_table.to_csv("segmentation_leakage_results.csv")