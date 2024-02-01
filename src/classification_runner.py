"""
This script is used to test the classification performance of the YOLOv5 model
on differently split datasets.
"""
#pylint: disable=use-maxsplit-arg,invalid-name
import os
from ultralytics import YOLO
import torch

# Set seed for reproducibility - still not reproducible :(
# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(42)


if __name__ == "__main__":
    datasets = ["datasets/patients_wise_datasets_voiced",
                "datasets/patients_random_segment_both_datasets_voiced",
                "datasets/patients_random_segments_datasets_voiced",
                "datasets/patients_same_segment_both_datasets_voiced",
                "datasets/patients_wise_datasets",
                "datasets/patients_random_segment_both_datasets",
                "datasets/patients_random_segments_datasets",
                "datasets/patients_same_segment_both_datasets",
                ]

    models = ["yolov8n-cls.pt",
              "yolov8s-cls.pt",
              "yolov8m-cls.pt",
              "yolov8l-cls.pt",
              "yolov8x-cls.pt"]

    epochs = 300

    for folder_name in datasets:
        for model_name in models:
            # Load a model
            model = YOLO(os.path.join(".","yolov8",model_name))

            # Train the model
            model.train(data=folder_name, optimizer="SGD", epochs=epochs,
                        name=f"{folder_name}_{model_name.split('.')[0]}_{epochs}_sgd",)
