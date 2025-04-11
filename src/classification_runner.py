"""
This script is used to test the classification performance of the YOLOv8 model
on differently split datasets.
"""
#pylint: disable=use-maxsplit-arg,invalid-name
import os
from ultralytics import YOLO

if __name__ == "__main__":
    datasets = [
                "patients_wise_datasets_voiced",
                "patients_random_segment_both_datasets_voiced",
                "patients_random_segments_datasets_voiced",
                "patients_same_segment_both_datasets_voiced",
                # "patients_wise_datasets",
                # "patients_random_segment_both_datasets",
                # "patients_random_segments_datasets",
                # "patients_same_segment_both_datasets",
                ]

    models = ["yolov8n-cls.pt",
              "yolov8s-cls.pt",
              "yolov8m-cls.pt",
              "yolov8l-cls.pt",
              "yolov8x-cls.pt"]

    epochs = 300
    
    for folder_name in datasets:
        # folder_abs_path = os.path.join(ABS_PATH, folder_name)
        for model_name in models:
            # Load a model
            model = YOLO(os.path.join(".","models",model_name))

            # Train the model
            model.train(data=folder_name, optimizer="SGD", epochs=epochs,
                        name=f"{folder_name}_{model_name.split('.')[0]}_{epochs}_sgd")
