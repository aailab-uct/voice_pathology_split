import csv
from pathlib import Path

if __name__ == '__main__':
    dataset_path = Path(".", "datasets")
    for split_strategy in dataset_path.iterdir():
        print(split_strategy)
        files_list = [[str(Path(*image_path.parts[2:]))] for image_path in split_strategy.glob("**/*.png")]
        print(len(files_list))
        print(files_list)
        with open(split_strategy.name + ".csv", "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar="'", lineterminator='\n')
            writer.writerows(files_list)
