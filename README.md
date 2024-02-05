# Voice Pathology Split

TODO! fill in the description and citation

## Requirements

- Docker
- Nvidia GPU (for speed reasons)
- Nvidia Docker (for GPU support in Docker)

For dataset preparation
- Python
- tqdm (for progress bar)

For dataset creation (not in scope of this repository)
- voiced and SVD dataset
- Python
- scipy
- numpy
- matplotlib
- pydub

## Dataset preparation

The dataset is not included in this repository. It can be created using the voiced and SVD dataset, which are converted into spectograms. Spectograms from SVD are stored in ```datasets/spectogram``` and are named in the following fashion:

    ```svdadult[four_digit_number]_[healthy/unhealthy]_[frequency]_[number_of_the_split].png```

Spectograms from voiced are stored in ```datasets/spectogram_voiced``` and are named in the following fashion:

    ```voice[three_digit_number]_[healthy/nonhealthy]_[frequency]_[number_of_the_split].png```

To split the dataset into train and test set, run the following command:

```python src/split_all.py```

or run the individual script.

After the split, the dataset is ready for use. If you want to verify the split is reproducible, run the following command:

```python src/test_reproducibility_split.py```

and compare the datasets_filelist_pub.txt or the SHA256 hash of filenames.

```df5994930be1229837a730e3c6036fe4554ad9ba84c74f79e7f3bf43dd308f28```

## Docker usage

First, build the docker image:

```docker build -t voice_pathology_split .```

Then, run the docker image:

```
docker run -it --runtime=nvidia --gpus all \
    --network none \
    --mount 'type=bind,source=[path1],target=/app/runs' \
    --mount 'type=bind,source=[path2],target=/app/datasets' \
    --shm-size=1g voice_pathology_split
```

[path1] is the path to the folder where the runs will be stored (useful for easier access) and [path2] is the path to the folder where the datasets are stored. Without ```--runtime=nvidia --gpus all``` the GPU will not be used (please note that runtime name can vary). ```--shm-size=1g``` is needed for the docker container to have enough memory to run the code, otherwise it will crash.

The computation will take a while (more than 2 days). Feel free to change used datasets and yolo weights in the ```src/classification_runner.py``` file (do so before building the image).
