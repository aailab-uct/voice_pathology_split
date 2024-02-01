# Voice Pathology Split

TODO! fill in the description and citation

## Requirements

- Docker
- Nvidia GPU (for speed reasons)
- Nvidia Docker (for GPU support in Docker)

For dataset preparation
- Python

For dataset creation (not in scope of this repository)
- voiced and SVD dataset
- Python
- scipy
- numpy
- matplotlib
- pydub

## Dataset preparation

TODO! add dataset preparation description and instructions

## Docker usage

First, build the docker image:

```docker build -t voice_pathology_split .```

Then, run the docker image:

```
docker run -it --runtime=nvidia --gpus all \
    --mount 'type=bind,source=[path1]/,target=/app/runs' \
    --mount 'type=bind,source=[path2],target=/app/datasets' \
    --shm-size=1g voice_pathology_split
```

[path1] is the path to the folder where the runs will be stored (useful for easier access) and [path2] is the path to the folder where the datasets are stored. Without ```--runtime=nvidia --gpus all``` the GPU will not be used (please note that runtime name can vary). ```--shm-size=1g``` is needed for the docker container to have enough memory to run the code, otherwise it will crash.

The computation will take a while (more than 2 days). Feel free to change used datasets and yolo weights in the ```src/classification_runner.py``` file (do so before building the image).
