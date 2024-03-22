# Voice Pathology Split

This repository contains the code for the paper "Unmasking Biomedical Dataset Bias: The Pitfalls of Misaligned Data Splitting in Deep Learning-Based Pathology Detection" by Jakub Steinbach, Zuzana Urbániová, Tomáš Jirsa, Laura Verde, Zuzana
Sedláková, Martin Chovanec, Noriyasu Homma and Jan Vrba.

## Requirements

- Docker
- Nvidia GPU (for speed reasons)
- Nvidia Docker (for GPU support in Docker)
- Prepared dataset (see below)
- yolo weights (not included - see below)

For dataset preparation
- Python
- tqdm (for progress bar)

For spectogram dataset creation
- Python
- scipy
- numpy
- matplotlib
- pydub

## Dataset preparation

The dataset is not included in this repository due to the license reason, but it can be created from publicly available datasets. It is composed of VOICED and Saarbruecken Voice Database, which are converted into spectograms. 

First you need to download the Saarbruecken Voice Database [available here](https://stimmdb.coli.uni-saarland.de/index.php4). You need to download only normal /a/ vowel encoded as wav. As there is a limit of files you can download in one archive, we suggest you download first females, then males and combine the downloaded recordings in one folder. Then aquire the VOICED dataset [available here](https://doi.org/10.13026/C25Q2N). Create the ```datasets``` folder and put recordings into folders ```datasets/svd``` and ```datasets/voiced```.

To create the spectograms, run the following command:

```
python src/create_spectograms.py
```

Spectograms from SVD are stored in ```datasets/spectogram``` and are named in the following fashion:

```
svdadult[four_digit_number]_[healthy/unhealthy]_[frequency]_[number_of_the_split].png
```

Spectograms from VOICED are stored in ```datasets/spectogram_voiced``` and are named in the following fashion:

```
voice[three_digit_number]_[healthy/nonhealthy]_[frequency]_[number_of_the_split].png
```

To split the dataset into train and test sets for each experiment, run the following command:

```
python src/reproducibility_split.py
```

After the split, the dataset is ready for use. If you want to verify the split is same as ours, compare the output of the script with following SHA256 hash of filenames.

```
df5994930be1229837a730e3c6036fe4554ad9ba84c74f79e7f3bf43dd308f28
```

## Docker usage

First aquire the yolo weights and place them in ```yolov8``` folder in root of this repository. We used the ```yolov8[nsmlx]-cls.pt``` weights from the [ultralytics release v0.0.0](https://github.com/ultralytics/assets/releases/tag/v0.0.0). If you want to use different weights, change the name in the ```src/classification_runner.py``` file. Ultralytics package also requires the ```yolov8n.pt``` weights in the same folder.

First, build the docker image:

```
docker build -t voice_pathology_split .
```

Then, run the docker image:

```
docker run -it --runtime=nvidia --gpus all \
    --network none \
    --mount 'type=bind,source=[path1],target=/app/runs' \
    --mount 'type=bind,source=[path2],target=/app/datasets' \
    --shm-size=1g voice_pathology_split
```

where [path1] is the path to the folder where the runs will be stored (useful for easier access) and [path2] is the path to the folder where the datasets are stored. Without ```--runtime=nvidia --gpus all``` the GPU will not be used (please note that runtime name can vary). ```--shm-size=1g``` is needed for the docker container to have enough memory to run the code, otherwise it will crash.

The computation will take a while (more than 2 days on our setup). Feel free to change used datasets and yolo weights in the ```src/classification_runner.py``` file (do so before building the image).
