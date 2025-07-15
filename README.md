# Quantifying Segmentation-Induced Data Leakage in Machine Learning for Voice Pathology Detection Pipeline

This repository contains the code for the paper "Quantifying Segmentation-Induced Data 
Leakage in Machine Learning for Voice Pathology Detection" by Jakub Steinbach, Tomáš 
Jirsa, Jiří Ingr, Zuzana Urbániová, Laura Verde, Zuzana Sedláková, Martin Chovanec, 
Zhang Zhang, Noriyasu Homma, Jan Vrba.

## <span style="color:crimson;font-weight:bold">!!!UPDATE!!!</span> Requirements <span style="color:crimson;font-weight:bold">!!!UPDATE!!!</span>


For running experiments
- Nvidia GPU (for speed reasons)
- Prepared dataset (see below)
- yolo weights (not included - see below)

For dataset preparation
- Python
- pandas
- pathlib
- tqdm (for progress bar)

For spectogram dataset creation
- Python
- scipy
- numpy
- matplotlib
- pydub
- tqdm (for progress bar)
- scikit-learn
- pandas

The ```requirements.txt``` file is NOT representative of the requirements for the whole 
project, but only for the Docker image. For the dataset preparation, you need to use the 
```requirements_dataset.txt``` file.

## Dataset preparation

The data are not included in this repository due to the license reason, but they can be 
created from publicly available datasets. They are composed of VOICED and Saarbruecken Voice 
Database, which are converted into spectograms. 

First you need to download the Saarbruecken Voice Database 
[available here](https://stimmdb.coli.uni-saarland.de/). 
You need to download only normal /a:/ vowel encoded as wav. Additionaly, you can download 
only recordings with ids listed in [svd_filelist.txt](misc/used_svd_recordings.txt) because 
other recordings are not used. Then aquire the VOICED dataset 
[available here](https://doi.org/10.13026/C25Q2N). Put the recordings into folders 
```svd_db``` and ```voiced_db```.

At this step, we assume following folder structure:
    
```
svd_db/
    1-a_n.wav
    2-a_n.wav
    ...
voiced_db/
    RECORDS
    SHA256SUMS.txt
    voice001-info.txt
    voice001.dat
    voice001.wav
    voice001.txt
    voice002-info.txt
    ...
...
main.py
...
```

To start the experiments, you can simply run the ```main.py``` script. The script checks 
the presence of spectrograms and creates them on demand, before splitting the data
according to the scenarios. Then, it trains selected YOLOv8 classification models on 
the prepared datasets.

Due to the randomness in the dataset preparation, you can find the exact split we used
during our experiments in the ```dataset_lists``` folder. Similarly, you can compare your
results with ours in the ```misc/segmentation_leakage_results.csv``` file.