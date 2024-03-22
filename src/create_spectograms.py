from pathlib import Path
from scipy.io.wavfile import read
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

from utilities.converters import wav2spectrogram, txt2wav

# SVD wav to txt
svd_path = Path("datasets","svd")
folders = [x for x in svd_path.iterdir()]
list_arrays = []
list_names = []
scaler = MinMaxScaler((-1, 1))

for file in svd_path.glob("*.wav"):
  file_array = read(file)
  file_array_float = np.array(file_array[1], dtype="f")
  file_array_norm = scaler.fit_transform(file_array_float.reshape(-1, 1))
  list_arrays.append(file_array_norm)
  list_names.append(file.name.split("-")[0])

info = pd.read_csv("svd_file_information.csv", index_col=0)
info.talkersex = info.talkersex.replace("w", "f")
info.pathologies = info.pathologies.fillna("healthy")
info.loc[info.pathologies != "healthy", "pathologies"] = "unhealthy"

svd_path_renamed = Path("datasets","svdadult_renamed")
if not svd_path_renamed.exists():
  svd_path_renamed.mkdir()

print("Transforming files SVD wav files to txt files...")
for index, (array, name) in tqdm(enumerate(zip(list_arrays, list_names)), 
                                 total=len(list_arrays),
                                 unit="file"):
  info_row = info[info.sessionid == int(name)]
  if info_row.shape[0] != 1:
    raise Exception("info_row is supposed to have only one row!")

  if info.talkerage.values[index] >= 18:
    file_name = svd_path_renamed.joinpath(f'svdadult{name:0>4}_{info_row.pathologies.values[0]}_50000.txt')
    np.savetxt(file_name, array, fmt="%f", delimiter='\n')
  else:
    #print(f"Record {name} skipped.")
    continue

# Rename VOICED files
voiced_path = Path("datasets","voiced")
voiced_path_renamed = Path("datasets","voiced_renamed")

if not voiced_path_renamed.exists():
  voiced_path_renamed.mkdir()

print("Renaming files VOICED wav files...")
for file in tqdm(voiced_path.glob("*.hea"), unit="file",
                 total=len(list(voiced_path.glob("*.hea")))):
  with open(file, "r") as f:
    data = f.readlines()
    if "healthy" in data[-1]:
      destination_filename = file.stem + "_healthy_8000.txt"
    else:
      destination_filename = file.stem + "_nonhealthy_8000.txt"
    shutil.copy(voiced_path.joinpath(file.stem + ".txt"),
                voiced_path_renamed.joinpath(destination_filename))


# Create spectrograms of voiced
dataset = "voiced"
svd_path = Path("datasets", f"{dataset}_renamed")
destination_path_wavs = Path("datasets", f"{dataset}_renamed_wavs")
destination_path_spectrogram = Path("datasets", f"spectrogram_{dataset}")
destination_path_wavs.mkdir(parents=True, exist_ok=True)
destination_path_spectrogram.mkdir(parents=True, exist_ok=True)

fft_len = 256  # length of fft window (Hamming is used by default)
fft_overlap = 128  # overlap of ffts windows
spectrogram_resolution = (480, 480)  # desired spectrogram resolution as tuple
octaves = []  # specify octave filters if needed, see utilities.octave_filter_bank.py for details
chunks = 10  # each wav file is split into multiple chunks, set the number of chunks
sample_rate = 8000

for txt_file in svd_path.glob("*.*"):
  txt2wav(txt_file, destination_path_wavs, sample_rate, chunks)

print("Creating spectrograms of VOICED...")
for wav_file in tqdm(destination_path_wavs.glob("*.*"),
                     total=len(list(destination_path_wavs.glob("*.*"))),
                     unit="file"):
    wav2spectrogram(wav_file, destination_path_spectrogram, fft_len, fft_overlap,
                                    spectrogram_resolution, octaves=[], standard_chunk=False,
                                    resampling_freq=None)

# Create spectrograms of svd
dataset = "svdadult"
svd_path = Path("datasets", f"{dataset}_renamed")
destination_path_wavs = Path("datasets", f"{dataset}_renamed_wavs")
destination_path_spectrogram = Path("datasets", f"spectrogram")
destination_path_wavs.mkdir(parents=True, exist_ok=True)
destination_path_spectrogram.mkdir(parents=True, exist_ok=True)

fft_len = 256  # length of fft window (Hamming is used by default)
fft_overlap = 128  # overlap of ffts windows
spectrogram_resolution = (480, 480)  # desired spectrogram resolution as tuple
octaves = []  # specify octave filters if needed, see utilities.octave_filter_bank.py for details
chunks = 10  # each wav file is split into multiple chunks, set the number of chunks
sample_rate = 50000

for txt_file in svd_path.glob("*.*"):
  txt2wav(txt_file, destination_path_wavs, sample_rate, chunks)

print("Creating spectrograms of SVD...")
for wav_file in tqdm(destination_path_wavs.glob("*.*"),
                     total=len(list(destination_path_wavs.glob("*.*"))),
                     unit="file"):
    wav2spectrogram(wav_file, destination_path_spectrogram, fft_len, fft_overlap,
                                    spectrogram_resolution, octaves=[], standard_chunk=False,
                                    resampling_freq=None)