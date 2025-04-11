from pathlib import Path
from tqdm import tqdm
import shutil

from utilities.converters import wav2spectrogram, txt2wav


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
    shutil.copy(voiced_path.joinpath(file.stem + "_trimmed.txt"),
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
