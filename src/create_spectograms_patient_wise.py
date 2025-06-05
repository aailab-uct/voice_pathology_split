from pathlib import Path
from scipy.io.wavfile import read
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pandas as pd

from utilities.converters import wav2spectrogram, txt2wav

# SVD wav to txt
print("Transforming files SVD wav files to txt files...")
svd_path = Path("datasets","svd")
svd_path_renamed = Path("datasets","svdadult_renamed_pw_test")
svd_path_renamed.mkdir(parents=True, exist_ok=True)
scaler = MinMaxScaler((-1, 1))

df = pd.read_csv("svd_information.csv", 
                   usecols=["sessionid", "sessiondate", "talkerid", "pathologies"])
df['sessiondate'] = pd.to_datetime(df['sessiondate'], format="%Y-%m-%d")

downloaded_files = [i.stem for i in svd_path.glob("*.wav")]
not_found = []

# load all talkers data of SVD
talker_dict = {}
for _, row in tqdm(df.iterrows(), unit="file"):
  filename = str(row['sessionid']).strip() + "-a_n"
  if not filename in downloaded_files:
    not_found.append(filename)
    continue
  file = svd_path.joinpath(filename + ".wav")

  label = "healthy"
  if pd.notna(row['pathologies']):
    label = "unhealthy"

  talker_id = int(row['talkerid'])
  
  if talker_id not in talker_dict.keys():
    talker_dict[talker_id] = []

  talker_dict[talker_id].append({'session_id' : row['sessionid'], 
                                 'session_date' : row['sessiondate'], 
                                 'label' : label, 
                                 'recording_name' : file})

# Prepare txt files for each recording
for talker_id, recordings in tqdm(talker_dict.items()):
  recordings = sorted(recordings, key=lambda r: r['session_date'])
  healthy_order = 0
  unhealthy_order = 0
  for recording in recordings:
    file_array = read(recording['recording_name'])
    file_array_float = np.array(file_array[1], dtype="f")
    file_array_norm = scaler.fit_transform(file_array_float.reshape(-1, 1))

    recording_order = None
    if recording['label'] == "healthy":
      recording_order = healthy_order
      healthy_order +=1
    else:
      recording_order = unhealthy_order
      unhealthy_order += 1
      
    # filename: svdadult[talkerId]_[label]_[sessionId]_order[duplicitNum].txt
    file_name = svd_path_renamed.joinpath(f'svdadult{talker_id:0>4}_{recording['label']}_{recording['session_id']:0>4}_order{recording_order:0>2}.txt')
    np.savetxt(file_name, file_array_norm, fmt="%f", delimiter='\n')


if not_found:
  print("The following SVD recordings were not found:")
  for file in not_found:
    print(file)
  print("You need to download them before proceeding.")
  if input("Do you want still to proceed with spectogram creation? (y/n)") != "y":
    exit()
else:
  print("All files found!")

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