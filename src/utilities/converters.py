"""
Module with various data preprocessing functions.
"""
import shutil
from pathlib import Path
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

from utilities.octave_filter_bank import octave_filtering


def wav2spectrogram(source_path: Path, destination_path: Path, fft_window_length: int, fft_overlap: int,
                    spectrogram_resolution: tuple, dpi: int = 300, octaves: list = None,
                    standard_chunk: bool = False, resampling_freq: float = None):
    """
    Converts sound file (source_path) to its spectrogram and save it to destination_path folder.
    Filename is the same as source sound file (but with .png extension).
    :param fft_window_length: length of FFT window (Hamming)
    :param fft_overlap: number of points overlapping between neighboring window
    :param source_path: path to sound file (pathlib)
    :param destination_path: path to folder where the spectrogram is saved. (pathlib)
    :param spectrogram_resolution: resolution of the resulting image in pixels
    :param dpi: resolution density (dots per inch)
    :param octaves: used for octave filtering, ignored if not defined when calling the function
    :param standard_chunk: used 1 chunk wav partitioning, so the last second of wav file is used
    :param resampling_freq: used for resampling recordings to the same frequency
    :return: None
    """

    # Convert the dimensions from pixels to inches
    inch_x = spectrogram_resolution[0] / dpi
    inch_y = spectrogram_resolution[1] / dpi

    # Create spectrogram
    if octaves is None:
        octaves = []

    sample_rate, samples = wavfile.read(source_path)

    if resampling_freq is not None:
        number_of_samples = round(len(samples) * resampling_freq / sample_rate)
        samples = signal.resample(samples, number_of_samples)
        sample_rate = resampling_freq

    if octaves is not None:
        samples = octave_filtering(octaves, samples)

    if standard_chunk:
        if len(samples) > sample_rate + 1:
            middle_point = int(len(samples) / 2)
            samples = samples[- middle_point - int(sample_rate / 2): \
                              - middle_point + int(sample_rate / 2)]
        else:
            return
    frequencies, times, spectrogram = signal.spectrogram(samples,
                                                         fs=sample_rate,
                                                         scaling="spectrum", nfft=None, mode="psd",
                                                         window=np.hamming(fft_window_length),
                                                         noverlap=fft_overlap)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(inch_y, inch_x)
    plot_axes = plt.Axes(fig, [0., 0., 1., 1.])
    plot_axes.set_axis_off()
    fig.add_axes(plot_axes)
    plot_axes.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap="Greys")
    plt.savefig(destination_path.joinpath(f"{source_path.stem}.png"), format="png",
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close("all")


def stereo2mono(source_path: Path, destination_path: Path):
    """
    Converts stereo wav sound file to mono (single channel) wav file.
    :param source_path: path to stereo wav file
    :param destination_path: path to save converted mono wav file
    :return: None
    """
    sound = AudioSegment.from_wav(source_path)
    sound = sound.set_channels(1)
    sound.export(destination_path.joinpath(source_path.name), format="wav")


def txt2wav(source_path: Path, destination_path: Path, sample_rate: int, chunks: int = 1):
    """
    Converts voiced db, where data files are text files, cointaining wav sample values.
    :param source_path: path to voiced database txt files
    :param destination_path: path to destination folder
    :param sample_rate: target wav sample rate
    :param chunks: number of chunks -> each txt is divided to multiple wav files
    :return: None
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    # print(source_path)
    txt_data = np.loadtxt(source_path)
    if chunks > 1:
        wav_chunks = np.array_split(txt_data, chunks)
        wav_chunks.pop(0)  # to remove bad data at start
    else:
        wav_chunks = np.array_split(txt_data, chunks)

    for idx, wav_chunk in enumerate(wav_chunks):
        chunk_path = destination_path.joinpath(f"{source_path.stem}_{idx:05d}.wav")
        if not chunk_path.is_file():
            # print(f"creating {chunk_path}")
            wavfile.write(filename=chunk_path, rate=sample_rate, data=wav_chunk)


def rename_voiced(voiced_path: Path, destination_path: Path):
    """
    Add label (healty/nonhealty) to voiced txt files and save them to separate folder.
    :param voiced_path: path to original voiced database
    :param destination_path: path to destination folder with txt files
    :return: None
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    for description_file in voiced_path.glob("*.hea"):
        with open(description_file, "r") as f:
            processed_filename = description_file.stem
            data = f.readlines()
            if "healthy" in data[-1]:
                destination_filename = processed_filename + "_healthy.txt"
                shutil.copy(voiced_path.joinpath(processed_filename + ".txt"),
                            destination_path.joinpath(processed_filename + "_healthy.txt"))
            else:
                destination_filename = processed_filename + "_nonhealthy.txt"
                shutil.copy(voiced_path.joinpath(processed_filename + ".txt"),
                            destination_path.joinpath(destination_filename))
