"""
Module with various data preprocessing functions.
"""
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

from src.utils.octave_filter_bank import octave_filtering


def wav2spectrogram(chunk: np.array, destination_path: Path, fft_window_length: int, fft_overlap: int,
                    spectrogram_resolution: tuple, dpi: int = 300, samplerate: int = 50000):
    """
    Converts sound file (chunk) to its spectrogram and save it to destination_path folder.
    Filename is the same as source sound file (but with .png extension).
    :param fft_window_length: length of FFT window (Hamming)
    :param fft_overlap: number of points overlapping between neighboring window
    :param chunk: numpy array containing the chunk values (numpy)
    :param destination_path: path to folder where the spectrogram is saved. (pathlib)
    :param spectrogram_resolution: resolution of the resulting image in pixels
    :param dpi: resolution density (dots per inch)
    :param samplerate: samplerate of the recording from which the chunk originates
    :return: None
    """

    # Convert the dimensions from pixels to inches
    inch_x = spectrogram_resolution[0] / dpi
    inch_y = spectrogram_resolution[1] / dpi

    # Create spectrogram
    frequencies, times, spectrogram = signal.spectrogram(chunk,
                                                         fs=samplerate,
                                                         scaling="spectrum", nfft=None, mode="psd",
                                                         window=np.hamming(fft_window_length),
                                                         noverlap=fft_overlap)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(inch_y, inch_x)
    plot_axes = plt.Axes(fig, [0., 0., 1., 1.])
    plot_axes.set_axis_off()
    fig.add_axes(plot_axes)
    plot_axes.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap="Greys")

    plt.savefig(destination_path, format="png",
                bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close("all")

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
