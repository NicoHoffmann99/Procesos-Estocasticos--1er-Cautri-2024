import numpy as np
import sounddevice as sd
import soundfile as sf
from exercise_1 import pca_compression, segment_signal, reconstruct_signal


def calculate_mse(original_signal, reconstructed_signal):
    mse = 0
    length = len(reconstructed_signal)
    for i in range(length):
        mse += np.power((original_signal[i] - reconstructed_signal[i]), 2)
    return mse / length


def normalize_compress_decompress(audio, samplerate, sample_length, compress_rates, graph=False):
    print(f"Reproduciendo audio original")
    sd.play(audio, samplerate)
    sd.wait()
    signal_rms = np.linalg.norm(audio)
    audio = np.apply_along_axis(lambda x: x / signal_rms, 0, audio)
    sample_number = int(len(audio) / sample_length)
    corrected_array = audio[np.arange(sample_length * sample_number)]
    X_m = segment_signal(corrected_array, sample_length)
    for compression_rate in compress_rates:
        Y_m, U = pca_compression(X_m, compression_rate)
        reconstructed_signal = reconstruct_signal(Y_m, U, signal_rms)
        print(f"Reproduciendo audio con compresion {compression_rate}")
        sd.play(reconstructed_signal, samplerate)
        sd.wait()
        if graph:
            print(calculate_mse(corrected_array, reconstructed_signal))
