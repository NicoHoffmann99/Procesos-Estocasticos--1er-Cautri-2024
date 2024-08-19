import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

data, fs = sf.read('audio_01_2024a.wav')

signal_rms = np.linalg.norm(data)


def divide_rms(x):
    return x / signal_rms


#sd.play(data, fs)
#sd.wait()
data = np.apply_along_axis(divide_rms, 0, data)

array_even_elements = data[np.arange(data.size - 1)]


def segment_signal(signal, length):
    array_list = []
    for i in range(0, len(signal), length):
        array_list.append(signal[i:i + length])
    return np.array(array_list)


def reconstruct_signal(signal, U, signal_rms):
    signal = np.dot(U, signal)
    return (signal.reshape(-1, order='F')) * signal_rms


my_array = segment_signal(array_even_elements, 2)

# plt.hist(x=my_array[:, 0], bins=50)
# plt.show()
#
# plt.hist(x=my_array[:, 1], bins=50)
# plt.show()
#
# plt.scatter(my_array[:, 0], my_array[:, 1])
# plt.show()


# 1.c
def calculate_mean(array, number_samples):
    number_columns = len(array[1, :])
    value_list = []
    for i in range(number_columns):
        sum_i = np.sum(array[:, i])
        value_list.append(sum_i)
    return (1 / number_samples) * np.array(value_list)


def covariance_matrix(array):
    array_size = array.shape
    rows = array_size[0]
    columns = array_size[1]
    mean = calculate_mean(array, rows)
    matrix = np.zeros((columns, columns))
    for i in range(rows):
        aux_array = array[i, :] - mean
        aux_matrix = np.outer(aux_array, aux_array)
        matrix += aux_matrix
    return (1 / rows) * matrix


covariance_matrix_x = covariance_matrix(my_array)

# eigenvalues / eigenvectors
w, v = np.linalg.eig(covariance_matrix_x)

# 1.d)
Y_m = np.dot(v, np.transpose(my_array))
plt.hist(x=Y_m[0, :], bins=50)
plt.show()
plt.hist(x=Y_m[1, :], bins=50)
plt.show()
plt.scatter(Y_m[0, :], Y_m[1, :])
plt.show()


def pca_compression(X_m, compression_rate):
    covariance_x = covariance_matrix(X_m)
    sample_length = X_m.shape[1]
    eigenvalues, eigenvectors = np.linalg.eig(covariance_x)
    k = int(np.ceil((1-compression_rate) * sample_length))
    U = np.zeros((sample_length, k))
    for i in range(k):
        U[:, i] = eigenvectors[:, i]
    y_matrix = np.dot(np.transpose(U), np.transpose(X_m))
    return y_matrix, U
