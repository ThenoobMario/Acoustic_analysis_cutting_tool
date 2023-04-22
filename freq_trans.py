import os
import pickle

import librosa
import librosa
import librosa.core
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt



# class visualise(object):
#     def __init__(self):
#         self.plt = plt
#         self.fig = self.plt.figure(figsize = (30, 10))
#         self.plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

#     def plot_loss(self, loss, val_loss):
#         ax = self.fig.add_subplot(1, 1, 1)
#         ax.cla()
#         ax.plot(loss)
#         ax.plot(val_loss)
#         ax.set_title("Model loss")
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Loss")
#         ax.legend(["Train", "Test"], loc="upper right")

#     def save_fig(self, name):
#         self.plt.savefig(name)


# def load_file(wav_name, mono = False):
#     """
#     Loading the wav files
#     """
#     try:
#         return librosa.load(wav_name, sr = None, mono = mono)
#     except:
#         print(f"file broker or does not exist!!: {wav_name}")

# def demux_wav(wav_name, channel = 0):
#     """
#     Demuxing the wav file
#     """
#     try:
#         multi_channel_data, sr = load_file(wav_name)
#         if multi_channel_data.ndim <= 1:
#             return sr, multi_channel_data

#         return sr, np.array(multi_channel_data)[channel, :]

#     except ValueError as msg:
#         print(f'{msg}')

# def file_to_vectorArr(file_name, n_mels = 64, frames = 5, n_fft = 1024, hop_length = 512, power = 2):
#     """
#     Creating a vector array from the file and generating the 
#     """
#     dims = n_mels * frames

#     sr, y = load_file(file_name)

#     # Generating a Mel Spectrogram
#     mel_spec = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = n_fft, 
#                                               hop_length = hop_length, n_mels = n_mels, power = power)

#     # Converting Mel Spectrogram to log mel energy
#     log_mel = librosa.power_to_db(mel_spec)

#     vecArr_size = len(log_mel[0, :]) - frames + 1

#     # Skipping too short clips
#     if vecArr_size < 1:
#         return np.empty((0, dims), float)

#     vectorarray = np.zeros((vecArr_size, dims), float)
#     for t in range(frames):
#         vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel[:, t: t + vecArr_size].T

#     return vectorarray

# def list_to_vectorArr(file_list, n_mels = 64, frames = 5, n_fft = 1024, hop_length = 512, power = 2):
#     dims = n_mels * frames

#     for idx in range(len(file_list)):
#         vecArr = file_to_vectorArr(file_list[idx], n_mels = n_mels, frames = frames, n_fft = n_fft,
#                                    hop_length = hop_length, power = power)
        
#         if idx == 0:
#             dataset = np.zeros((vecArr.shape[0] * len(file_list), dims), float)

#         dataset[vecArr.shape[0] * idx: vecArr.shape[0] * (idx + 1), :] = vecArr

#         return dataset


# if __name__ == "__main__":
#     os.makedirs("ImageDirectory", exist_ok = True)

#     visualiser = visualise()
    

audioPath = 'Segmented Data\FRESH520\RPM520A0011.wav'

y, sr = librosa.load(audioPath, sr = None)

spectrum = librosa.stft(y = y)
M_spec = librosa.feature.melspectrogram(y = y, sr = sr)
S_db = librosa.amplitude_to_db(np.abs(), ref = np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, ax=ax)
fig.colorbar(img, ax=ax)

plt.show()