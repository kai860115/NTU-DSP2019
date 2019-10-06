import keras
import os
import glob
import math
import scipy
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint

noise_dirs = ['./speechdata/training_noise_snr-5/', './speechdata/training_noise_snr0/',
              './speechdata/training_noise_snr5/']
clean_dir = './speechdata/training/'
nd_outfile = './speechdata/noise_data.npy'
cd_outfile = './speechdata/clean_data.npy'
model_path = './model.h5'
files = os.listdir(clean_dir)
SR = 8000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256
WINDOW = scipy.signal.hamming
FRAMEWIDTH = 5


def get_noise_data(dirs, files):
    noise_data = []
    for d in noise_dirs:
        for f in files:
            y, sr = librosa.load(d + f, sr=SR)
            X = librosa.stft(y, n_fft=N_FFT, win_length=WIN_LENGTH,
                             hop_length=HOP_LENGTH, window=WINDOW)
            S = np.log10(abs(X)**2)
            Mean = np.mean(S, axis=1).reshape(S.shape[0], 1)
            Std = np.std(S, axis=1).reshape(S.shape[0], 1)
            S = (S - Mean)/Std
            for i in range(FRAMEWIDTH//2, S.shape[1] - FRAMEWIDTH//2):
                noise_data.append(
                    S[:, i - FRAMEWIDTH // 2: i + FRAMEWIDTH // 2 + 1])
    noise_data = np.array(noise_data)
    noise_data = noise_data.reshape(
        len(noise_data), noise_data.size//len(noise_data))
    return noise_data


def get_clean_data(dirName, files):
    clean_data = []
    for f in files:
        y, sr = librosa.load(dirName + f, sr=SR)
        X = librosa.stft(y, n_fft=N_FFT, win_length=WIN_LENGTH,
                         hop_length=HOP_LENGTH, window=WINDOW)
        S = np.log10(abs(X)**2)
        for i in range(FRAMEWIDTH//2, S.shape[1] - FRAMEWIDTH//2):
            clean_data.append(S[:, i])
    clean_data = np.array(clean_data)
    return clean_data


def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(math.floor(all_data_size * percentage))

    X_all, Y_all = shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:-valid_data_size], Y_all[0:-valid_data_size]
    X_valid, Y_valid = X_all[-valid_data_size:], Y_all[-valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


def nn_model(in_shape, out_shape):
    model = Sequential()
    model.add(Dense(2048, input_shape=(in_shape,)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(out_shape))
    model.summary()
    return model


x_data = get_noise_data(noise_dirs, files)
y_data = get_clean_data(clean_dir, files)
y_data_tile = y_data
for i in range(1, len(x_data)//len(y_data)):
    y_data_tile = np.tile(y_data, (3, 1))
y_data = y_data_tile

np.save(nd_outfile, x_data)
np.save(cd_outfile, y_data)

x_data = np.load(nd_outfile)
y_data = np.load(cd_outfile)

x_train, y_train, x_valid, y_valid = split_valid_set(x_data, y_data, 0.3)

model = nn_model(x_data.shape[1], y_data.shape[1])
model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min', save_weights_only=False)
model.fit(x_train, y_train, batch_size=256, epochs=30, verbose=1, validation_data=(x_valid, y_valid),
          callbacks=[checkpointer])
