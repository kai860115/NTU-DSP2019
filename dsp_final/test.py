from keras.models import load_model
import os
import glob
import math
import scipy
import librosa
import librosa.display
import numpy as np


SR = 8000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256
WINDOW = scipy.signal.hamming
FRAMEWIDTH = 5
test_dirs = [{'in': './speechdata/testing_noise_snr-5/', 'out': './speechdata/testing_enh_snr-5/'},
             {'in': './speechdata/testing_noise_snr0/',
                 'out': './speechdata/testing_enh_snr0/'},
             {'in': './speechdata/testing_noise_snr5/', 'out': './speechdata/testing_enh_snr5/'}]
model_path = './model.h5'

for d in test_dirs:
    if not os.path.exists(d['out']):
        os.makedirs(d['out'])


model = load_model(model_path)

for d in test_dirs:
    for f in os.listdir(d['in']):
        x_test = []
        out_path = d['out']+f
        y, sr = librosa.load(d['in'] + f, sr=SR)
        X = librosa.stft(y, n_fft=N_FFT, win_length=WIN_LENGTH,
                         hop_length=HOP_LENGTH, window=WINDOW)
        phase = np.exp(1j * np.angle(X))
        S = np.log10(abs(X)**2)
        Mean = np.mean(S, axis=1).reshape(S.shape[0], 1)
        Std = np.std(S, axis=1).reshape(S.shape[0], 1)
        S = (S - Mean)/Std
        for i in range(FRAMEWIDTH//2, S.shape[1] - FRAMEWIDTH//2):
            x_test.append(S[:, i - FRAMEWIDTH // 2: i + FRAMEWIDTH // 2 + 1])
        x_test = np.array(x_test)
        x_test = x_test.reshape(len(x_test), x_test.size//len(x_test))
        y_test = model.predict(x_test)
        for i in range(FRAMEWIDTH//2, S.shape[1] - FRAMEWIDTH//2):
            S[:, i] = y_test[i - FRAMEWIDTH//2]
        for i in range(0, FRAMEWIDTH // 2):
            S[:, i] = S[:, FRAMEWIDTH // 2]
            S[:, -1 * i - 1] = S[:, S.shape[1] - FRAMEWIDTH//2 - 1]
        S = np.sqrt(10 ** S)
        X_ = np.multiply(S, phase)
        y_ = librosa.istft(X_, hop_length=HOP_LENGTH,
                           win_length=WIN_LENGTH, window=WINDOW)
        y_ = librosa.util.fix_length(y_, y.shape[0])
        scipy.io.wavfile.write(out_path, SR, np.int16(y_*32767))
