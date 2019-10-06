import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import re
import os

SR = 8000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256
WINDOW = scipy.signal.hamming
FRAMEWIDTH = 5
outdir = "./spectral/"
indirs = ["./speechdata/testing/",
          "./speechdata/testing_enh_snr-5/",
          "./speechdata/testing_enh_snr0/",
          "./speechdata/testing_enh_snr5/",
          "./speechdata/testing_noise_snr-5/",
          "./speechdata/testing_noise_snr0/",
          "./speechdata/testing_noise_snr5/"]

if not os.path.exists(outdir):
    os.makedirs(outdir)

for t in indirs:
    for i in os.listdir(t):
        infile = os.path.join(t, i)
        y, sr = librosa.load(infile, sr=8000)
        X = librosa.stft(y, n_fft=N_FFT, win_length=WIN_LENGTH,
                         hop_length=HOP_LENGTH, window=scipy.signal.hamming)
        S = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15, 5))
        plt.title("noise SNR = 5db", fontsize=20)
        librosa.display.specshow(
            S, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear', cmap="jet")
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(os.path.join(outdir, i.split('.')[
                    0] + "_" + t.split('/')[-2] + '.png'))
        plt.close('all')
        print(infile)
