## How to Use

### Prerequisites

Python
Keras
Librosa 
Scipy
Numpy 
matplotlib

### Download dataset
Speechdata and hw2_1
$ bash ./get_data.sh

### Train model
$ python3 train.py

### Generate Enhancement File(wav in ./speechdata/testing_enh_snr*/)
$ python3 test.py

### Test HMM

Copy enhancement file(in ./speechdata/testing_enh_snr*/) or noisy file (in ./speechdata/testing_noise_snr*) to ./hw2_1/speechdata/test, then run
$ 01_run_HCopy.sh
$ 04_testing.sh

### Generate spectral(Need enhancement data, noisy data, test data)

$ python3 gen_spectral.py
