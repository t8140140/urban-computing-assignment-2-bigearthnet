import numpy as np

n_list = [1000,2000,5000,10000,15000]
long_inp = np.load("arrays/clean_input.npy")
long_fld = np.load("arrays/clean_folders.npy")
long_lbl = np.load("arrays/clean_labels.npy")

for n in n_list:
    randints = np.random.choice(19001, n, replace=False)
    np.save(f"arrays/input_{n}.npy",long_inp[randints])
    np.save(f"arrays/folders_{n}.npy",long_fld[randints])
    np.save(f"arrays/labels_{n}.npy",long_lbl[randints])