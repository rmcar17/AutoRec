import numpy as np

def read_data(fpath, ncol=1682):
    with open(fpath, 'r') as f:
        data = np.genfromtxt(f, delimiter="\t", dtype=int)
        data = data[:, :-1]

    if ncol is None:
        ncol = max(data[:, 1])

    rw_data = np.zeros((max(data[:, 0]), ncol))
    for u, i, r in data:
        rw_data[u-1, i-1] = r - 2.5
        
    return np.array(rw_data, dtype=float)