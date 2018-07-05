
from joblib import Parallel, delayed
import numpy as np
import glob

data_list = glob.glob('converted/*.npz')
def load_npz(f):
    data = np.load(f)
    return data['mu'], data['logvar'], data['actions'], data['rewards'], data['dones']

datas = Parallel(n_jobs=48, verbose=1)(delayed(load_npz)(f) for f in data_list)

N = []
for data in datas:
    N.append(data[0].shape[0])

info = "Sum: {}, Mean: {}, Length {}".format(sum(N), sum(N) / len(N), len(N))
print(info)
