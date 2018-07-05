import numpy as np
import glob
import time
from tqdm import tqdm
from joblib import Parallel, delayed


now = time.time()
files = glob.glob('converted/*.npz')
def load_npz(f):
    data = np.load(f)
    return data['mu'], data['logvar'], data['actions'], data['rewards'], data['dones']

datas = Parallel(n_jobs=48)(delayed(load_npz)(f) for f in files)

print('Read files in ', time.time() - now, ' seconds')
now = time.time()
data = list(zip(*datas))
print(len(data[0]))
print('Concat in ', time.time() - now, 'seconds')

now = time.time()
np.savez_compressed('series2.npz', action=data[2], mu=data[0], logvar=data[1])
print('Save in ', time.time() - now, ' seconds')
