from joblib import Parallel, delayed
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import time
from collections import OrderedDict
from multiprocessing import Process
from model import VAE




data_list = glob.glob('record/*.npz')
N = len(data_list) // 4

def convert(fs, idx):
    model = VAE().cuda(idx)
    old_stat_dict = torch.load('./save/vae_e11.pth')['model']
    stat_dict = OrderedDict()
    for k, v in old_stat_dict.items():
        stat_dict[k[7:]] = v
    model.load_state_dict(stat_dict)
    for n, f in enumerate(fs):
        data = np.load(f)
        imgs = data['sx'].transpose(0, 3, 1, 2)
        actions = data['ax']
        rewards = data['rx']
        dones = data['dx']
        x = torch.from_numpy(imgs).float().cuda(idx) / 255.0
        mu, logvar, _, _ = model(x)
        np.savez_compressed('converted/' + f.split('/')[-1], mu=mu.detach().cpu().numpy(), logvar=logvar.detach().cpu().numpy(), dones=dones, rewards=rewards, actions=actions)

        if n % 10 == 0:
            print('Process %d: %5d / %5d' % (idx, n, N))

procs = []
for idx in range(4):
    p = Process(target=convert, args=(data_list[idx*N:(idx+1)*N], idx))
    procs.append(p)
    p.start()
    time.sleep(1)

for p in procs:
    p.join()


