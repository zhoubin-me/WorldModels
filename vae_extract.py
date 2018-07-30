import torch

from model import VAE
from multiprocessing import Process

import time
import numpy as np
import glob
from common import Logger
from config import cfg

def extract(fs, idx, N):
    model = VAE()
    model.load_state_dict(torch.load(cfg.vae_save_ckpt, map_location=lambda storage, loc: storage)['model'])
    model = model.cuda(idx)


    for n, f in enumerate(fs):
        data = np.load(f)
        imgs = data['sx'].transpose(0, 3, 1, 2)
        actions = data['ax']
        rewards = data['rx']
        dones = data['dx']
        x = torch.from_numpy(imgs).float().cuda(idx) / 255.0
        mu, logvar, _, z = model(x)
        save_path = "{}/{}".format(cfg.seq_extract_dir, f.split('/')[-1])

        np.savez_compressed(save_path,
                mu=mu.detach().cpu().numpy(),
                logvar=logvar.detach().cpu().numpy(),
                dones=dones,
                rewards=rewards,
                actions=actions)

        if n % 10 == 0:
            print('Process %d: %5d / %5d' % (idx, n, N))

def vae_extract():
    logger = Logger("{}/vae_extract_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    logger.log(cfg.info)

    print("Loading Dataset")
    data_list = glob.glob(cfg.seq_save_dir +'/*.npz')
    data_list.sort()
    N = len(data_list) // 4

    procs = []
    for idx in range(4):
        p = Process(target=extract, args=(data_list[idx*N:(idx+1)*N], idx, N))
        procs.append(p)
        p.start()
        time.sleep(1)

    for p in procs:
        p.join()

if __name__ == '__main__':
    vae_extract()
