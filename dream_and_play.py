import torch
import torch.nn as nn
from torch.distributions import Categorical

import glob
import numpy as np
from joblib import Parallel, delayed
from collections import OrderedDict

import numpy as np
from random import shuffle, choice
import cv2
import os
from model import VAE, RNNModel
from es_train import sample_init_z, load_init
from main import cfg


def transform(x):
    return torch.from_numpy(np.array(x)).unsqueeze(0).unsqueeze(0)

def write_video(frames, fname):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(fname, fourcc, 10, (64, 64))
    for frame in frames:
        video.write(frame.astype(np.uint8))

def dream():
    data_list = glob.glob(cfg.seq_extract_dir + '/*.npz')
    datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_init)(f) for f in data_list)

    vae = VAE()
    vae_stat_dict = torch.load(cfg.vae_save_ckpt)['model']
    new_vae_stat_dict = OrderedDict()
    for k, v in vae_stat_dict.items():
        new_vae_stat_dict[k[7:]] = v
    vae.load_state_dict(new_vae_stat_dict)

    model = RNNModel()
    rnn_stat_dict = torch.load(cfg.rnn_save_ckpt)['model']
    new_rnn_stat_dict = OrderedDict()
    for k, v in rnn_stat_dict.items():
        new_rnn_stat_dict[k[7:]] = v
    model.load_state_dict(new_rnn_stat_dict)

    frames = []
    z = sample_init_z(datas)
    done = 0
    model.reset()
    for step in range(1000):
        action = np.random.randint(0, 2)
        z, action = [transform(x) for x in [z, action]]
        z = z.float()
        frames += [vae.decode(z).detach().numpy().transpose(2, 3, 1, 0)[:, :, :, 0] * 255.0]
        logmix, mu, logstd, done_p = model.step(z, action)

        logmix = logmix / cfg.temperature
        logmix -= logmix.max()
        logmix = torch.exp(logmix)

        m = Categorical(logmix)
        idx = m.sample()

        new_mu = torch.FloatTensor([mu[i, j] for i, j in enumerate(idx)])
        new_logstd = torch.FloatTensor([logstd[i, j] for i, j in enumerate(idx)])
        z_next = new_mu + new_logstd.exp() * torch.randn_like(new_mu) * np.sqrt(cfg.temperature)

        z = z_next.detach().numpy()

        if done_p.squeeze().item() > 0:
            break

    write_video(frames, 'dream.avi')
    os.system('scp dream.avi bzhou@10.80.43.125:/home/bzhou/Dropbox')


def real():
    data = glob.glob(cfg.seq_save_dir + '/*.npz')
    data = choice(data)
    frames = np.load(data)['sx']

    write_video(frames, 'real.avi')
    os.system('scp real.avi bzhou@10.80.43.125:/home/bzhou/Dropbox')








