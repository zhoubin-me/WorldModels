import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import glob
import cv2
import pdb
import time

from joblib import Parallel, delayed
from multiprocessing import Process
from collections import OrderedDict

from main import cfg
from common import Logger
from model import VAE


class NumpyData(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def vae_train():
    logger = Logger("{}/vae_train_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    logger.log(cfg.info)

    print("Loading Dataset")
    def load_npz(f):
        return np.load(f)['sx'].transpose(0, 3, 1, 2)
    data_list = glob.glob(cfg.seq_save_dir +'/*.npz')
    data_list.sort()
    datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_npz)(f) for f in data_list)

    datasets = [NumpyData(x) for x in datas]
    total_data = ConcatDataset(datasets)
    dataloader = DataLoader(total_data, batch_size=cfg.vae_batch_size, shuffle=True)

    model = torch.nn.DataParallel(VAE()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)

    for epoch in range(cfg.vae_num_epoch):
        for idx, imgs in enumerate(dataloader):
            now = time.time()
            imgs = imgs.float().cuda() / 255.0
            mu, logvar, imgs_rc, z = model(imgs)

            r_loss = (imgs_rc - imgs).pow(2).view(imgs.size(0), -1).sum(dim=1).mean()

            kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            min_kl = torch.zeros_like(kl_loss) + cfg.vae_kl_tolerance * cfg.vae_z_size
            kl_loss = torch.max(kl_loss, min_kl).mean()

            loss = r_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            duration = time.time() - now

            if idx % 10 == 0:
                info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Loss {:6.3f}\t R_Loss {:6.3f}\t \
                        KL_Loss {:6.3f}\t Maxvar {:6.3f}\t Speed {:6.3f}".format(
                    epoch, idx, len(dataloader), loss.item(), r_loss.item(),
                    kl_loss.item(), logvar.max().item(), imgs.size(0) / duration)

                logger.log(info)

        model_save_path = "{}/vae_{}_epoch_{:03d}.pth".format(
                cfg.model_save_dir, cfg.timestr, epoch)
        torch.save({'model': model.state_dict()}, model_save_path)


def convert(fs, idx, N):

    model = VAE().cuda(idx)
    old_stat_dict = torch.load(cfg.vae_save_ckpt)['model']
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
        p = Process(target=convert, args=(data_list[idx*N:(idx+1)*N], idx, N))
        procs.append(p)
        p.start()
        time.sleep(1)

    for p in procs:
        p.join()


