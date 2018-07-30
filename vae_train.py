import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import glob
import time

from joblib import Parallel, delayed

from config import cfg
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

def load_npz(f):
    return np.load(f)['sx'].transpose(0, 3, 1, 2)

def vae_train():
    logger = Logger("{}/vae_train_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    logger.log(cfg.info)

    logger.log("Loading Dataset")

    data_list = glob.glob(cfg.seq_save_dir +'/*.npz')
    datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_npz)(f) for f in data_list)

    datasets = [NumpyData(x) for x in datas]
    total_data = ConcatDataset(datasets)
    train_data_loader = DataLoader(total_data, batch_size=cfg.vae_batch_size, shuffle=True, num_workers=10, pin_memory=False)

    print('Total frames: {}'.format(len(total_data)))

    model = torch.nn.DataParallel(VAE()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)

    for epoch in range(cfg.vae_num_epoch):
        current_loss = 0
        now = time.time()
        for idx, imgs in enumerate(train_data_loader):
            data_duration = time.time() - now

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
            current_loss += loss.item() * imgs.size(0)

            model_duration = time.time() - now
            total_duration = data_duration + model_duration
            if idx % 10 == 0:
                info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Loss {:6.3f}\t R_Loss {:6.3f}\t \
                        KL_Loss {:6.3f}\t Maxvar {:6.3f}\t Speed {:6.3f}\t Time: [{:.5f}/{:.5f}]\t".format(
                    epoch, idx, len(train_data_loader), loss.item(), r_loss.item(),
                    kl_loss.item(), logvar.max().item(), imgs.size(0) / total_duration, data_duration, total_duration)
                logger.log(info)

            now = time.time()


        to_save_data = {'model': model.module.state_dict(), 'loss': current_loss}
        to_save_path = '{}/vae_{}_e{:03d}.pth'.format(cfg.model_save_dir, cfg.timestr, epoch)
        torch.save(to_save_data, to_save_path)


if __name__ == '__main__':
    vae_train()
