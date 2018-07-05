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

from model import VAE
from main import cfg
from common import Logger

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

def get_vae_data_loader():
    print("Loading Dataset")
    data_list = glob.glob(cfg.seq_save_dir +'/*.npz')
    datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_npz)(f) for f in data_list)
    datasets = [NumpyData(x) for x in datas]
    total_data = ConcatDataset(datasets)
    dataloader = DataLoader(total_data, batch_size=cfg.vae_batch_size, shuffle=True)


def vae_train():
    logger = Logger("{}/vae_train_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    dataloader = get_vae_data_loader()
    model = torch.nn.DataParallel(VAE()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)

    for epoch in range(cfg.vae_num_epoch):
        for idx, imgs in enumerate(dataloader):
            now = time.time()
            imgs = imgs.float().cuda() / 255.0
            mu, logvar, imgs_rc, z = model(imgs)

            r_loss = (imgs_rc - imgs).pow(2).view(imgs.size(0), -1).sum(dim=1).mean()

            kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            min_kl = torch.zeros_like(kl_loss) + kl_tolerance * z_size
            kl_loss = torch.max(kl_loss, min_kl).mean()

            loss = r_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            duration = time.time() - now

            if idx % 10 == 0:
                info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Loss {:6.3f}\t R_Loss {:6.3f}\t  \
                    KL_Loss {:6.3f}\t Maxvar {:6.3f}\t Speed {:6.3f}".format(
                    epoch, idx, len(dataloader), loss.item(), r_loss.item(),
                    kl_loss.item(), logvar.max().item(), imgs.size(0) / duration)

                logger.log()

        model_save_path = "{}/vae_{}_epoch_{:03d}.pth".format(
                cfg.model_save_dir, cfg.timestr, epoch)
        torch.save({'model': model.state_dict()}, model_save_path)

def test():
    data = torch.load('out.pth')
    for idx, imgs in enumerate(dataloader):
        imgs = imgs.cuda() / 255.0
        mu, logvar, imgs_rc, z = model(imgs)
        imgy = imgs_rc.detach().cpu().numpy()[0].transpose(1, 2, 0)
        imgx = imgs.detach().cpu().numpy()[0].transpose(1, 2, 0)
        cv2.imwrite('imgx.png', imgx * 255.0)
        cv2.imwrite('imgy.png', imgy * 255.0)
        break


