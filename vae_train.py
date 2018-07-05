import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from joblib import Parallel, delayed
import numpy as np
import glob
import cv2
from model import VAE
import pdb
import time

class NumpyData(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

data_list = glob.glob('record/*.npz')
print("Loading Dataset")
datas = Parallel(n_jobs=48, verbose=1)(delayed(lambda x: np.load(x)['sx'].transpose(0, 3, 1, 2))(f) for f in data_list[:])
datasets = [NumpyData(x) for x in datas]
total_data = ConcatDataset(datasets)
dataloader = DataLoader(total_data, batch_size=1024, shuffle=True)
kl_tolerance = 0.5
z_size = 64

model = torch.nn.DataParallel(VAE()).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train():
    for epoch in range(12):
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
                info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Loss {:6.3f}\t R_Loss {:6.3f}\t KL_Loss {:6.3f}\t Maxvar {:6.3f}\t Speed {:6.3f}".format(
                        epoch, idx, len(dataloader), loss.item(), r_loss.item(), kl_loss.item(), logvar.max().item(), imgs.size(0) / duration)
                print(info)

        torch.save({'model': model.state_dict()}, 'save/vae_e%02d.pth' % epoch)

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


def main():
    train()
    # test()

main()









