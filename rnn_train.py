import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from joblib import Parallel, delayed
import glob
import pdb
import time

from common import Logger

from main import cfg
from model import RNNModel


class SeqData(Dataset):
    def __init__(self, mu, logvar, actions, rewards, dones):
        seq_length = cfg.rnn_seq_len
        total_frames = mu.shape[0]
        num_batches = total_frames // seq_length
        N = num_batches * seq_length

        self.mu = mu[:N].reshape(-1, seq_length, cfg.vae_z_size)
        self.logvar = logvar[:N].reshape(-1, seq_length, cfg.vae_z_size)
        self.actions = actions[:N].reshape(-1, seq_length)
        self.rewards = rewards[:N].reshape(-1, seq_length)
        self.dones = dones[:N].reshape(-1, seq_length)

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        return self.mu[idx], self.logvar[idx], self.actions[idx], \
                self.rewards[idx], self.dones[idx]


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = (cfg.rnn_lr_max - cfg.rnn_lr_min) * cfg.rnn_lr_decay ** step + cfg.rnn_lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def load_npz(f):
    data = np.load(f)
    return data['mu'], data['logvar'], data['actions'], data['rewards'], data['dones']

def rnn_train():
    logger = Logger("{}/rnn_train_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    logger.log(cfg.info)

    data_list = glob.glob(cfg.seq_extract_dir + '/*.npz')
    datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_npz)(f) for f in data_list)

    model = torch.nn.DataParallel(RNNModel()).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    global_step = 0

    for epoch in range(cfg.rnn_num_epoch):
        np.random.shuffle(datas)
        data = map(np.concatenate, zip(*datas))
        dataset = SeqData(*data)
        dataloader = DataLoader(dataset, batch_size=cfg.rnn_batch_size, shuffle=False)

        for idx, idata in enumerate(dataloader):
            # mu, logvar, actions, rewards, dones
            now = time.time()
            lr = adjust_learning_rate(optimizer, global_step)

            with torch.no_grad():
                idata = list(x.cuda() for x in idata)
                z = idata[0] + torch.exp(idata[1] / 2.0) * torch.randn_like(idata[1])
                target_z = z[:, 1:, :].contiguous().view(-1, 1)
                target_d = idata[-1][:, 1:].float()

            if z.size(0) != cfg.rnn_batch_size:
                continue

            logmix, mu, logstd, done_p = model(z, idata[2], idata[4])

            v = logmix - 0.5 * ((target_z - mu) / torch.exp(logstd)) ** 2
            v = v - logstd - cfg.logsqrt2pi
            v = v.exp().sum(dim=1, keepdim=True).log()
            z_loss = -v.mean()

            r_loss = F.binary_cross_entropy_with_logits(done_p, target_d, reduce=False)
            r_factor = torch.ones_like(r_loss) + target_d * cfg.rnn_r_loss_w
            r_loss = torch.mean(r_loss * r_factor)

            loss = z_loss + r_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            duration = time.time() - now

            if idx % 10 == 0:
                info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Z_Loss {:5.3f}\t \
                        R_Loss {:5.3f}\t Loss {:5.3f}\t LR {:.5f}\t Speed {:5.2f}".format(
                                epoch, idx, len(dataloader), z_loss.item(),
                                r_loss.item(), loss.item(), lr, cfg.rnn_batch_size / duration)
                logger.log(info)

        if (epoch + 1) % 10 == 0:
            model_save_path = "{}/rnn_{}_epoch_{:03d}.pth".format(
                    cfg.model_save_dir, cfg.timestr, epoch)
            torch.save({'model': model.state_dict()}, model_save_path)

