import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from joblib import Parallel, delayed
import glob
import pdb
import time
from model import RNNModel

batch_size = 100
seq_length = 500
z_size = 64
rnn_size = 512

data_list = glob.glob('converted/*.npz')

def load_npz(f):
    data = np.load(f)
    return data['mu'], data['logvar'], data['actions'], data['rewards'], data['dones']

datas = Parallel(n_jobs=48, verbose=1)(delayed(load_npz)(f) for f in data_list)

class SeqData(Dataset):
    def __init__(self, mu, logvar, actions, rewards, dones):
        total_frames = mu.shape[0]
        num_batches = total_frames // seq_length
        N = num_batches * seq_length

        self.mu = mu[:N].reshape(-1, seq_length, z_size)
        self.logvar = logvar[:N].reshape(-1, seq_length, z_size)
        self.actions = actions[:N].reshape(-1, seq_length)
        self.rewards = rewards[:N].reshape(-1, seq_length)
        self.dones = dones[:N].reshape(-1, seq_length)

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        return self.mu[idx], self.logvar[idx], self.actions[idx], self.rewards[idx], self.dones[idx]


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = (0.001 - 0.00001) * 0.99999 ** step + 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


model = torch.nn.DataParallel(RNNModel()).cuda()
optimizer = torch.optim.Adam(model.parameters())


start_epoch = 0
model.load_state_dict(torch.load('save/rnn_model_e{:03d}.pth'.format(start_epoch))['model'])

for epoch in range(start_epoch+1, 600):
    np.random.shuffle(datas)
    data = map(np.concatenate, zip(*datas))
    dataset = SeqData(*data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for idx, idata in enumerate(dataloader):
        # mu, logvar, actions, rewards, dones
        now = time.time()
        adjust_learning_rate(optimizer, idx)
        idata = list(x.cuda() for x in idata)
        z = idata[0] + torch.exp(idata[1] / 2.0) * torch.randn_like(idata[1])
        target_z = z[:, 1:, :].contiguous().view(-1, 1)
        target_d = idata[-1][:, 1:].float()

        if z.size(0) != batch_size:
            continue

        logmix, mu, logstd, done_p = model(z, idata[2], idata[4])
        logmix -= logmix.exp().sum(dim=1, keepdim=True).log()


        v = logmix - 0.5 * ((target_z - mu) / torch.exp(logstd)) ** 2 - logstd - model.module.const
        v = v.exp().sum(dim=1, keepdim=True).log()
        z_loss = - v.mean()

        r_loss = F.binary_cross_entropy_with_logits(done_p, target_d, reduce=False)
        r_factor = torch.ones_like(r_loss) + target_d * 9
        r_loss = torch.mean(r_loss * r_factor)

        loss = z_loss + r_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            duration = time.time() - now
            info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Z_Loss {:5.3f}\t R_Loss {:5.3f}\t Loss {:5.3f}\t Speed {:5.2f}".format(
                    epoch, idx, len(dataloader), z_loss.item(), r_loss.item(), loss.item(), batch_size / duration)
            print(info)

    if epoch % 10 == 0:
        torch.save({'model': model.state_dict()}, 'save/rnn_model_e{:03d}.pth'.format(epoch))