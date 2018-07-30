import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import glob
import cv2
import numpy as np
import os

from model import VAE, RNNModel, Controller
from es_train import load_init_z, sample_init_z
from config import cfg


def write_video(frames, fname, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(fname, fourcc, 20, size)
    for frame in frames:
        video.write(frame.astype(np.uint8))

def test_rnn():
    mus, logvars = load_init_z()

    vae = VAE()
    vae.load_state_dict(torch.load(cfg.vae_save_ckpt)['model'])
    model = RNNModel()
    model.load_state_dict(torch.load(cfg.rnn_save_ckpt)['model'])

    model.reset()
    z = sample_init_z(mus, logvars)
    frames = []

    for step in range(cfg.max_steps):
        z = torch.from_numpy(z).float().unsqueeze(0)
        curr_frame = vae.decode(z).detach().numpy()
        frames.append(curr_frame.transpose(0, 2, 3, 1)[0] * 255.0)
        x = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
        action = torch.randint(0, 3, (1, ))

        logmix, mu, logstd, done_p = model.step(z.unsqueeze(0), action.unsqueeze(0))

        # logmix = logmix - reduce_logsumexp(logmix)
        logmix_max = logmix.max(dim=1, keepdim=True)[0]
        logmix_reduce_logsumexp = (logmix - logmix_max).exp().sum(dim=1, keepdim=True).log() + logmix_max
        logmix = logmix - logmix_reduce_logsumexp

        # Adjust temperature
        logmix = logmix / cfg.temperature
        logmix -= logmix.max(dim=1, keepdim=True)[0]
        logmix = F.softmax(logmix, dim=1)

        m = Categorical(logmix)
        idx = m.sample()

        new_mu = torch.FloatTensor([mu[i, j] for i, j in enumerate(idx)])
        new_logstd = torch.FloatTensor([logstd[i, j] for i, j in enumerate(idx)])
        z_next = new_mu + new_logstd.exp() * torch.randn_like(new_mu) * np.sqrt(cfg.temperature)

        z = z_next.detach().numpy()
        if done_p.squeeze().item() > 0:
            break

    print('RNN Reward {}'.format(step))
    write_video(frames, 'rnn.avi')
    os.system('mv rnn.avi /home/bzhou/Dropbox/share')







def test_vae():
    data = glob.glob('../../data/doom_frames/*.npz')
    data = np.random.choice(data)
    frames = np.load(data)['sx']

    model = VAE().cuda(3)
    stat_dict = torch.load(cfg.vae_save_ckpt)['model']
    model.load_state_dict(stat_dict)
    x = frames.transpose(0, 3, 1, 2)
    x = torch.from_numpy(x).float().cuda(3) / 255.0
    _, _, x_rec, _ = model(x)
    x_rec = x_rec.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

    new_frames = np.zeros((x_rec.shape[0], x_rec.shape[1]*2 + 20, x_rec.shape[2], x_rec.shape[3]))
    new_frames[:, :x_rec.shape[1], :, :] = x_rec
    new_frames[:, -x_rec.shape[1]:, :, :] = frames.astype(np.float)

    print(new_frames.shape, 'vae.avi')

    write_video(new_frames, 'vae.avi', (64, 148))
    os.system('mv vae.avi /home/bzhou/Dropbox/share')




def test_frames():
    data = glob.glob('../../data/doom_frames/*.npz')
    data = np.random.choice(data)
    frames = np.load(data)['sx']

    print(frames.shape, 'data.avi')

    write_video(frames, 'data.avi')
    os.system('mv data.avi /home/bzhou/Dropbox/share')

# test_frames()
# test_vae()
test_rnn()

# play_data()















