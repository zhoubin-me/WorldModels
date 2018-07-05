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

def load_init_mu_var(f):
    data = np.load(f)
    return data['mu'][0], data['logvar'][0]


data_list = glob.glob('converted/*.npz')
datas = Parallel(n_jobs=48, verbose=1)(delayed(load_init_mu_var)(f) for f in data_list)

temperature = 1.15



vae = VAE()
vae_stat_dict = torch.load('./save/ckpt_e11.pth')['model']
new_vae_stat_dict = OrderedDict()
for k, v in vae_stat_dict.items():
    new_vae_stat_dict[k[7:]] = v
vae.load_state_dict(new_vae_stat_dict)

model = RNNModel()
rnn_stat_dict = torch.load('./save/rnn_model_e200.pth')['model']
new_rnn_stat_dict = OrderedDict()
for k, v in rnn_stat_dict.items():
    new_rnn_stat_dict[k[7:]] = v
model.load_state_dict(new_rnn_stat_dict)



def sample_init_z():
    idx = np.random.randint(0, len(datas))
    mu, logvar = datas[idx]
    z = mu + np.exp(logvar / 2.0) * np.random.randn(*mu.shape)
    return z

def transform(x):
    return torch.from_numpy(np.array(x)).unsqueeze(0).unsqueeze(0)

def dream():
    frames = []
    z = sample_init_z()
    done = 0
    model.reset()
    for step in range(1000):
        action = np.random.randint(0, 2)
        z, action = [transform(x) for x in [z, action]]
        z = z.float()
        frames += [vae.decode(z).detach().numpy()]
        logmix, mu, logstd, done_p = model.step(z, action)

        logmix = logmix / temperature
        logmix -= logmix.max()
        logmix = torch.exp(logmix)

        m = Categorical(logmix)
        idx = m.sample()

        new_mu = torch.FloatTensor([mu[i, j] for i, j in enumerate(idx)])
        new_logstd = torch.FloatTensor([logstd[i, j] for i, j in enumerate(idx)])
        z_next = new_mu + new_logstd.exp() * torch.randn_like(new_mu) * np.sqrt(temperature)

        z = z_next.detach().numpy()
        print(step)


        if done_p.squeeze().item() > 0:
            break
    os.system('rm dream/*.png')
    for idx, frame in enumerate(frames):
        img = frame.transpose(2, 3, 1, 0) * 255.0
        cv2.imwrite('dream/frame_{:03d}.png'.format(idx), img[:, :, :, 0])
    print('Making GIF')
    os.system('convert -delay 10 -loop 0 dream/*.png dream.gif')
    os.system('scp dream.gif bzhou@10.80.43.125:/home/bzhou/Dropbox')

def real():

    data = glob.glob('record/*.npz')
    data = choice(data)
    frames = np.load(data)['sx']

    for idx, frame in enumerate(frames):
        cv2.imwrite('real/frame_{:03d}.png'.format(idx), frame)

dream()





