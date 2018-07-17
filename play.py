import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import glob
import numpy as np
from joblib import Parallel, delayed
from collections import OrderedDict
import cv2
import os

from model import VAE, RNNModel, Controller
from collect_data import initialize_vizdoom, preprocess
from es_train import sample_init_z, load_init
from main import cfg



def write_video(frames, fname, reward=None):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(fname, fourcc, 10, (64, 64))
    for frame in frames:
        video.write(frame.astype(np.uint8))

def play_in_dream():
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


    if cfg.ctrl_save_ckpt is not None:
        controller = Controller()
        ctrl_stat_dict = torch.load(cfg.ctrl_save_ckpt)['model']
        controller.load_state_dict(ctrl_stat_dict)
    else:
        controller = None

    rewards = []
    frames = []

    for epi in range(cfg.trials_per_pop):
        done = 0
        model.reset()
        frames_ = []
        z = sample_init_z(datas)

        for step in range(cfg.max_steps):

            z = torch.from_numpy(z).float().unsqueeze(0)
            if controller is not None:
                x = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
                y = controller(x)
                m = Categorical(F.softmax(y, dim=1))
                action = m.sample()
            else:
                action = torch.randint(0, 2, (1,))


            frames_ += [vae.decode(z).detach().numpy().transpose(2, 3, 1, 0)[:, :, :, 0] * 255.0]

            logmix, mu, logstd, done_p = model.step(z.unsqueeze(0), action.unsqueeze(0))
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

        rewards.append(step)
        frames.append(frames_)

    idx = rewards.index(max(rewards))
    frames = frames[idx]

    print(rewards)
    print(len(frames))

    write_video(frames, 'temp/dream.avi')
    os.system('scp temp/dream.avi bzhou@10.80.43.125:/home/bzhou/Dropbox/share')


def play_in_real():

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

    if cfg.ctrl_save_ckpt is not None:
        controller = Controller()
        ctrl_stat_dict = torch.load(cfg.ctrl_save_ckpt)['model']
        controller.load_state_dict(ctrl_stat_dict)
    else:
        controller = None

    game = initialize_vizdoom()
    actions = cfg.game_actions
    frames = []
    rewards = []

    for epi in range(cfg.trials_per_pop):
        game.new_episode()
        repeat = np.random.randint(1, 11)
        model.reset()
        frames_ = []
        for step in range(cfg.max_steps):
            s1 = game.get_state().screen_buffer
            s1 = preprocess(s1)
            frames_ += [s1]
            s1 = torch.from_numpy(s1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            mu, logvar, x, z = vae(s1)

            if controller is not None:
                inp = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
                y = controller(inp)
                m = Categorical(F.softmax(y, dim=1))
                action = m.sample().item()
            else:
                action = np.random.randint(0, 2)
            action = actions[action]


            reward = game.make_action(action)
            done = game.is_episode_finished()
            if done:
                break

        rewards.append(step)
        frames.append(frames_)
    idx = rewards.index(max(rewards))
    frames = frames[idx]

    print(rewards)
    print(len(frames))


    write_video(frames, 'temp/play.avi')
    os.system('scp temp/play.avi bzhou@10.80.43.125:/home/bzhou/Dropbox/share')


















