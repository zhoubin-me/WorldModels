import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import glob
import cv2
import numpy as np
import os

from collect_data import DoomTakeOver
from model import VAE, RNNModel, Controller
from es_train import load_init_z, sample_init_z
from config import cfg
from common import Logger



def slave(comm):

    vae = VAE()
    vae.load_state_dict(torch.load(cfg.vae_save_ckpt, map_location=lambda storage, loc: storage)['model'])

    model = RNNModel()
    model.load_state_dict(torch.load(cfg.rnn_save_ckpt, map_location=lambda storage, loc: storage)['model'])

    controller = Controller()
    controller.load_state_dict(torch.load(cfg.ctrl_save_ckpt, map_location=lambda storage, loc: storage)['model'])

    env = DoomTakeOver(False)

    rewards = []
    for epi in range(cfg.trials_per_pop * 4):
        obs = env.reset()
        model.reset()
        for step in range(cfg.max_steps):
            obs = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            mu, logvar, _, z = vae(obs)

            inp = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
            y = controller(inp)

            y = y.item()
            if y > 1 / 3.0:
                action = torch.LongTensor([1])
            elif y < -1 / 3.0:
                action = torch.LongTensor([2])
            else:
                action = torch.LongTensor([0])

            model.step(z.unsqueeze(0), action.unsqueeze(0))
            obs_next, reward, done, _ = env.step(action.item())
            obs = obs_next
            if done:
                break
        rewards.append(step)
        print('Workder {} got reward {} at epi {}'.format(comm.rank, step, epi))
    rewards = np.array(rewards)
    comm.send(rewards, dest=0, tag=1)
    print('Worker {} sent rewards to master'.format(comm.rank))

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    if rank == 0:
        f = open('result.txt', 'a')
        rewards = []
        for idx in range(size):
            reward = comm.recv(source=idx+1, tag=1)
            print('Master received rewards from slave {}'.format(idx))
            rewards.append(reward)
        rewards = np.array(rewards)
        info = 'Mean {}\t Max {}\t Min {}\t Std {}'.format(rewards.mean(), rewards.max(), rewards.min(), rewards.std())
        f.write(info)
        f.flush()
        f.close()
        print(info)
    else:
        slave(comm)












