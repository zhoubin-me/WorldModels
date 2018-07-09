"""
Evolutionary Strategies module for PyTorch models -- modified from https://github.com/alirezamika/evostra
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import pickle
import time
import glob
import pdb

from joblib import Parallel, delayed
from collections import OrderedDict
from torch.distributions import Categorical

from model import RNNModel, Controller
from common import Logger
from main import cfg



def load_init(f):
    data = np.load(f)
    return data['mu'][0], data['logvar'][0]

def sample_init_z(datas):
    idx = np.random.randint(0, len(datas))
    mu, logvar = datas[idx]
    z = mu + np.exp(logvar / 2.0) * np.random.randn(*mu.shape)
    return z

def sample_new_controller(old_controller):
    new_weights = OrderedDict()
    for k, v in old_controller.state_dict().items():
        noise = torch.randn_like(v) * cfg.es_sigma
        new_weights[k] = v + noise
    new_controller = Controller()
    new_controller.load_state_dict(new_weights)
    return new_controller

def dream(model_x, controller, z):

    model = copy.deepcopy(model_x)
    rewards = []

    for epi in range(cfg.trials_per_pop):
        done = 0
        model.reset()

        for step in range(cfg.max_steps):
            z = torch.from_numpy(z).float().unsqueeze(0)
            x = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
            y = controller(x)
            m = Categorical(F.softmax(y, dim=1))
            action = m.sample()

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
                rewards.append(step)
                break

    return np.mean(rewards)




def es_train():
    data_list = glob.glob(cfg.seq_extract_dir + '/*.npz')
    datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_init)(f) for f in data_list)

    model = RNNModel()
    rnn_stat_dict = torch.load(cfg.rnn_save_ckpt)['model']
    new_rnn_stat_dict = OrderedDict()
    for k, v in rnn_stat_dict.items():
        new_rnn_stat_dict[k[7:]] = v
    model.load_state_dict(new_rnn_stat_dict)

    controller = Controller()
    logger = Logger("{}/es_train_{}.log".format(cfg.logger_save_dir, cfg.timestr))

    lr = cfg.es_lr

    for step in range(cfg.es_steps):
        new_controllers = [sample_new_controller(controller) for _ in range(cfg.population_size)]
        init_zs = [sample_init_z(datas) for _ in range(cfg.population_size)]

        rewards = Parallel(n_jobs=48, verbose=1)(delayed(dream)(model, c, z) for c, z in zip(new_controllers, init_zs))

        if np.std(rewards) != 0:
            rewards_x = torch.tensor((rewards - np.mean(rewards)) / np.std(rewards)).float()

            for k, v in controller.state_dict().items():
                A = [c.state_dict()[k].unsqueeze(-1) for c in new_controllers]
                A = torch.cat(A, dim=-1)
                B = torch.matmul(A, rewards_x)

                v += lr / (cfg.population_size * cfg.es_sigma) * B
                lr *= cfg.es_lr_decay

        info = "Step {:d}\t Max_R {:4f}\t Mean_R {:4f}\t Min_R {:4f}".format(step, max(rewards), np.mean(rewards), min(rewards))
        logger.log(info)

        if step % 10 == 0:
            save_path = "{}/controller_{}_step_{:05d}.pth".format(cfg.model_save_dir, cfg.timestr, step)
            torch.save({'model': controller.state_dict()}, save_path)

