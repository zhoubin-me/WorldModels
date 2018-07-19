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
import cma
import os

import multiprocessing as mp
from joblib import Parallel, delayed
from collections import OrderedDict
from torch.distributions import Categorical

from model import RNNModel, Controller, VAE
from common import Logger
from main import cfg
from collect_data import initialize_vizdoom, preprocess



def load_init(f):
    data = np.load(f)
    return data['mu'][0], data['logvar'][0]

def load_or_save_init_z():
    if os.path.exists('init_z.npz'):
        data = np.load('init_z.npz')
        mus = data['mus']
        logvars = data['logvars']
    else:
        data_list = glob.glob(cfg.seq_extract_dir + '/*.npz')
        datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_init)(f) for f in data_list)
        mus = np.array([data[0] for data in datas])
        logvars = np.array([data[1] for data in datas])
        np.savez_compressed('init_z.npz', mus=mus, logvars=logvars)
    return mus, logvars

def sample_init_z(mus, logvars):
    idx = np.random.randint(0, mus.shape[0])
    mu, logvar = mus[idx], logvars[idx]
    z = mu + np.exp(logvar / 2.0) * np.random.randn(*mu.shape)
    return z

def flatten_controller(controller):
    param_array = []
    for param in controller.parameters():
        param_array.append(param.data.view(-1))

    param_array = torch.cat(param_array, dim=0)
    return param_array.numpy()


def deflatten_controller(param_array):
    controller = Controller()
    for param in controller.parameters():
        size = param.data.view(-1).size(0)
        param.data = torch.FloatTensor(param_array[:size]).view_as(param.data)
        param_array = param_array[size:]
    return controller


def rollout(model_x, cs, zs):

    model = copy.deepcopy(model_x)
    out = []

    for c, z in zip(cs, zs):
        rewards = []
        for epi in range(cfg.trials_per_pop):
            model.reset()
            for step in range(cfg.max_steps):
                z = torch.from_numpy(z).float().unsqueeze(0)
                x = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
                y = c(x)
                m = Categorical(F.softmax(y, dim=1))
                action = m.sample()

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
            rewards.append(step)
        out.append(np.mean(rewards))
    return out


def evaluate(model_x, vae_x, controller):
    model = copy.deepcopy(model_x)
    vae = copy.deepcopy(vae_x)
    game = initialize_vizdoom()
    rewards = []

    for epi in range(cfg.trials_per_pop):
        game.new_episode()
        model.reset()

        for step in range(cfg.max_steps):
            s1 = game.get_state().screen_buffer
            s1 = preprocess(s1)
            s1 = torch.from_numpy(s1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            mu, logvar, x, z = vae(s1)

            inp = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
            y = controller(inp)
            m = Categorical(F.softmax(y, dim=1))
            action = m.sample().item()
            action = cfg.game_actions[action]

            reward = game.make_action(action)
            done = game.is_episode_finished()

            if done:
                break
        rewards.append(step)
    return rewards

def es_train():
    logger = Logger("{}/es_train_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    logger.log(cfg.info)

    mus, logvars = load_or_save_init_z()

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

    controller = Controller()

    es = cma.CMAEvolutionStrategy(flatten_controller(controller), cfg.es_sigma, {'popsize': cfg.population_size})

    for step in range(cfg.es_steps):
        init_zs = [sample_init_z(mus, logvars) for _ in range(cfg.population_size)]
        solutions = es.ask()
        new_controllers = [deflatten_controller(s) for s in solutions]

        N = cfg.population_size // cfg.num_workers
        inp = [(new_controllers[idx*N:(idx+1)*N], init_zs[idx*N:(idx+1)*N]) for idx in range(cfg.num_workers)]
        rewards = Parallel(n_jobs=cfg.num_workers, verbose=0)(delayed(rollout)(model, cs, zs) for cs, zs in inp)
        rewards = sum(rewards, [])
        # rewards = Parallel(n_jobs=cfg.num_workers, verbose=0)(delayed(evaluate)(model, vae, c) for c in new_controllers)
        info = "Step {:d}\t Max_R {:4f}\t Mean_R {:4f}\t Min_R {:4f}".format(step, max(rewards), np.mean(rewards), min(rewards))
        logger.log(info)
        cost = [-x for x in rewards]
        es.tell(solutions, cost)

        idx = rewards.index(max(rewards))
        best_controller = new_controllers[idx]
        reward = evaluate(model, vae, best_controller)
        info = "BEST--Step {:d}\t Max_R {:4f}\t Mean_R {:4f}\t Min_R {:4f}".format(step, np.max(reward), np.mean(reward), np.min(reward))
        logger.log(info)
        save_path = "{}/controller_{}_step_{:05d}.pth".format(cfg.model_save_dir, cfg.timestr, step)
        torch.save({'model': best_controller.state_dict()}, save_path)
