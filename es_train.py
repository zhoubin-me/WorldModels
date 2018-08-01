"""
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import time
import glob
import cma
import os

from mpi4py import MPI
from torch.distributions import Categorical

from model import RNNModel, Controller, VAE
from common import Logger
from config import cfg





def load_init_z():
    data = np.load('init_z.npz')
    mus = data['mus']
    logvars = data['logvars']

    return mus, logvars

def sample_init_z(mus, logvars):
    idx = np.random.randint(0, mus.shape[0])
    mu, logvar = mus[idx], logvars[idx]
    # mu /= 2.0
    # logvar /= 2.0
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


def rollout(model, controller, zs):
    rewards = []
    for epi in range(cfg.trials_per_pop):
        model.reset()
        z = zs[epi]
        for step in range(cfg.max_steps):
            z = torch.from_numpy(z).float().unsqueeze(0)
            x = torch.cat((model.hx.detach(), model.cx.detach(), z), dim=1)
            y = controller(x)
            # m = Categorical(F.softmax(y, dim=1))
            # action = m.sample()
            y = y.item()
            if y > 1 / 3.0:
                action = torch.LongTensor([1])
            elif y < -1 / 3.0:
                action = torch.LongTensor([0])
            else:
                action = torch.LongTensor([2])

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
    return np.array(rewards)


def l2_reg(x):
    x = np.array(x)
    return cfg.es_w_reg * np.mean(x * x, axis=1)

def slave(comm):
    mus, logvars = load_init_z()
    vae = VAE()
    vae.load_state_dict(torch.load(cfg.vae_save_ckpt, map_location=lambda storage, loc: storage)['model'])
    model = RNNModel()
    model.load_state_dict(torch.load(cfg.rnn_save_ckpt, map_location=lambda storage, loc: storage)['model'])
    count = 1
    status = MPI.Status()

    gpuid = comm.rank % 4
    # device = torch.device('cuda:{}'.format(gpuid))
    # vae.to(device)
    # model.to(device)
    print('Worker {} Started, model on GPU {}'.format(comm.rank, gpuid))



    while True:
        solution = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == 1:
            print('Worker {} received solution {}'.format(comm.rank, count))
            zs = [sample_init_z(mus, logvars) for _ in range(cfg.trials_per_pop)]
            controller = deflatten_controller(solution)
            reward = rollout(model, controller, zs)
            print('Worker {} finished solution {}, reward: mean {} | max {} | min {} | std {}'.format(
                comm.rank, count, reward.mean(), reward.max(), reward.min(), reward.std()))
            comm.send(reward.mean(), dest=0, tag=2)
            count += 1
        elif tag == 3:
            print('Worker {} evaluate current solution'.format(comm.rank))
            controller = deflatten_controller(solution)
            reward = evaluate(model, vae, controller)
            comm.send(reward, dest=0, tag=2)



def master(comm):
    logger = Logger("{}/es_train_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    logger.log(cfg.info)
    controller = Controller()
    es = cma.CMAEvolutionStrategy(flatten_controller(controller), cfg.es_sigma, {'popsize': cfg.population_size})


    for step in range(cfg.es_steps):
        solutions = es.ask()
        for idx, solution in enumerate(solutions):
            comm.send(solution, dest=idx+1, tag=1)

        check = np.ones(cfg.num_workers)
        rewards = []
        for idx in range(cfg.num_workers):
            reward = comm.recv(source=idx+1, tag=2)
            rewards.append(reward)
            check[idx] = 0

        assert check.sum() == 0
        assert len(rewards) == cfg.num_workers

        r_cost = - np.array(rewards)
        reg_cost = l2_reg(solutions)
        cost =  reg_cost + r_cost
        es.tell(solutions, cost.tolist())

        sigma = es.result[6]
        rms_var = np.mean(sigma * sigma)



        info = "Step {:d}\t Max_R {:4f}\t Mean_R {:4f}\t Min_R {:4f}\t RMS_Var {:4f}\t Reg_Cost {:4f}\t R_Cost {:4f}".format(
                step, max(rewards), np.mean(rewards), min(rewards), rms_var, r_cost.mean(), reg_cost.mean())
        logger.log(info)

        if step % 25 == 0:
            current_param = es.result[5]
            current_controller = deflatten_controller(current_param)
            save_path = "{}/controller_curr_{}_step_{:05d}.pth".format(cfg.model_save_dir, cfg.timestr, step)
            torch.save({'model': current_controller.state_dict()}, save_path)

            best_param = es.result[0]
            best_controller = deflatten_controller(best_param)
            save_path = "{}/controller_best_{}_step_{:05d}.pth".format(cfg.model_save_dir, cfg.timestr, step)
            torch.save({'model': best_controller.state_dict()}, save_path)

def es_train():
    comm = MPI.COMM_WORLD

    rank = comm.rank
    size = comm.size

    if rank == 0:
        master(comm)
    else:
        slave(comm)


if __name__ == '__main__':
    es_train()
