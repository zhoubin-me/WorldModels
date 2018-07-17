#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.

"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


import cma
from joblib import Parallel, delayed
import gym


def rollout(controller):
    env = gym.make('CartPole-v0')
    rewards = []
    for epi in range(10):
        x = env.reset()
        for step in range(2000):
            x = torch.from_numpy(x).unsqueeze(0).float()
            y = controller(x).squeeze().detach().numpy()
            a = np.argmax(y)
            x, r, d, _ = env.step(a)

            if d:
                rewards.append(step)
                break
    return np.mean(rewards)





class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

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

def main():
    controller = Controller()
    es = cma.CMAEvolutionStrategy(flatten_controller(controller), 0.1, {'popsize': 50})

    for step in range(100):
        solutions = es.ask()
        controllers = [deflatten_controller(x) for x in solutions]
        rewards = Parallel(n_jobs=48, verbose=1)(delayed(rollout)(x) for x in controllers)
        cost = [-x for x in rewards]
        es.tell(solutions, cost)
        es.disp()
        info = "Step {:d}\t Max_R {:4f}\t Mean_R {:4f}\t Min_R {:4f}".format(step, max(rewards), np.mean(rewards), min(rewards))
        print(info)


if __name__ == '__main__':
    main()
