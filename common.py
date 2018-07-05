#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.
import time

from joblib import Parallel, delayed
import numpy as np
import glob

class Logger:
    def __init__(self, logpath):
        self.f = open(logpath, 'w')

    def log(self, info):
        print(info)
        self.f.write(info + '\n')
        self.f.flush()


data_list = glob.glob('converted/*.npz')
def load_npz(f):
    data = np.load(f)
    return data['mu'], data['logvar'], data['actions'], data['rewards'], data['dones']

datas = Parallel(n_jobs=48, verbose=1)(delayed(load_npz)(f) for f in data_list)

N = []
for data in datas:
    N.append(data[0].shape[0])

info = "Sum: {}, Mean: {}, Length {}".format(sum(N), sum(N) / len(N), len(N))
print(info)
