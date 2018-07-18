#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.


import re
import fire
import numpy as np
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column
from bokeh.palettes import *
import os

output_file('temp/worldmodel.html')

def parse_and_plot(vae=None, rnn=None, es=None):
    figs = []
    if vae is not None:
        figs += [parse_vae(vae)]

    if rnn is not None:
        figs += [parse_rnn(rnn)]

    if es is not None:
        figs += [parse_es(es)]

    save(column(*figs))
    os.system('scp temp/worldmodel.html bzhou@10.80.43.125:/home/bzhou/Dropbox/share')

def parse_es(log_file):


    with open(log_file, 'r') as f:
        data = re.findall(r'(Max_R\s*)([\d\.]+)\s*(Mean_R\s*)([\d\.]+)\s*(Min_R\s*)([\d\.]+)', f.read())
        rs = [(float(x[1]), float(x[3]), float(x[5])) for x in data]
        rs = np.array(rs)

    print('ES', rs.shape)
    x = np.arange(0, rs.shape[0])
    colors = Category10[8]

    fig = figure(width=800, height=600, title='Rewards, Doom')
    fig.circle(x, rs[:, 0], legend='Max R', line_color=colors[0], fill_color=colors[0])
    fig.circle(x, rs[:, 1], legend='Mean R', line_color=colors[1], fill_color=colors[1])
    fig.circle(x, rs[:, 2], legend='Min R', line_color=colors[2], fill_color=colors[2])

    return fig




def parse_rnn(log_file):

    with open(log_file, 'r') as f:
        data = re.findall(r'(Z_Loss\s*)([\d\.]+)\s*(R_Loss\s*)([\d\.]+)\s*(Loss\s*)([\d\.]+)', f.read())
        loss = [(float(x[1]), float(x[3]), float(x[5])) for x in data]
        loss = np.array(loss)

    print('RNN', loss.shape)
    x = np.arange(0, loss.shape[0])
    colors = Category10[8]

    fig = figure(width=800, height=600, title='Loss, RNN Model')
    fig.circle(x, loss[:, 0], legend='Z Loss', line_color=colors[0], fill_color=colors[0])
    fig.circle(x, loss[:, 1], legend='R Loss', line_color=colors[1], fill_color=colors[1])
    fig.circle(x, loss[:, 2], legend='Total Loss', line_color=colors[2], fill_color=colors[2])


    return fig


def parse_vae(log_file):


    with open(log_file, 'r') as f:
        data = re.findall(r'(Loss\s*)([\d\.]+)\s*(R_Loss\s*)([\d\.]+)\s*(KL_Loss\s*)([\d\.]+)', f.read())
        loss = [(float(x[1]), float(x[3]), float(x[5])) for x in data]
        loss = np.array(loss)

    print('VAE', loss.shape)
    x = np.arange(0, loss.shape[0])
    colors = Category10[8]

    fig = figure(width=800, height=600, title='Loss, VAE')
    fig.circle(x, loss[:, 0], legend='Loss', line_color=colors[0], fill_color=colors[0])
    fig.circle(x, loss[:, 1], legend='R Loss', line_color=colors[1], fill_color=colors[1])
    fig.circle(x, loss[:, 2], legend='KL Loss', line_color=colors[2], fill_color=colors[2])

    return fig


fire.Fire(parse_and_plot)




