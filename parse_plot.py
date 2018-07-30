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


def parse_plot(vae=None, rnn=None, es=None):
    output_file('worldmodels.html')
    figs = []
    if vae is not None:
        fig = parse_vae(vae)
        figs.append(fig)
    if rnn is not None:
        fig = parse_rnn(rnn)
        figs.append(fig)
    if es is not None:
        fig = parse_es(es)
        figs.append(fig)

    save(column(*figs))


def parse_es(log_file):
    with open(log_file, 'r') as f:
        regex = r'Step ([\d]+)\s*Max_R ([\d\.]+)\s*Mean_R ([\d\.]+)\s*Min_R ([\d\.]+)'
        data = re.findall(regex, f.read())
        data = [[float(x) for x in row] for row in data]
        data = np.array(data)

    print('ES', data.shape)
    x = data[:, 0]
    colors = Category10[8]

    fig = figure(width=800, height=600, title='Rewards, Doom')
    fig.circle(x, data[:, 1], legend='Max R', line_color=colors[0], fill_color=colors[0])
    fig.circle(x, data[:, 2], legend='Mean R', line_color=colors[1], fill_color=colors[1])
    fig.circle(x, data[:, 3], legend='Min R', line_color=colors[2], fill_color=colors[2])
    return fig


def parse_rnn(log_file):

    with open(log_file, 'r') as f:
        regex = r'Z_Loss ([\d\.]+)\s*R_Loss ([\d\.]+)\s*Loss ([\d\.]+)'
        data = re.findall(regex, f.read())
        data = [[float(x) for x in row] for row in data]
        data = np.array(data)

    print('RNN', data.shape)
    x = np.arange(0, data.shape[0])
    colors = Category10[8]

    fig = figure(width=800, height=600, title='Loss, RNN Model')
    fig.circle(x, data[:, 0], legend='Z Loss', line_color=colors[0], fill_color=colors[0])
    fig.circle(x, data[:, 1], legend='R Loss', line_color=colors[1], fill_color=colors[1])
    fig.circle(x, data[:, 2], legend='Total Loss', line_color=colors[2], fill_color=colors[2])
    return fig


def parse_vae(log_file):
    with open(log_file, 'r') as f:
        regex = r'Loss ([\d\.]+)\s*R_Loss ([\d\.]+)\s*KL_Loss ([\d\.]+)\s*'
        data = re.findall(regex, f.read())
        data = [[float(x) for x in row] for row in data]
        data = np.array(data)

    print('VAE', data.shape)
    x = np.arange(0, data.shape[0])
    colors = Category10[8]

    fig = figure(width=800, height=600, title='Loss, VAE')
    fig.circle(x, data[:, 0], legend='Loss', line_color=colors[0], fill_color=colors[0])
    fig.circle(x, data[:, 1], legend='R Loss', line_color=colors[1], fill_color=colors[1])
    fig.circle(x, data[:, 2], legend='KL Loss', line_color=colors[2], fill_color=colors[2])
    return fig


fire.Fire(parse_plot)




