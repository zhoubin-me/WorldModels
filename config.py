#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.

"""

"""
import fire
import os
import numpy as np
import time

####################################################################
# Parameter Setup
####################################################################

class Config:
    # Data Collection Setup
    total_seq = 12000 # Total Number of Sequence to collect
    max_seq_len = 2000 #
    min_seq_len = 100

    # VAE Setting
    vae_batch_size = 128
    vae_num_epoch = 50
    vae_lr = 1e-4
    vae_image_size = 64
    vae_z_size = 64
    vae_kl_tolerance = 0.5

    vae_save_ckpt = "./ckpt/vae_2018-Jul-29@18:08:14_e012.pth"


    # RNN Model Setting
    action_embed_size = 4
    rnn_size = 512
    rnn_batch_size = 128
    rnn_seq_len = 500
    rnn_num_epoch = 400
    num_mixtures = 5
    rnn_lr_min = 0.00001
    rnn_lr_max = 0.001
    rnn_lr_decay = 0.99999
    rnn_r_loss_w = 9

    rnn_save_ckpt = "./ckpt/rnn_2018-Jul-29@20:37:43_e080.pth"


    # Controller ES Setting
    max_steps = 2100
    es_steps = 2000
    es_sigma = 0.1
    es_w_reg = 0.01

    num_workers = 20
    population_size = 20
    temperature = 1.15
    trials_per_pop = 16
    eval_step = 25

    ctrl_save_ckpt = None


    # Game Related Setting
    game_cfg_path = "./scenarios/take_cover.cfg"
    game_actions = [[True, False], [False, True], [False, False]]

    action_repeat = 12          # maximum action repeat
    resolution = (64, 64)       # frame resolution to resize to

    # Others
    logsqrt2pi = np.log(np.sqrt(2.0 * np.pi))
    num_cpus = 20
    timestr = ""
    task = ""
    seq_save_dir = "../../data/doom_frames"
    seq_extract_dir = "../../data/doom_extracted"
    logger_save_dir = "../../logs/doom_model_exp"
    model_save_dir = "../../ckpt/doom_model_exp"
    info = ""


cfg = Config

def parse(**kwargs):
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        elif k != "help":
            raise ValueError('No such keyword: {}' % k)

    cfg.timestr = time.strftime('%Y-%b-%d@%H:%M:%S')

fire.Fire(parse)

os.system('mkdir -p {}'.format(cfg.seq_save_dir))
os.system('mkdir -p {}'.format(cfg.logger_save_dir))
os.system('mkdir -p {}'.format(cfg.seq_extract_dir))
os.system('mkdir -p {}'.format(cfg.model_save_dir))

info = "\n"
for k, v in cfg.__dict__.items():
    if not k.startswith("__"):
        info += "{}:\t {}\n".format(k, v)

cfg.info = info

