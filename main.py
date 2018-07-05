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
from datetime import datetime
import numpy as np
from common import Logger

####################################################################
# Parameter Setup
####################################################################

class Config:
    # Data Collection Setup
    total_seq = 12000 # Total Number of Sequence to collect
    max_seq_len = 2000 #
    min_seq_len = 100



    # RNN Model Setting
    action_embed_size = 4
    rnn_size = 512
    rnn_batch_size = 1024
    rnn_seq_len = 500
    rnn_num_epoch = 600
    num_mixtures = 5
    rnn_lr_min = 0.00001
    rnn_lr_max = 0.001
    rnn_lr_decay = 0.9999
    rnn_r_loss_w = 9

    # VAE Setting
    vae_batch_size = 1024
    vae_num_epoch = 12
    vae_lr = 1e-4
    vae_image_size = 64
    vae_z_size = 64
    vae_kl_tolerance = 0.5

    vae_extract_ckpt = ""



    # Game Related Setting
    game_cfg_path = "./scenarios/take_cover.cfg"
    game_actions = [[True, False], [False, True]]

    action_repeat = 12          # maximum action repeat
    resolution = (64, 64)       # frame resolution to resize to

    # Others
    logsqrt2pi = np.log(np.sqrt(2.0 * np.pi))
    num_cpus = 48
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

    cfg.timestr = datetime.now().strftime('%Y%b%d_%H%M%S')

fire.Fire(parse)

info = ""
for k, v in cfg.__dict__.items():
    if not k.startswith("__"):
        info += "{}:\t {}\n".format(k, v)

cfg.info = info


####################################################################
# Here we go
####################################################################

import collect_data
import vae_train

if __name__ == '__main__':
    os.environ['LD_LIBRARY_PATH'] += ":/opt/gcc-4.9.2/lib64"
    os.system('mkdir -p %s' % cfg.seq_save_dir)
    os.system('mkdir -p %s' % cfg.seq_extract_dir)
    os.system('mkdir -p %s' % cfg.model_save_dir)
    os.system('mkdir -p %s' % cfg.logger_save_dir)

    if cfg.task == "collect":
        collect_data.collect_all()
    elif cfg.task == "vae_train":
        vae_train.vae_train()
    elif cfg.task == "vae_extract":
        vae_train.extract()
    elif cfg.task == "rnn_train":
        rnn_train.rnn_train()

