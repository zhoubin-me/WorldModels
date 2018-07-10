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



    # RNN Model Setting
    action_embed_size = 4
    rnn_size = 512
    rnn_batch_size = 256
    rnn_seq_len = 500
    rnn_num_epoch = 600
    num_mixtures = 5
    rnn_lr_min = 0.00001
    rnn_lr_max = 0.001
    rnn_lr_decay = 0.99995
    rnn_r_loss_w = 9

    rnn_save_ckpt = ""

    # VAE Setting
    vae_batch_size = 1024
    vae_num_epoch = 12
    vae_lr = 1e-4
    vae_image_size = 64
    vae_z_size = 64
    vae_kl_tolerance = 0.5

    vae_save_ckpt = ""

    # Controller ES Setting
    max_steps = 5000
    es_steps = 4000
    es_lr = 0.001
    es_lr_decay = 0.999
    es_sigma = 0.1
    es_sigma_decay = 0.999

    population_size = 100
    temperature = 1.15
    trials_per_pop = 10


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

    cfg.timestr = time.strftime('%Y-%b-%d@%H:%M:%S')

fire.Fire(parse)

info = "\n"
for k, v in cfg.__dict__.items():
    if not k.startswith("__"):
        info += "{}:\t {}\n".format(k, v)

cfg.info = info


####################################################################
# Here we go
####################################################################

#import dream_and_play

if __name__ == '__main__':
    import collect_data
    import vae_train
    import rnn_train
    import es_train
    import dream_and_play

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
        vae_train.vae_extract()
    elif cfg.task == "rnn_train":
        rnn_train.rnn_train()
    elif cfg.task == "es_train":
        es_train.es_train()
    elif cfg.task == "dream":
        dream_and_play.dream()
        dream_and_play.real()


