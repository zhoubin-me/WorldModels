#! /bin/sh
#
# run.sh
# Copyright (C) 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.
#

# 1. Collect data
# python main.py --task collect --total_seq 12000

# 2. Train VAE, est 2h
# python main.py --task vae_train

# 3. Extract Frames with VAE
# python main.py --task vae_extract --vae_save_ckpt ../../ckpt/doom_model_exp/vae_2018Jul06_102503_epoch_011.pth

# 4. Train RNN
python main.py --task rnn_train

# 5. Train Controller
# python main.py --task es_train --rnn_save_ckpt ../../ckpt/doom_model_exp/rnn_2018-Jul-07@03:17:51_epoch_499.pth

