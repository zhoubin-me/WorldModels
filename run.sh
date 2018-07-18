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
# python main.py --task vae_extract --vae_save_ckpt ../../ckpt/doom_model_exp/vae_2018-Jul-18@02:51:16_epoch_011.pth

# 4. Train RNN
# python main.py --task rnn_train

# 5. Train Controller

#python main.py --task es_train \
#    --vae_save_ckpt ../../ckpt/doom_model_exp/vae_2018-Jul-18@02:51:16_epoch_011.pth \
#    --rnn_save_ckpt ../../ckpt/doom_model_exp/rnn_2018-Jul-18@05:14:33_epoch_299.pth

# 6. Play with the model
python main.py --task play \
    --vae_save_ckpt ../../ckpt/doom_model_exp/vae_2018-Jul-18@02:51:16_epoch_011.pth \
    --rnn_save_ckpt ../../ckpt/doom_model_exp/rnn_2018-Jul-18@05:14:33_epoch_299.pth \
    --ctrl_save_ckpt ../../ckpt/doom_model_exp/controller_2018-Jul-18@08:55:39_step_00015.pth

# 7. Plot
