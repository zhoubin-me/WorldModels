#! /bin/sh
#
# run.sh
# Copyright (C) 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.
#



if [ $1 -eq 1 ]
then
    # 1. Collect data
    mpirun --hostfile hosts.txt python collect_data.py --total-seq 12000

elif [ $1 -eq 2 ]
then
    # 2. Train VAE
    export CUDA_VISIBLE_DEVICES=0,1
    python vae_train.py --vae-batch-size 128

elif [ $1 -eq 3 ]
then
    # 3. Extract Dataset
    python vae_extract.py \
        --vae-save-ckpt ./ckpt/vae_2018-Jul-29@18:08:14_e012.pth

elif [ $1 -eq 4 ]
then
    # 4. Train MDN-RNN
    export CUDA_VISIBLE_DEVICES=0,1
    python rnn_train.py --rnn-batch-size 128

elif [ $1 -eq 5 ]
then
    # 5. Train Controller
    mpirun --hostfile hosts.txt python es_train.py \
        --vae-save-ckpt ./ckpt/vae_2018-Jul-29@18:08:14_e012.pth \
        --rnn-save-ckpt ./ckpt/rnn_2018-Jul-29@20:37:43_e390.pth
elif [ $1 -eq 6 ]
then
    # 5. Play Model
    export CUDA_VISIBLE_DEVICES=0,1
    python play.py \
        --vae-save-ckpt ./ckpt/ \
        --rnn-save-ckpt ./ckpt/
fi

