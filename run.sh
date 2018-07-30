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
    # 6. Eval Controller
    rm temp/*.png
    for i in $(seq -f "%05g" 0 25 225)
    do
        mpirun -np 17 --hostfile hosts.txt python eval.py \
            --vae-save-ckpt ./ckpt/vae_2018-Jul-29@18_08_14_e012.pth \
            --rnn-save-ckpt ./ckpt/rnn_2018-Jul-29@20_37_43_e390.pth \
            --ctrl-save-ckpt ./ckpt/controller_2018-Jul-30@15_08_02_step_$i.pth
    done
elif [ $1 -eq 7 ]
then
    # 7. Plot Result
    python parse_plot.py \
        --vae ./logs/vae_train_2018-Jul-29@18_08_14.log \
        --rnn ./logs/rnn_train_2018-Jul-29@20_37_43.log \
        --es ./logs/es_train_2018-Jul-30@15_08_02.log
fi
