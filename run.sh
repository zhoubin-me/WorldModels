#! /bin/sh
#
# run.sh
# Copyright (C) 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.
#

task=$1

if [ $task -eq 1 ]
then
    # 1. Collect data
    mpirun --hostfile hosts.txt python collect_data.py --total-seq 12000

elif [ $task -eq 2 ]
then
    # 2. Train VAE
    export CUDA_VISIBLE_DEVICES=0,1
    python vae_train.py --vae-batch-size 128

elif [ $task -eq 3 ]
then
    # 3. Extract Dataset vae_train_2018_Jul_31-23_00_45
    python vae_extract.py \
        --vae-save-ckpt ./ckpt/vae_2018_Aug_04-12_39_03_e011.pth

elif [ $task -eq 4 ]
then
    # 4. Train MDN-RNN
    export CUDA_VISIBLE_DEVICES=0,1
    python rnn_train.py --rnn-batch-size 128

elif [ $task -eq 5 ]
then
    # 5. Train Controller
    mpirun --hostfile hosts.txt python es_train.py \
        --vae-save-ckpt ./ckpt/vae_2018_Aug_04-12_39_03_e011.pth \
        --rnn-save-ckpt ./ckpt/rnn_2018_Aug_04-17_03_31_e390.pth
elif [ $task -eq 6 ]
then
    # 6. Eval Controller
    # rm temp/*.png
    for i in $(seq -f "%05g" 300 25 300)
    do
        mpirun -np 17 --hostfile hosts.txt python eval.py \
            --vae-save-ckpt ./ckpt/vae_2018_Aug_04-12_39_03_e011.pth \
            --rnn-save-ckpt ./ckpt/rnn_2018_Aug_04-17_03_31_e390.pth \
            --ctrl-save-ckpt ./ckpt/controller_curr_2018_Aug_05-14_30_57_step_$i.pth
    done
elif [ $task -eq 7 ]
then
    # 7. Plot Result
    python parse_plot.py \
        --vae ./logs/vae_train_2018-Jul-29@18:08:14.log \
        --rnn ./logs/rnn_train_2018-Jul-29@20:37:43.log \
        --es ./logs/es_train_2018-Jul-30@15:08:02.log
elif [ $task -eq 8 ]
then
    python play.py \
        --vae-save-ckpt ./ckpt/vae_2018_Aug_04-12_39_03_e011.pth \
        --rnn-save-ckpt ./ckpt/rnn_2018_Aug_04-17_03_31_e390.pth \
        --ctrl-save-ckpt ./ckpt/controller_curr_2018_Aug_05-14_30_57_step_00300.pth
fi

echo "Finished"
