import numpy as np

from concurrent.futures import ThreadPoolExecutor as TPE
from vizdoom import *

import os
import cv2

from main import cfg
from common import Logger

def preprocess(img):
    img = img[:, 80:-80, :]
    img = cv2.resize(img, cfg.resolution)
    return img


def initialize_vizdoom():
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(cfg.game_cfg_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

def collect_once(index):
    actions = cfg.game_actions
    nepi = cfg.total_seq // cfg.num_cpus + 1

    for epi in range(nepi):
        game = initialize_vizdoom()
        game.new_episode()
        repeat = np.random.randint(1, 11)
        traj = []

        for step in range(cfg.max_seq_len):
            s1 = game.get_state().screen_buffer
            s1 = preprocess(s1)
            if step % repeat == 0:
                a = np.random.randint(0, len(actions))
                action = actions[a]
                repeat = np.random.randint(1, 11)
            reward = game.make_action(action)
            done = game.is_episode_finished()
            traj += [(s1, a, reward, done)]
            if done:
                break

        if step > cfg.min_seq_len:
            sx, ax, rx, dx = [np.array(x, dtype=np.uint8) for x in zip(*traj)]
            save_path = "{}/{:04d}_{:05d}.npz".format(cfg.seq_save_dir, index, epi)
            np.savez_compressed(save_path, sx=sx, ax=ax, rx=rx, dx=dx)

        print("Worker {}: {}/{}".format(index, epi, nepi))

def collect_all():
    logger = Logger("{}/data_collection_{}.log".format(cfg.logger_save_dir, cfg.timestr))
    logger.log(cfg.info)
    with TPE(max_workers=cfg.num_cpus) as e:
        e.map(collect_once, [i for i in range(cfg.num_cpus)])

