import numpy as np
import cv2
from config import cfg
import os
# from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
# from doom_py.vizdoom import *
from vizdoom import *
from scipy.misc import imresize as resize
import pdb


class DoomTakeCover:
    def __init__(self, visible=False):
        game = DoomGame()
        game.load_config('./scenarios/take_cover.cfg')
        if visible:
            # game.set_screen_resolution(ScreenResolution.RES_640X480)
            game.set_screen_resolution(ScreenResolution.RES_160X120)
        else:
            game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.BGR24)
        game.set_window_visible(visible)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = [[True, False], [False, True], [False, False]]
        self.game = game
        self.num_actions = len(self.actions)

    def preprocess_(self, img):
        img = cv2.resize(img, (64, 64))
        return img

    def preprocess(self, obs):
        obs = np.array(obs[0:400, :, :]).astype(np.float)/255.0
        obs = np.array(resize(obs, (64, 64)))
        obs = ((1.0 - obs) * 255).round().astype(np.uint8)
        return obs

    def reset(self):
        self.game.new_episode()
        img = self.game.get_state().screen_buffer
        img = self.preprocess(img)
        return img

    def step(self, action):
        action = self.actions[action]
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()
        if not done:
            img = self.game.get_state().screen_buffer
            img = self.preprocess(img)
        else:
            img = None
        return img, reward, done, None


def collect_once(size, rank):
    env = DoomTakeCover()
    nepi = cfg.total_seq // size + 1
    for epi in range(nepi):
        obs = env.reset()
        repeat = np.random.randint(1, cfg.action_repeat)
        traj = []
        for step in range(cfg.max_seq_len):
            if step % repeat == 0:
                action = np.random.randint(0, 3)
                '''
                action = [0] * 43
                if action_ == 1:
                    action[10] = 1
                elif action_ == 2:
                    action[11] = 1
                '''
            obs_next, reward, done, _ = env.step(action)
            # obs = preprocess(obs)
            traj += [(obs, action, reward, done)]
            obs = obs_next
            if done:
                break

        if step > cfg.min_seq_len:
            sx, ax, rx, dx = [np.array(x, dtype=np.uint8) for x in zip(*traj)]
            rind = np.random.randint(0, 99999)
            save_path = "{}/{:04d}_{:05d}_{:05d}.npz".format(cfg.seq_save_dir, rank, epi, rind)
            np.savez_compressed(save_path, sx=sx, ax=ax, rx=rx, dx=dx)

        print("Worker {}: {}/{}, frames {}".format(rank, epi, nepi, len(traj)))

def collect_all():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    collect_once(size, rank)

if __name__ == '__main__':
    collect_all()
