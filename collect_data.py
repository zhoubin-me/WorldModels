from vizdoom import *
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import cv2

action_repeat = 12
nepis = 1000
resolution = (64, 64)
min_length = 100
max_legnth = 2000


def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

config_file_path = "./scenarios/take_cover.cfg"
game = initialize_vizdoom(config_file_path)
num_actions = game.get_available_buttons_size()
actions = [[True, False], [False, True]]

def preprocess(img):
    img = img[:, 80:-80, :]
    img = cv2.resize(img, resolution)
    return img

class Memory:
    def __init__(self, capacity):
        pass

for epis in range(nepis):
    traj = []
    game.new_episode()
    repeat = np.random.randint(1, 11)

    for step in range(max_legnth):
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

    sx, ax, rx, dx = [np.array(x, dtype=np.uint8) for x in zip(*traj)]
    random_generated_int = np.random.randint(0, 2**31-1)
    np.savez_compressed('record/' + str(random_generated_int) + '.npz', sx=sx, ax=ax, rx=rx, dx=dx)
    print("%5d / %5d Collected" % (epis, nepis))
