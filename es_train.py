"""
Evolutionary Strategies module for PyTorch models -- modified from https://github.com/alirezamika/evostra
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import pickle
import time

from joblib import Parallel, delayed
from collections import OrderedDict
from torch.distributions import Categorical

from model import RNNModel, Controller
from main import cfg



class EvolutionModule:
    def __init__(
        self,
        controller,
        reward_func,
        population_size=50,
        sigma=0.1,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        threadcount=4,
        render_test=False,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None,
        datas = None
    ):
        np.random.seed(int(time.time()))
        self.weights = list(controller.parameters())
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.cuda = cuda
        self.decay = decay
        self.sigma_decay = sigma_decay
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path
        self.datas = datas
        self.model = model


    def get_reward(self):
        model_x = copy.deepcopy(self.model)
        frames = []
        z = sample_init_z(self.datas)
        done = 0
        model_x.reset()

        for step in range(cfg.max_steps):
            z = torch.from_numpy(z).float().unsqueeze(0)
            x = torch.cat((model_x.hx.detach(), model_x.cx.detach(), z), dim=1)
            y = controller(x)
            m = Categorical(F.softmax(y, dim=1))
            action = m.sample()

            logmix, mu, logstd, done_p = model_x.step(z.unsqueeze(0), action.unsqueeze(0))

            logmix = logmix / cfg.temperature
            logmix -= logmix.max()
            logmix = torch.exp(logmix)

            m = Categorical(logmix)
            idx = m.sample()

            new_mu = torch.FloatTensor([mu[i, j] for i, j in enumerate(idx)])
            new_logstd = torch.FloatTensor([logstd[i, j] for i, j in enumerate(idx)])
            z_next = new_mu + new_logstd.exp() * torch.randn_like(new_mu) * np.sqrt(cfg.temperature)

            z = z_next.detach().numpy()
            if done_p.squeeze().item() > 0:
                return step
        return step

    def jitter_weights(self, weights, population=[], no_jitter=False):
        new_weights = []
        for i, param in enumerate(weights):
            if no_jitter:
                new_weights.append(param.data)
            else:
                jittered = torch.from_numpy(self.SIGMA * population[i]).float()
                new_weights.append(param.data + jittered)
        controller = Controller()
        controller.init_weight(new_weights)
        return controller


    def run(self, iterations, print_step=10):
        for iteration in range(iterations):

            population = []
            for _ in range(self.POPULATION_SIZE):
                x = []
                for param in self.weights:
                    x.append(np.random.randn(*param.data.size()))

                population.append(x)


            def get_reward(pop):
                controller = self.jitter_weights(copy.deepcopy(self.weights), pop)
                return self.reward_function(controller)

            rewards = Parallel(n_jobs=40, verbose=0)(delayed(get_reward)(pop) \
                    for pop in population)

            if np.std(rewards) != 0:
                normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                for index, param in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    rewards_pop = torch.from_numpy(np.dot(A.T, normalized_rewards).T).float()
                    param.data += self.LEARNING_RATE / (self.POPULATION_SIZE * self.SIGMA) \
                            * rewards_pop

                    self.LEARNING_RATE *= self.decay
                    self.SIGMA *= self.sigma_decay

            if (iteration+1) % print_step == 0:
                test_reward = self.reward_function(
                    self.jitter_weights(copy.deepcopy(self.weights), no_jitter=True))
                print('iter %d. reward: %f' % (iteration+1, test_reward))

                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))

                if self.reward_goal and self.consecutive_goal_stopping:
                    if test_reward >= self.reward_goal:
                        self.consecutive_goal_count += 1
                    else:
                        self.consecutive_goal_count = 0

                    if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                        return self.weights

        return self.weights


def load_init(f):
    data = np.load(f)
    return data['mu'][0], data['logvar'][0]

def sample_init_z():
    idx = np.random.randint(0, len(datas))
    mu, logvar = datas[idx]
    z = mu + np.exp(logvar / 2.0) * np.random.randn(*mu.shape)
    return z

def es_train():

    data_list = glob.glob(cfg.seq_extract_dir + '/*.npz')
    datas = Parallel(n_jobs=cfg.num_cpus, verbose=1)(delayed(load_init)(f) for f in data_list)

    model = RNNModel()
    rnn_stat_dict = torch.load(cfg.rnn_save_ckpt)['model']
    new_rnn_stat_dict = OrderedDict()
    for k, v in rnn_stat_dict.items():
        new_rnn_stat_dict[k[7:]] = v
    model.load_state_dict(new_rnn_stat_dict)

    controller = Controller()

    es = EvolutionModule(controller, population_size=100, \
            sigma=0.1, learning_rate=0.001, reward_goal=2000, consecutive_goal_stopping=20, \
            threadcount=50, cuda=False, render_test=False, datas=datas)

    '''
    es = EvolutionModule(controller, population_size=100, \
            sigma=0.1, learning_rate=0.001, reward_goal=2000, consecutive_goal_stopping=20, \
            threadcount=50, cuda=False, render_test=False, datas=datas)
    '''

    controller = es.run(cfg.ES_steps, print_step=1)
    save_path = "{}/controller_{}_final.pth".format(cfg.model_save_dir, cfg.timestr)
    torch.save({"model": model.stat_dict()}, save_path)

    reward = get_reward(controller)
    print('Final Reward', reward)

