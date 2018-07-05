import torch
import torch.nn as nn
from evolution import EvolutionModule
from model import RNNModel, Controller
from joblib import Parallel, delayed
from collections import OrderedDict
from torch.distributions import Categorical
import torch.nn.functional as F
import copy
import glob
import numpy as np


def load_init_mu_var(f):
    data = np.load(f)
    return data['mu'][0], data['logvar'][0]

def sample_init_z():
    idx = np.random.randint(0, len(datas))
    mu, logvar = datas[idx]
    z = mu + np.exp(logvar / 2.0) * np.random.randn(*mu.shape)
    return z

data_list = glob.glob('converted/*.npz')
datas = Parallel(n_jobs=48, verbose=1)(delayed(load_init_mu_var)(f) for f in data_list)
model = RNNModel()
rnn_stat_dict = torch.load('./save/rnn_model_e200.pth')['model']
new_rnn_stat_dict = OrderedDict()
for k, v in rnn_stat_dict.items():
    new_rnn_stat_dict[k[7:]] = v
model.load_state_dict(new_rnn_stat_dict)
controller = Controller()
temperature = 1.25

def get_reward(controller):
    model_x = copy.deepcopy(model)
    frames = []
    z = sample_init_z()
    done = 0
    model_x.reset()

    for step in range(1000):
        z = torch.from_numpy(z).float().unsqueeze(0)
        x = torch.cat((model_x.hx.detach(), model_x.cx.detach(), z), dim=1)
        y = controller(x)
        m = Categorical(F.softmax(y, dim=1))
        action = m.sample()

        logmix, mu, logstd, done_p = model_x.step(z.unsqueeze(0), action.unsqueeze(0))

        logmix = logmix / temperature
        logmix -= logmix.max()
        logmix = torch.exp(logmix)

        m = Categorical(logmix)
        idx = m.sample()

        new_mu = torch.FloatTensor([mu[i, j] for i, j in enumerate(idx)])
        new_logstd = torch.FloatTensor([logstd[i, j] for i, j in enumerate(idx)])
        z_next = new_mu + new_logstd.exp() * torch.randn_like(new_mu) * np.sqrt(temperature)

        z = z_next.detach().numpy()
        if done_p.squeeze().item() > 0:
            return step

    return step

es = EvolutionModule(controller, get_reward, population_size=100,
        sigma=0.1, learning_rate=0.001, reward_goal=999, consecutive_goal_stopping=20,
        threadcount=50, cuda=False, render_test=False)

controller = es.run(4000, print_step=1)

reward = partial_func()
print('Final Reward', reward)

