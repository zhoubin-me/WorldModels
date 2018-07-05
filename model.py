import numpy as np
import torch
import torch.nn as nn


z_size = 64
rnn_size = 512

class VAE(nn.Module):
    def __init__(self, z_size=64):
        super(VAE, self).__init__()
        self.z_size = z_size

        # (N, 3, 64, 64) -> (N, 256, 2, 2)
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 4, 2),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 4, 2),
                nn.ReLU(True))

        # (N, 256 * 4, 1, 1) -> (N, 3, 64, 64)
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256 * 4, 128, 5, 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 5, 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 6, 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, 6, 2),
                nn.Sigmoid())

        self.h = nn.Linear(self.z_size, 256 * 4)
        self.mu = nn.Linear(256 * 4, self.z_size)
        self.logvar = nn.Linear(256 * 4, self.z_size)


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.mu(x), self.logvar(x)
        z = mu + torch.exp(logvar / 2.0) * torch.randn_like(mu)
        x = self.decode(z)
        return mu, logvar, x, z

    def decode(self, z):
        x = self.h(z)
        x = x.view(z.size(0), 256 * 4, 1, 1)
        x = self.decoder(x)
        return x


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.action_embed = nn.Embedding(2, 4)
        # action_embeded + z_size
        self.rnn = nn.LSTMCell(64 + 4, rnn_size)
        # z_size * (mix, mu, logvar) * num_mixtures +  done
        self.output = nn.Linear(rnn_size, 64 * 3 * 5 + 1)
        self.const = np.log(np.sqrt(2.0 * np.pi))

    def reset(self):
        self.hx, self.cx = torch.zeros(1, rnn_size), torch.zeros(1, rnn_size)

    def forward(self, z, actions, dones):
        actions = self.action_embed(actions.long())
        dones = dones.float()
        inp = torch.cat((z, actions), dim=2)
        outputs = []
        hx, cx = torch.zeros(z.size(0), rnn_size).cuda(), torch.zeros(z.size(0), rnn_size).cuda()
        seq_length = inp.size(1)
        for step in range(0, seq_length):
            hx, cx = self.rnn(inp[:, step, :], (hx, cx))
            outputs.append(hx.unsqueeze(1))
            hx = hx * (1.0 - dones[:, step].unsqueeze(1))
            cx = cx * (1.0 - dones[:, step].unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        # (batch_size, seq_length, 961)
        output = self.output(output[:, :-1, :])

        logmix, mu, logstd, done_p = torch.split(output, 320, 2)

        logmix = logmix.contiguous().view(-1, 5)
        mu = mu.contiguous().view(-1, 5)
        logstd = logstd.contiguous().view(-1, 5)
        logmix = logmix - logmix.exp().sum(dim=1, keepdim=True).log()

        return logmix, mu, logstd, done_p.squeeze()

    def step(self, z, action):
        action = self.action_embed(action.long())
        inp = torch.cat((z, action), dim=2)
        self.hx, self.cx = self.rnn(inp[:, 0, :], (self.hx, self.cx))
        output = self.output(self.hx)
        logmix, mu, logstd, done_p = torch.split(output, 320, 1)

        logmix = logmix.contiguous().view(-1, 5)
        mu = mu.contiguous().view(-1, 5)
        logstd = logstd.contiguous().view(-1, 5)
        logmix = logmix - logmix.exp().sum(dim=1, keepdim=True).log()

        return logmix, mu, logstd, done_p.squeeze()


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(512 + 512 + 64, 128),
                nn.ReLU(True),
                nn.Linear(128, 2),)

    def forward(self, x):
        return self.fc(x)

    def init_weight(self, params):
        for param1, param2 in zip(self.parameters(), params):
            param1.data.copy_(param2.data)


