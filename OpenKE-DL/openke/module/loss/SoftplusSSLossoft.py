import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Loss import Loss
import math


class SoftplusSSLossoft(Loss):

    def __init__(self, u1=0.95, u2=0.005, lam=1.0):
        super(SoftplusSSLossoft, self).__init__()
        self.u1 = u1
        self.u2 = u2
        self.lam = lam
        self.act = nn.Softmax(dim=-1)
        self.criterion = nn.Softplus()
        self.relu = nn.ReLU()

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        total = torch.cat((p_score, n_score), 1)
        t_soft = self.act(total)
        p_s, n_s = torch.split(t_soft, [1, 10], dim=1)
        log_p = -torch.log(p_s)
        log_n = -torch.log(n_s)
        p_loss = self.relu(log_p+math.log(self.u1)).mean()
        n_loss = self.relu(-math.log(self.u2)-log_n).mean()
        return p_loss + self.lam * n_loss

    def predict(self, p_score, n_score):
        score = (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2
        return score.cpu().data.numpy()