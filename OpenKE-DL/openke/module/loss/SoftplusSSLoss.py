import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Loss import Loss
import math


class SoftplusSSLoss(Loss):

    def __init__(self, u1=0.9, u2=0.3, lam=1.0):
        super(SoftplusSSLoss, self).__init__()
        self.u1 = u1
        self.u2 = u2
        self.lam = lam
        self.act = torch.nn.Sigmoid()
        self.criterion = nn.Softplus()
        self.relu = nn.ReLU()

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        log_p = -torch.log(self.act(p_score))
        log_n = -torch.log(self.act(n_score))
        p_loss = self.relu(log_p+math.log(self.u1)).mean()
        n_loss = self.relu(-math.log(self.u2)-log_n).mean()
        return p_loss + self.lam * n_loss

    def predict(self, p_score, n_score):
        score = (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2
        return score.cpu().data.numpy()