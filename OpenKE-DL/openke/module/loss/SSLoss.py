import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss


class SSLoss(Loss):

    def __init__(self, m=5.0, u1=3.0, u2=4.0, lam=2.0):
        super(SSLoss, self).__init__()
        self.u1 = u1
        self.u2 = u2
        self.lam = lam
        self.margin = m


    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        zeros_neg = torch.zeros_like(n_score)
        zeros_pos = torch.zeros_like(p_score)
        return torch.max(p_score - self.u1, zeros_pos).sum() + self.lam * torch.max(self.u2 - n_score, zeros_neg).sum()

    def predict(self, p_score, n_score):
        score = torch.max(p_score - n_score, -self.margin).mean() + self.margin
        return score.cpu().data.numpy()