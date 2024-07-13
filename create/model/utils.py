import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Regularization(nn.Module):
    def __init__(self, model, weight_decay1=0.0, weight_decay2=0.0):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay1 = weight_decay1
        self.weight_decay2 = weight_decay2
        self.weight_list = self.get_weight(model)

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay1, self.weight_decay2)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay1, weight_decay2):
        reg_loss1 = 0
        reg_loss2 = 0
        for name, w in weight_list:
            l1_reg = torch.norm(w, p=1)
            l2_reg = torch.norm(w, p=2)
            reg_loss1 = reg_loss1 + l1_reg
            reg_loss2 = reg_loss2 + l2_reg

        reg_loss = weight_decay1 * reg_loss1 + weight_decay2 * reg_loss2
        return reg_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=[1]*2, gamma=[2]*2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCELoss(reduction='none')

    def forward(self, logits, label):
        with torch.no_grad():
            alphas = torch.empty_like(label).fill_(self.alpha[0])
            gammas = torch.empty_like(label).fill_(self.gamma[0])
            for i in range(1, len(self.alpha)):
                alphas[label == i] = self.alpha[i]
                gammas[label == i] = self.gamma[i]
                
        ce_loss = self.crit(logits, label)
        pt = torch.exp(-ce_loss)
        loss = (alphas * torch.pow(1 - pt, gammas) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_mean_score(label, out, aug=2, cls=5):
    score = np.zeros((len(label), cls))
    for j in range(len(score)):
        for l in range(aug):
            score[j] += out[aug * j + l]
    score = (score / aug)
    return score
