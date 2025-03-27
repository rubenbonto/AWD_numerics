import torch.nn as nn
import torch.nn.functional as F
from FVI.config import *


#### Copy paster from https://github.com/hanbingyan/FVIOT.

h = 8


class DQN(nn.Module):
    def __init__(self, x_dim, y_dim, T):
        super(DQN, self).__init__()
        self.T = T
        self.linear1 = nn.Linear(x_dim + y_dim, h)
        # self.linear1.weight.data.fill_(10.0)
        # torch.nn.init.xavier_uniform_(self.linear1.weight)
        # torch.nn.init.zeros_(self.linear1.weight)
        # torch.nn.init.zeros_(self.linear1.bias)
        # self.bn = nn.BatchNorm1d(h)
        self.linear2 = nn.Linear(h, h)
        # torch.nn.init.xavier_uniform_(self.linear2.weight)
        # torch.nn.init.zeros_(self.linear2.bias)
        # torch.nn.init.zeros_(self.linear2.weight)

        # self.dropout = nn.Dropout(p=0.5)

        self.linear3 = nn.Linear(h, 1)

        self.linear5 = nn.Linear(2, 1)
        # torch.nn.init.zeros_(self.linear5.bias)
        # torch.nn.init.zeros_(self.linear5.weight)
        # torch.nn.init.xavier_uniform_(self.linear5.weight)
        self.linear6 = nn.Linear(2, 1)
        # torch.nn.init.zeros_(self.linear6.bias)

    def forward(self, time, x, y):
        state = torch.cat((x, y), dim=1)
        state = torch.relu(self.linear1(state))
        # state = self.bn(state)
        state = torch.relu(self.linear2(state))
        # state = self.dropout(state)
        state = torch.sigmoid(self.linear3(state))
        time_f2 = torch.cat((self.T - time, (self.T - time) ** 2), dim=1)
        time_f1 = self.linear5(time_f2)
        time_f2 = self.linear6(time_f2)
        return state * time_f1 + time_f2
