from collections import namedtuple
import torch.nn as nn
from FVI.config import *


#### Copy paster from https://github.com/hanbingyan/FVIOT.

# def sinkhorn_knopp(mu, nu, C, reg, niter):
#     K = np.exp(-C/C.max()/reg)
#     u = np.ones((len(mu), ))
#     for i in range(1, niter):
#         v = nu/np.dot(K.T, u)
#         u = mu/(np.dot(K, v))
#     Pi = np.diag(u) @ K @ np.diag(v)
#     return Pi


Transition = namedtuple("Transition", ("time", "x", "y", "value"))


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def clear(self):
        self.memory.clear()
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return samples

    def __len__(self):
        return len(self.memory)


def optimize_model(policy_net, memory, optimizer, Trunc_flag):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    values_batch = torch.stack(batch.value)
    x_batch = torch.stack([torch.tensor([x], dtype=torch.float32) for x in batch.x])
    y_batch = torch.stack([torch.tensor([y], dtype=torch.float32) for y in batch.y])

    time_batch = torch.stack([torch.tensor(t, dtype=torch.float32) for t in batch.time])

    left_values = policy_net(time_batch, x_batch, y_batch)

    # # Compute the expected Q values
    Loss_fn = nn.SmoothL1Loss()
    # Loss_fn = nn.MSELoss()
    loss = Loss_fn(left_values, values_batch)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    if Trunc_flag:
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss
