"""
A system implementing several methods to reduce variance in policy
gradient learning.

Allows a value function baseline or Generalized Advantage Estimation.
"""

import torch

class ValueFunction():
    def __init__(self, model, batch_size, max_iters = 10) -> None:
        self.model = model
        self.batch_size = batch_size
        self.max_iters = max_iters

    def update_value_fn(self, obs, reward_to_go, mask):
        optimizer = torch.optim.Adam(self.model.parameters())
        for _ in range(self.max_iters):
            optimizer.zero_grad()
            loss = (mask * (self.model(obs).squeeze(dim=-1) 
                        - reward_to_go)**2).sum() / self.batch_size
            loss.backward()
            optimizer.step()

    def __call__(self, obs):
        return self.model(obs).squeeze(dim=-1).detach()