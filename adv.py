"""
A system implementing several methods to reduce variance or bias
in policy gradient learning.
The bias-variance tradeoff is captured by the gamma and lambda
parameters in Generalized Advantage Estimation.

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

    def value_fn(self, obs):
        return self.model(obs).squeeze(dim=-1).detach()

class GeneralizedAdvantageEstimator(ValueFunction):
    def __init__(self, model, batch_size, max_iters=10, gamma=0.99, lambda_=0.97) -> None:
        super().__init__(model, batch_size, max_iters)
        self.g = gamma
        self.l = lambda_

    def td_residual(self, obs_seq, rew_seq, mask_seq):
        vals = self.value_fn(obs_seq) * mask_seq
        # Expects [seq_len, batch_dim]
        shifted_vals = torch.cat((vals[1:], torch.zeros_like(vals[0]).unsqueeze(dim=0)))
        return (rew_seq + self.g * shifted_vals - vals) * mask_seq

    def advantage(self, obs_seq, rew_seq, mask_seq):
        res = self.td_residual(obs_seq, rew_seq, mask_seq)
        prod = self.g * self.l
        # Reverse sweep along sequence to compute advantage sums
        for seq_elem in range(obs_seq.size()[0]-2, -1, -1):
            res[seq_elem] += prod * res[seq_elem+1]
        return res