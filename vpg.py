"""
A fully parallel implementation of the Vanilla Policy Gradient
algorithm described in OpenAI's Spinning Up resource.
Implements reward-to-go but not generalized advantage estimation.

Parallelization is automated by the gymnasium package, maintained
by the Farama Foundation.
Each batch of batch_size rollouts is played simultaneously before the
policy is updated in a single "episode".

CartPole-v1 Results:
This code, with weights given by simply the total rollout reward,
can produce a policy within 200-250 training episodes (policy updates)
which achieves over 10 test rollouts an average reward 
of 495.4 with variance (torch.var) 3.8.
This qualifies as a solution to CartPole-v1 as defined by the
Farama Foundation, which asks for an average reward of just 475
out of the truncation limit of 500.

With reward-to-go, the required episodes drop to less than 50-100,
although the variance of the resulting model is larger.
The reward-to-go implementation can achieve a mean reward of 490.8
with variance 22.4.
In 50 episodes, the average reward is already over 400.
This is an improvement over the total reward weight, which sits at
just 140 average reward after 50 episodes.
"""

import torch
import numpy as np
import gymnasium as gym
import model

problem = 'CartPole-v1'
obs_dim = 4
act_dim = 2
batch_size = 32

env = gym.vector.make(problem, num_envs=batch_size)

hidden_dims = [128, 128]
logits_net = model.MLP([obs_dim] + hidden_dims + [act_dim])

def play_episode():
    obs, info = env.reset()
    obs = torch.from_numpy(obs)
    obs_seq = obs.unsqueeze(dim=0)
    mask_seq = torch.zeros(batch_size).unsqueeze(dim=0)
    act_seq = torch.zeros_like(mask_seq)
    rew_seq = torch.zeros_like(mask_seq)

    while not mask_seq[-1].all():
        logits = logits_net(obs)
        policy = torch.distributions.categorical.Categorical(logits=logits)
        actions = policy.sample()
        act_seq = torch.cat((act_seq, actions.unsqueeze(dim=0)), dim=0)

        obs, rew, term, trunc, info = env.step(actions.numpy())
        
        obs = torch.from_numpy(obs)
        obs_seq = torch.cat((obs_seq, obs.unsqueeze(dim=0)), dim=0)
        rew_seq = torch.cat((rew_seq, torch.from_numpy(rew).unsqueeze(dim=0)), dim=0)
        newest_mask = torch.logical_or(mask_seq[-1],
            torch.from_numpy(np.logical_or(term, trunc))).unsqueeze(dim=0)
        mask_seq = torch.cat((mask_seq, newest_mask), dim=0)
    
    # Return dimensions of obs_seq: [seq_len, batch_dim, obs_dim]
    # All others: [seq_len, batch_dim]
    return obs_seq[:-1], act_seq[1:], rew_seq[1:], mask_seq[:-1]

def vpg_loss(obs_seq, act_seq, rew_seq, mask_seq):
    not_mask = torch.logical_not(mask_seq)
    logits_seq = logits_net(obs_seq)
    policy = torch.distributions.categorical.Categorical(logits=logits_seq)
    log_prob_seq = policy.log_prob(act_seq)

    # Total reward weighting:
    # weights = (rew_seq*not_mask).sum(dim=0, keepdim=True)
    # Reward-to-go weighting:
    total_weights = (rew_seq*not_mask).sum(dim=0, keepdim=True)
    weights = total_weights - (rew_seq*not_mask).cumsum(dim=0)

    log_probs = log_prob_seq * not_mask

    return -(log_probs * weights).sum() / batch_size, total_weights.sum() / batch_size
    # Depending on the weights, make sure to report _ , average reward here


episodes = 1000
optimizer = torch.optim.Adam(logits_net.parameters())

def train():
    logits_net.train()
    for ep in range(episodes):
        obs, act, rew, mask = play_episode()
        loss, avg_rew = vpg_loss(obs, act, rew, mask)
        if avg_rew.item() > 490:
            break 
        # Overtraining an already-strong policy can lead to instability
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            print(f"Episode {ep:>5d}/{episodes:>5d}")
            print(f"Loss = {loss:>7f}")
            print(f"Average reward = {avg_rew.item():>7f}\n")
    
def test():
    logits_net.eval()
    obs, act, rew, mask = play_episode()
    loss, avg_rew = vpg_loss(obs, act, rew, mask)
    # print(f"Test loss = {loss:>7f}")
    # print(f"Average test reward = {avg_rew.item():>7f}\n")
    return avg_rew

train()
avg_rew_list = []
for _ in range(10):
    avg_rew_list.append(test())

avgs = torch.tensor(avg_rew_list)
print(avgs)
print(avgs.mean())
print(avgs.var())