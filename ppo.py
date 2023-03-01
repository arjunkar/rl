"""
A fully parallel implementation of the Proximal Policy Optimization
algorithm described in OpenAI's Spinning Up resource.
Implements reward-to-go but not generalized advantage estimation.
The specific PPO variant we choose is PPO-Clip, without the KL
divergence term in the objective.

Parallelization is automated by the gymnasium package, maintained
by the Farama Foundation.
Each batch of batch_size rollouts is played simultaneously before the
policy is updated in a single "episode".

CartPole-v1 Results:
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
    
    # Return dimensions: [seq_len, batch_dim, type_dim]
    # type_dim e.g. obs_dim or act_dim if multi-dimensional action space
    return obs_seq[:-1], act_seq[1:], rew_seq[1:], torch.logical_not(mask_seq[:-1])

def ppo_update(obs, act, rew, mask):
    # Finish
    return

episodes = 1000

def train():
    logits_net.train()
    for ep in range(episodes):
        obs, act, rew, mask = play_episode()
        avg_rew = (rew*mask).sum() / batch_size
    
        if ep % 1 == 0:
            print(f"Episode {ep:>5d}/{episodes:>5d}")
            print(f"Average reward = {avg_rew.item():>7f}\n")


def test():
    logits_net.eval()
    obs, act, rew, mask = play_episode()
    avg_rew = (rew*mask).sum() / batch_size
    print(f"Average reward = {avg_rew.item():>7f}\n")

train()
for _ in range(10):
    test()