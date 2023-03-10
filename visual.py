"""
A visualization of the trained network versus an untrained network.
"""

import torch
import model
import gymnasium as gym
import os

problem = 'CartPole-v1'
obs_dim = 4
act_dim = 2
hidden_dims = [128, 128]
logits_net = model.MLP([obs_dim] + hidden_dims + [act_dim])

logits_net.load_state_dict(
    torch.load(
        os.path.abspath(os.getcwd()) + "/rl/cart_model.pth"
    )
)

untrained_net = model.MLP([obs_dim] + hidden_dims + [act_dim])

logits_net.eval()

env = gym.make(problem, render_mode = "human")

visual_eps = 30
max_len = 500

for ep in range(visual_eps):
    obs, info = env.reset()

    for _ in range(max_len):
        if ep % 2 == 1:
            logits = logits_net(torch.from_numpy(obs))
        else:
            logits = untrained_net(torch.from_numpy(obs))
        policy = torch.distributions.categorical.Categorical(logits=logits)
        act = policy.sample()
        
        obs, rew, term, trunc, info = env.step(act.numpy())

        if term or trunc:
            print("Episode terminated.")
            break

env.close()