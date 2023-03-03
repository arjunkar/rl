"""
A fully parallel implementation of the Proximal Policy Optimization
algorithm described in OpenAI's Spinning Up resource.
Implements reward-to-go with value function baseline 
but not generalized advantage estimation.
The specific PPO variant we choose is PPO-Clip, without the KL
divergence term in the objective.

Parallelization is automated by the gymnasium package, maintained
by the Farama Foundation.
Each batch of batch_size rollouts is played simultaneously before the
policy is updated in a single "episode".

CartPole-v1 Results:
PPO displays the most impressive learning behavior of the three
policy optimization algorithms (VPG and TRPO being the other two).
It is consistent, learns extremely quickly, and is relatively stable.
In about 40 episodes, it reaches the maximum 500 average reward, which
is considerably faster than the other algorithms can achieve stably.
It does suffer from large fluctuations (hundreds of reward) even after
seemingly stabilizing, but unlike VPG and TRPO it seems to quickly
recover from these fluctuations and reach peak performance again.
In a handful of trials, it reached several 500 reward episodes in a
row by episode 50, and the fluctuations around the maximum were
only 10-20 reward (excepting the large fluctuations mentioned previously).

Adding a value function baseline increases training efficiency further,
and can reach consistent max reward in about half the episodes of a pure
reward-to-go scheme.

Generalized advantage estimation does not lead to a notable improvement
over a value function baseline, perhaps because the problem is too simple
or the variance effects are too large compared to issues arising when
bias is introduced.
"""


import torch
import numpy as np
import gymnasium as gym
import model
import adv

problem = 'CartPole-v1'
obs_dim = 4
act_dim = 2
batch_size = 32
gamma = 1.
lambda_ = 1.

env = gym.vector.make(problem, num_envs=batch_size)

hidden_dims = [128, 128]
logits_net = model.MLP([obs_dim] + hidden_dims + [act_dim])
value_net = model.MLP([obs_dim] + hidden_dims + [1])
gae_module = adv.GeneralizedAdvantageEstimator(value_net, batch_size, gamma=gamma, lambda_=lambda_)

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


clip_param = 0.1
kl_div_lim = 0.02
episodes = 100
optimizer = torch.optim.Adam(logits_net.parameters())


def sample_surr_adv(prob_seq_new, prob_seq_old, weights, not_mask):
    # Detach old prob_seq to enable .backward derivative only on new,
    # which computes the vanilla policy gradient
    return ((prob_seq_new / prob_seq_old.detach()) * weights).sum() / not_mask.sum()


def sample_kl_div(logits_new, logits_old, not_mask):
    policy_new = torch.distributions.categorical.Categorical(logits=logits_new)
    policy_old = torch.distributions.categorical.Categorical(logits=logits_old)
    kl = torch.distributions.kl.kl_divergence(policy_new, policy_old)
    return (kl * not_mask).sum() / not_mask.sum()

def ppo_g_clip(weights):
    return (1+clip_param)*(weights.relu()) - (1-clip_param)*((-weights).relu())

def ppo_loss(surr_adv, weights):
    return torch.minimum(surr_adv, ppo_g_clip(weights)).sum() / batch_size

def ppo_update(obs_seq, act_seq, rew_seq, not_mask):
    logits_old = logits_net(obs_seq)
    policy_old = torch.distributions.categorical.Categorical(logits=logits_old)
    prob_seq_old = policy_old.log_prob(act_seq).exp()

    # Reward-to-go weighting with generalized advantage estimation
    # total_weights = (rew_seq*not_mask).sum(dim=0, keepdim=True)
    # reward_to_go = total_weights - (rew_seq*not_mask).cumsum(dim=0)
    reward_to_go = rew_seq * not_mask
    for elem in range(reward_to_go.size()[0]-2, -1, -1):
        reward_to_go[elem] += gamma * reward_to_go[elem+1]
    # weights = (reward_to_go - gae_module.value_fn(obs_seq)) * not_mask
    weights = gae_module.advantage(obs_seq, rew_seq, not_mask)
    gae_module.update_value_fn(obs_seq, reward_to_go, not_mask)

    logits_new = logits_net(obs_seq)
    policy_new = torch.distributions.categorical.Categorical(logits=logits_new)
    prob_seq_new = policy_new.log_prob(act_seq).exp()
    
    # Gradient steps with early stopping by KL divergence
    while sample_kl_div(logits_new, logits_old, not_mask) < kl_div_lim: 
        surr_adv = sample_surr_adv(prob_seq_new, prob_seq_old, weights, not_mask)
        loss = -ppo_loss(surr_adv, weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits_new = logits_net(obs_seq)
        policy_new = torch.distributions.categorical.Categorical(logits=logits_new)
        prob_seq_new = policy_new.log_prob(act_seq).exp()

    


def train():
    logits_net.train()
    for ep in range(episodes):
        obs, act, rew, mask = play_episode()
        avg_rew = (rew*mask).sum() / batch_size
    
        if ep % 1 == 0:
            print(f"Episode {ep:>5d}/{episodes:>5d}")
            print(f"Average reward = {avg_rew.item():>7f}\n")

        ppo_update(obs, act, rew, mask)


def test():
    logits_net.eval()
    obs, act, rew, mask = play_episode()
    avg_rew = (rew*mask).sum() / batch_size
    print(f"Average reward = {avg_rew.item():>7f}\n")

train()
for _ in range(10):
    test()