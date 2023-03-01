"""
A fully parallel implementation of the Trust Region Policy Optimization
algorithm described in OpenAI's Spinning Up resource.
Implements reward-to-go but not generalized advantage estimation.

Parallelization is automated by the gymnasium package, maintained
by the Farama Foundation.
Each batch of batch_size rollouts is played simultaneously before the
policy is updated in a single "episode".

CartPole-v1 Results:
TRPO updates are considerably slower than VPG.
However, the performance difference is notable.
After 100 episodes, TRPO hit the CartPole-v1 ceiling
of 500 reward and showed no instability or notable regression
in policy performance.
VPG, on the other hand, hit a ceiling and needed to be cut off
by hand to prevent significant regression.
Incredibly, the TRPO agent achieved an *average reward* of 500
for 7 episodes in a row (100-106), meaning that all 32 rollouts in the batch
achieved the maximum CartPole-v1 reward for 7 batches in a row.
This degree of consistency was not seen with VPG, which makes sense
as the TRPO updates are much more carefully chosen.

Note: Policy performance still crashes quite often.
This is unlike VPG which seems to learn more consistently
in the initial stages.
"""

import torch
import numpy as np
import gymnasium as gym
import model

problem = 'CartPole-v1'
threshold = 475
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

def trpo_update(cg_iters, back_coeff, back_lim, kl_div_lim, lr_kick,
        obs_seq, act_seq, rew_seq, not_mask):
    """
    The TRPO update is not as simple as computing a loss function and calling
    .backward to populate the gradients of the policy network.
    
    The relative entropy constraint means that the update must involve the
    Hessian, specifically the combination H^{-1}g where g is the policy gradient
    and H is the Hessian of the relative entropy between the old and new policies.

    In order to handle this efficiently, we require a finer degree of control
    than is typically afforded by optimizers and .backward in PyTorch.
    As such, our TRPO update will utilize certain torch.autograd APIs which are
    not typically needed in supervised learning problems.
    We are not sure if there is a more natural approach with the same efficiency.
    """
    logits_old = logits_net(obs_seq)
    policy_old = torch.distributions.categorical.Categorical(logits=logits_old)
    prob_seq_old = policy_old.log_prob(act_seq).exp()

    # Total reward weighting:
    # weights = (rew_seq*not_mask).sum(dim=0, keepdim=True)
    # Reward-to-go weighting:
    total_weights = (rew_seq*not_mask).sum(dim=0, keepdim=True)
    weights = total_weights - (rew_seq*not_mask).cumsum(dim=0)

    # Vanilla policy gradient
    log_prob_seq = policy_old.log_prob(act_seq)
    log_probs = log_prob_seq * not_mask
    vpg_loss = (log_probs * weights).sum() / batch_size
    # Manually zero grad
    for p in logits_net.parameters():
        p.grad = None
    # Populate policy network tensors with policy gradient g
    vpg_loss.backward()
    
    # Store policy gradient g in auxiliary list
    g = [p.grad.detach() for p in logits_net.parameters()]

    # Create x data for auxiliary conjugate gradient problem x = H^{-1}g
    x = [torch.zeros_like(p) for p in logits_net.parameters()]

    # Rebuild the full graph
    logits_old = logits_net(obs_seq)
    # First derivative of relative entropy
    kl_deriv = torch.autograd.grad(
                    sample_kl_div(logits_old, logits_old, not_mask), 
                    logits_net.parameters(),
                    create_graph=True # Allow for second derivative
                )

    # Use torch.autograd.grad to double-differentiate
    def compute_Hvec(vec):
        return torch.autograd.grad(
                kl_deriv,
                logits_net.parameters(), 
                grad_outputs=vec,
                retain_graph=True
            )

    # Inner products between iterables of tensor parameters
    def inner(v1, v2):
        return sum([(v1e * v2e).sum() for v1e, v2e in zip(v1, v2)])
    # Addition and subtraction of two iterables of tensor parameters
    def add(v1, v2):
        return [(v1e + v2e) for v1e, v2e in zip(v1, v2)]
    def sub(v1, v2):
        return [(v1e - v2e) for v1e, v2e in zip(v1, v2)]
    # Scalar multiplication
    def mult(scalar, v):
        return [(scalar * ve) for ve in v]
    # Approximate equality testing
    def equal(v1, v2):
        return all([torch.allclose(v1e, v2e, atol=1e-3) for v1e, v2e in zip(v1, v2)])

    # Set up initial conjugate gradient parameters d, r
    Hx = compute_Hvec(x)
    d = sub(g, Hx)
    r = sub(g, Hx)

    # Run conjugate gradient algorithm, updating x until close to H^{-1}g
    # or out of iterations
    iter = 0
    while not equal(Hx, g) and iter < cg_iters:
        # Debugging purposes
        objective = inner(x, Hx) - 2*inner(g, x)

        beta_denom = inner(r, r)
        Hd = compute_Hvec(d)

        alpha = beta_denom / inner(d, Hd)

        x = add(x, mult(alpha, d))
        r = sub(r, mult(alpha, Hd))
        beta = inner(r, r) / beta_denom
        d = add(r, mult(beta, d))

        Hx = compute_Hvec(x)
        iter += 1
    
    # Backtracking line search for valid update
    update_coeff = (2*kl_div_lim / inner(x, compute_Hvec(x)))**0.5

    j = 0
    while j < back_lim:
        for p, xe in zip(logits_net.parameters(), x):
            p.data += (back_coeff**j) * update_coeff * xe
        
        logits_new = logits_net(obs_seq)
        policy_new = torch.distributions.categorical.Categorical(logits=logits_new)
        prob_seq_new = policy_new.log_prob(act_seq).exp()

        surr_adv = sample_surr_adv(prob_seq_new, prob_seq_old, weights, not_mask)
        kl_div = sample_kl_div(logits_new, logits_old, not_mask)

        if surr_adv > 0 and kl_div < kl_div_lim:
            # Valid update found and taken
            return total_weights.sum() / batch_size
        else:
            # Undo trial update and prepare for subsequent attempt
            for p, xe in zip(logits_net.parameters(), x):
                p.data -= (back_coeff**j) * update_coeff * xe
            j += 1

    print("No valid TRPO update found within backtracking limit.")
    # Kick with vanilla policy gradient if reward not significant
    avg_rew = total_weights.sum() / batch_size
    # if avg_rew < threshold:
    #     for p, ge in zip(logits_net.parameters(), g):
    #             p.data -= lr_kick * ge
    return avg_rew



def sample_surr_adv(prob_seq_new, prob_seq_old, weights, not_mask):
    # Detach old prob_seq to enable .backward derivative only on new,
    # which computes the vanilla policy gradient
    return ((prob_seq_new / prob_seq_old.detach()) * weights).sum() / not_mask.sum()


def sample_kl_div(logits_new, logits_old, not_mask):
    policy_new = torch.distributions.categorical.Categorical(logits=logits_new)
    policy_old = torch.distributions.categorical.Categorical(logits=logits_old.detach())
    # Detach old logits to enable .backward derivative only on new,
    # relevant for Hessian computation
    kl = torch.distributions.kl.kl_divergence(policy_new, policy_old)
    return (kl * not_mask).sum() / not_mask.sum()


episodes = 1000
cg_iters = 50
back_coeff = 0.8
back_lim = 10
kl_div_lim = 0.01
lr_kick = 3e-4

def train():
    logits_net.train()
    for ep in range(episodes):
        obs, act, rew, mask = play_episode()
        avg_rew = (rew*mask).sum() / batch_size
    
        if ep % 1 == 0:
            print(f"Episode {ep:>5d}/{episodes:>5d}")
            print(f"Average reward = {avg_rew.item():>7f}\n")

        trpo_update(cg_iters, back_coeff, back_lim, kl_div_lim, lr_kick,
                obs, act, rew, mask)

def test():
    logits_net.eval()
    obs, act, rew, mask = play_episode()
    avg_rew = (rew*mask).sum() / batch_size
    print(f"Average reward = {avg_rew.item():>7f}\n")

train()
for _ in range(10):
    test()