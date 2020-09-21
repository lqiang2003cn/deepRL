#!/usr/bin/env python3
import gym
import copy
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

NOISE_STD = 0.01
POPULATION_SIZE = 1
PARENTS_COUNT = 1


class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(env, net,obs_size):
    obs = env.reset()
    obs = obs.reshape(-1)
    reward = 0.0
    while True:
        obs = obs.reshape(-1)
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        if done:
            break
    #env.close()
    #env.env.close()
    return reward


def mutate_parent(net):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net


from functools import reduce


if __name__ == "__main__":
    writer = SummaryWriter(comment="-cartpole-ga")
    #env = gym.make("CartPole-v1")
    env = gym.make("PongNoFrameskip-v4")
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    obs_shape = env.observation_space.shape
    obs_size = ln = reduce(lambda x, y: x * y, obs_shape)

    gen_idx = 0
    nets = [
        Net(obs_size, env.action_space.n)
        for _ in range(POPULATION_SIZE)
    ]#N agents
    population = [
        (net, evaluate(env, net,obs_size))
        for net in nets
    ]#n pairs of agent and fitness: initialization of the population
    while True:
        population.sort(key=lambda p: p[1], reverse=True)#order by the second term of the pair,i.e.,the fitness value
        rewards = [p[1] for p in population[:PARENTS_COUNT]]#the top PARENTS_COUNT individuals in the population
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)

        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f" % (
            gen_idx, reward_mean, reward_max, reward_std))
        if reward_mean > 199:
            print("Solved in %d steps" % gen_idx)
            break

        # generate next population
        prev_population = population
        population = [population[0]]#the best in the last generation is copied directly
        for _ in range(POPULATION_SIZE-1):#generate POPULATION_SIZE-1 new mutations
            parent_idx = np.random.randint(0, PARENTS_COUNT)#select from the top PARENTS_COUNT as parents for the next generation
            parent = prev_population[parent_idx][0]
            net = mutate_parent(parent)
            fitness = evaluate(env, net)
            population.append((net, fitness))
        gen_idx += 1

    pass
