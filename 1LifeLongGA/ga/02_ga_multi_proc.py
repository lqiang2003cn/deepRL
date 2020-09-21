#!/usr/bin/env python3
import sys
import gym
import pybullet_envs
import collections
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from halfcheetah_ga_import import OutputItem

from tensorboardX import SummaryWriter


NOISE_STD = 0.01
POPULATION_SIZE = 512
PARENTS_COUNT = 32

WORKERS_COUNT = 8
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1



class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mu(x)


def evaluate(env, net):
    obs = env.reset()
    from functools import reduce
    obs_shape = env.observation_space.shape
    obs_size = reduce(lambda x, y: x * y, obs_shape)

    reward = 0.0
    steps = 0
    while True:
        obs = obs.reshape(-1)
        obs_v = torch.FloatTensor([obs])
        action_v = net(obs_v)
        acts = action_v.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net


def build_net(env, seeds):
    torch.manual_seed(seeds[0])

    from functools import reduce
    obs_shape = env.observation_space.shape
    obs_size = reduce(lambda x, y: x * y, obs_shape)

    net = Net(obs_size, env.action_space.n)
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net





def worker_func(input_queue, output_queue):
    #env = gym.make("HalfCheetahBulletEnv-v0")
    env = gym.make("PongNoFrameskip-v4")
    cache = {}

    while True:
        parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        #print('parent seed:', parents)
        for net_seeds in parents:

            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1])
                else:
                    net = build_net(env, net_seeds)
            else:
                net = build_net(env, net_seeds)
            new_cache[net_seeds] = net
            reward, steps = evaluate(env, net)
            output_queue.put(OutputItem(seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache
        #print('cache is ')


if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-cheetah-ga")

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:#keep getting results from the
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
                #print('seed after sort:',seeds)
            worker_queue.put(seeds)
        gen_idx += 1

    pass
