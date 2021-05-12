# REFERNCE: https://github.com/colinskow/move37/tree/master/ars

'''
ESE 650: Final Project: Augmented Random Search (ARS)
Team 23
Anirudh Subramanyam : anisub @ seas.upenn.edu
Christopher Kennedy : cwkenned @ seas.upenn.edu
'''

import parser
import time
import os
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import save
from numpy import asarray

ENV_NAMES = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']


class hyper_params():

    def __init__(self, seed=1, episode_length=1000):
        self.alpha = 1e-3
        self.env_name = 'Hopper-v2'
        self.noise = 0.02
        self.num_directions = 4
        self.best_directions = 2
        self.episode_length = 1000
        self.seed = 1
        self.num_steps = 4500


class Normalizer():

    def __init__(self, ob_dim):
        self.n = np.zeros(ob_dim)
        self.mean = np.zeros(ob_dim)
        self.var = np.zeros(ob_dim)
        self.mean_difference = np.zeros(ob_dim)

    def running_stats(self, obs):
        self.n += 1
        self.previous_mean = self.mean.copy()
        self.mean += (obs - self.mean) / self.n
        self.mean_difference += (obs - self.previous_mean) * (obs - self.mean)
        self.var = self.mean_difference / self.n

    def normalize(self, input):
        ob_mean = self.mean
        ob_std = np.sqrt(self.var)
        np.seterr(divide='ignore')
        output = (input - ob_mean) / ob_std
        return np.nan_to_num(output, 0)


class Policy():
    '''
    ob_dim: dimension of the obseravtion space
    ac_dim: dimension of the action space
    policy_params : parameters theta of policy pi
    '''

    def __init__(self, ob_dim, ac_dim, hyper_params):
        self.policy_params = np.zeros((ac_dim, ob_dim))
        self.hyper_params = hyper_params

    def sample_deltas(self):
        return [np.random.randn(*self.policy_params.shape) for _ in range(self.hyper_params.num_directions)]

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.policy_params.dot(input)
        elif direction == 'positive':
            return (self.policy_params + self.hyper_params.noise * delta).dot(input)
        elif direction == 'negative':
            return (self.policy_params - self.hyper_params.noise * delta).dot(input)

    def update(self, rollout, sigma_r, variant):
        fd_rewards = np.zeros(self.policy_params.shape)
        for rp, rn, d in rollout:
            fd_rewards += (rp - rn) * d
        if variant[0]:
            self.policy_params += (self.hyper_params.alpha / sigma_r) * fd_rewards
        elif variant[1]:
            self.policy_params += (self.hyper_params.alpha /
                                   (self.hyper_params.best_directions * sigma_r)) * fd_rewards
        elif variant[2]:
            self.policy_params += (self.hyper_params.alpha / sigma_r) * fd_rewards
        elif variant[3]:
            self.policy_params += (self.hyper_params.alpha /
                                   (self.hyper_params.best_directions * sigma_r)) * fd_rewards


class ARS():

    def __init__(self,
                 hyper_params=None):

        self.hyper_params = hyper_params or hyper_params()
        np.random.seed(self.hyper_params.seed)
        self.env = gym.make(self.hyper_params.env_name)
        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.shape[0]
        self.normalizer = Normalizer(self.ob_dim)
        self.policy = Policy(self.ob_dim, self.ac_dim, self.hyper_params)

    def explore(self, delta=None, direction=None):
        state = self.env.reset()
        done = False
        sum_rewards = 0
        counter = 0
        # Use shift = 1 for any environment with bonus survival reward
        shift = 0
        while not done and counter < self.hyper_params.episode_length:
            self.normalizer.running_stats(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(input=state, delta=delta, direction=direction)
            state, reward, done, _ = self.env.step(action)
            sum_rewards += (reward - shift)
            counter += 1
        return sum_rewards

    def generate_rollout(self):

        # Sample deltas
        deltas = self.policy.sample_deltas()

        # Define the positive and negative fd_rewards
        positive_rewards = [0] * self.hyper_params.num_directions
        negative_rewards = [0] * self.hyper_params.num_directions

        # Compute the fd_rewards
        for n in range(self.hyper_params.num_directions):
            positive_rewards[n] = self.explore(deltas[n], direction='positive')
            negative_rewards[n] = self.explore(deltas[n], direction='negative')

        # Compute standard deviation of 2b fd_rewards
        std_dev = np.array(positive_rewards+negative_rewards).std()

        # Sort the deltas according to the maximum of the pos and neg fd_rewards
        dict_directions = {k: max(r_pos, r_neg) for k, (r_pos, r_neg)
                           in enumerate(zip(positive_rewards, negative_rewards))}
        best_order = sorted(dict_directions.keys(), key=lambda x: dict_directions[x])[
            :self.hyper_params.best_directions]
        outputs = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in best_order]
        return outputs, std_dev

    # Define the training loop
    def train(self, variant):
        for step in tqdm(range(self.hyper_params.num_steps)):

            # Perform rollout and calculate std deviation
            rollouts, sigma_r = self.generate_rollout()

            # Update policy
            self.policy.update(rollouts, sigma_r, variant)

            final_reward = self.explore()
            print('Step:', step, 'Reward:', final_reward)
            rewards_plot.append(final_reward)
            rewards_deviation.append(sigma_r)


if __name__ == '__main__':
    rewards_plot = []
    rewards_deviation = []
    variant = [False, False, True, False]

    hp = hyper_params(seed=1946)
    trainer = ARS(hyper_params=hp)
    trainer.train(variant)

    # Saving files locally
    rewards_out = asarray(rewards_plot)
    std_out = asarray(rewards_deviation)

    save('Hop-ARS-V1_reward', rewards_out)
    save('Hop-ARS-V1_std', std_out)

    # Plotting
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(np.arange(0, hp.num_steps), rewards_plot,
             color='b', linewidth=1)
    pos_deviation = np.array(rewards_plot) - np.array(rewards_deviation)
    neg_deviation = np.array(rewards_plot) + np.array(rewards_deviation)
    plt.fill_between(np.arange(0, hp.num_steps), neg_deviation,
                     pos_deviation, facecolor='#9FD9FF')
    plt.title('Hopper-v2 environment running ARS V1')
    plt.xlabel('Iterations')
    plt.ylabel('Rewards')
    plt.show()
