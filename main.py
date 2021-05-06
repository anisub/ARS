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


class hyper_params():

    def __init__(self, seed=1, episode_length=1000):
        self.alpha = 1e-3
        self.env_name = 'HalfCheetah-v2'
        self.noise = 0.03
        self.num_directions = 16
        self.best_directions = 16
        self.episode_length = 1000
        self.seed = 1
        self.num_steps = 1000
        self.record_flag = 50


class Normalizer():

    def __init__(self, ob_dim):
        self.n = ob_dim
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
        return (input - ob_mean) / ob_std


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

    def update(self, rollout, sigma_r):
        fd_rewards = np.zeros(self.policy_params.shape)
        for rp, rn, d in rollout:
            fd_rewards += (rp - rn) * d
        self.policy_params += self.hyper_params.alpha / \
            (self.hyper_params.best_directions * sigma_r) * fd_rewards


class ARS():

    def __init__(self,
                 hyper_params=None,
                 monitor_dir=None):

        self.hyper_params = hyper_params or hyper_params()
        np.random.seed(self.hyper_params.seed)
        self.env = gym.make(self.hyper_params.env_name)

        if monitor_dir is not None:
            def should_record(i): return self.record_video
            env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)

        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.shape[0]
        self.normalizer = Normalizer(self.ob_dim)
        self.policy = Policy(self.ob_dim, self.ac_dim, self.hyper_params)
        self.record_video = False

    def explore(self, delta=None, direction=None):
        state = self.env.reset()
        done = False
        sum_rewards = 0
        counter = 0
        while not done and counter < self.hyper_params.episode_length:
            self.normalizer.running_stats(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(input=state, delta=delta, direction=direction)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
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
    def train(self):
        for step in range(self.hyper_params.num_steps):

            # Perform rollout and calculate std deviation
            rollouts, sigma_r = self.generate_rollout()

            # Update policy
            self.policy.update(rollouts, sigma_r)

            if step % self.hyper_params.record_flag == 0:
                record_video = True

            final_reward = self.explore()
            print('Step:', step, 'Reward:', final_reward)
            rewards_plot.append(final_reward)
            rewards_deviation.append(sigma_r)
            record_video = False


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    rewards_plot = []
    rewards_deviation = []
    video_dir = mkdir('.', 'videos')
    monitor_dir = mkdir(video_dir, 'HalfCheetah-v2')
    hp = hyper_params(seed=1946)
    trainer = ARS(hyper_params=hp, monitor_dir=monitor_dir)
    trainer.train()

    plt.figure()
    plt.plot(np.arange(0, hp.num_steps), rewards_plot,
             color='#1B2ACC', marker='s', linewidth=2, markersize=6)
    pos_deviation = np.array(rewards_plot) - np.array(rewards_deviation)
    neg_deviation = np.array(rewards_plot) + np.array(rewards_deviation)
    plt.fill_between(np.arange(0, hp.num_steps), neg_deviation,
                     pos_deviation, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.title('HalfCheetah-v2')
    plt.xlabel('Iterations')
    plt.ylabel('Rewards')
    plt.show()
