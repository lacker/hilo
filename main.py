#!/usr/bin/env python

import random
import sys

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

RANGE = 31
MODEL = "ppo_basic"


class HiloEnv(gym.Env):
    """
    A custom environment for the hilo game.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(HiloEnv, self).__init__()

        self.action_space = spaces.Discrete(RANGE)
        self.observation_space = spaces.MultiDiscrete([RANGE, RANGE])
        self.reset()

    def observe(self):
        return np.array([self.lower_bound, self.upper_bound])

    def reset(self):
        self.secret = random.randrange(RANGE)
        self.lower_bound = 0
        self.upper_bound = RANGE - 1
        self.message = ""
        return self.observe()

    def step(self, action):
        """action is a number to be guessed"""
        if action <= self.secret:
            self.lower_bound = max(action, self.lower_bound)
            self.message = f"{action} is too low."
        if action >= self.secret:
            self.upper_bound = min(action, self.upper_bound)
            self.message = f"{action} is too high."
        if action == self.secret:
            self.message = f"{action} is correct!"
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return (self.observe(), reward, done, {})

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError
        print(self.message)


def play_human():
    env = HiloEnv()
    print(f"guess a number from 0 to {RANGE - 1}.")
    while True:
        s = input("guess: ")
        try:
            number = int(s)
        except ValueError:
            print("bad number")
            continue

        _, _, done, _ = env.step(number)
        env.render()
        if done:
            break


def check():
    env = HiloEnv()
    check_env(env)


def train():
    env = Monitor(HiloEnv(), "./tmp/")
    model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./tboard_log")
    model.learn(total_timesteps=250000)
    model.save(MODEL)


def demo():
    model = DQN.load(MODEL)
    env = HiloEnv()
    obs = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == "__main__":
    if "--play" in sys.argv:
        play_human()
    elif "--check" in sys.argv:
        check()
    elif "--train" in sys.argv:
        train()
    elif "--demo" in sys.argv:
        demo()
    else:
        print("use a flag plz")
