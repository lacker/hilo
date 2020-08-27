#!/usr/bin/env python

from datetime import timedelta
import random
import sys
import time

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

BITS = 7
RANGE = 2 ** BITS - 1
CORRECT = 10.0
INCORRECT = -1.0
MODEL = "ppo_basic"
PARALLELISM = 1


def optimal(r=RANGE):
    """The expected episode reward for optimal play"""
    if r <= 1:
        return CORRECT
    odds_correct = 1.0 / r
    odds_incorrect = 1 - odds_correct
    return odds_correct * CORRECT + odds_incorrect * (INCORRECT + optimal((r - 1) / 2))


def to_bits(n):
    answer = []
    for _ in range(BITS):
        answer.append(n % 2)
        n = n // 2
    answer.reverse()
    return answer


def from_bits(bits):
    answer = 0
    for bit in bits:
        answer = 2 * answer + bit
    return answer


class HiloEnv(gym.Env):
    """
    A custom environment for the hilo game.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(HiloEnv, self).__init__()

        self.action_space = spaces.MultiDiscrete([2] * BITS)
        self.observation_space = spaces.MultiDiscrete([2] * (2 * BITS))
        self.reset()

    def observe(self):
        return np.array(to_bits(self.lower_bound) + to_bits(self.upper_bound))

    def reset(self):
        self.secret = random.randrange(RANGE)
        self.lower_bound = 0
        self.upper_bound = RANGE - 1
        self.steps = 0
        self.message = ""
        return self.observe()

    def step(self, action):
        """action is a number to be guessed, in bits form"""
        self.steps += 1
        action = from_bits(action)
        if action <= self.secret:
            self.lower_bound = max(action, self.lower_bound)
            self.message = f"{action} is too low."
        if action >= self.secret:
            self.upper_bound = min(action, self.upper_bound)
            self.message = f"{action} is too high."
        if action == self.secret:
            self.message = f"{action} is correct!"
            reward = CORRECT
            done = True
        else:
            reward = INCORRECT
            done = False

        if self.steps >= 2 * RANGE:
            # Prevent the game from lasting infinite time.
            done = True

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

        _, _, done, _ = env.step(to_bits(number))
        env.render()
        if done:
            break


def check():
    print("reward for optimal play:", optimal())
    env = HiloEnv()
    check_env(env)


def make_env():
    return Monitor(HiloEnv(), "./tmp/")


def make_eval_callback():
    eval_env = make_env()
    return EvalCallback(eval_env, eval_freq=1000, deterministic=True, render=False)


def train():
    if PARALLELISM > 1:
        env = SubprocVecEnv([make_env] * PARALLELISM)
    else:
        env = make_env()
    net_arch = [{"pi": [64, 64], "vf": [64, 64]}]
    policy_kwargs = dict(net_arch=net_arch)
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tboard_log",
    )
    start = time.time()
    model.learn(total_timesteps=3000000)
    elapsed = time.time() - start
    print(f"{timedelta(seconds=elapsed)} time elapsed")
    model.save(MODEL)


def demo():
    model = PPO.load(MODEL)
    print(model.policy.net_arch)
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
