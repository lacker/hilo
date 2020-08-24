#!/usr/bin/env python

import random
import sys

import gym
from gym import spaces

RANGE = 256


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
        return [self.lower_bound, self.upper_bound]

    def reset(self):
        self.secret = random.randrange(RANGE)
        self.lower_bound = 0
        self.upper_bound = RANGE - 1
        self.message = ""

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
    game = HiloEnv()
    print(f"guess a number from 0 to {RANGE - 1}.")
    while True:
        s = input("guess: ")
        try:
            number = int(s)
        except ValueError:
            print("bad number")
            continue

        _, _, done, _ = game.step(number)
        game.render()
        if done:
            break


if __name__ == "__main__":
    if "--play" in sys.argv:
        play_human()
    elif "--check" in sys.argv:
        raise NotImplementedError("need to implement this")
    else:
        print("use a flag plz")
