#!/usr/bin/env python

import random
import sys

import gym
from gym import spaces

RANGE = 256


class Env(gym.Env):
    """
    A custom environment for the hilo game.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(Env, self).__init__()

        self.action_space = spaces.Discrete(RANGE)

        # TODO: set up self.observation_space


class Game(object):
    def __init__(self, verbose=False):
        self.secret = random.randint(0, 255)
        self.verbose = verbose

        # For now, share the upper and lower bounds with the player.
        # These are inclusive.
        self.lower_bound = 0
        self.upper_bound = RANGE - 1
        self.log(
            f"I am thinking of a number from {self.lower_bound} to {self.upper_bound}."
        )

    def log(self, message):
        if self.verbose:
            print(message)

    def guess(self, number):
        if number < self.secret:
            self.log(f"{number} is too low")
            self.lower_bound = max(number, self.lower_bound)
            return False
        if number > self.secret:
            self.log(f"{number} is too high")
            self.upper_bound = min(number, self.upper_bound)
            return False
        self.log(f"{number} is correct!")
        return True


def play_human():
    game = Game(verbose=True)
    while True:
        s = input("guess: ")
        try:
            number = int(s)
        except ValueError:
            print("bad number")
            continue

        if game.guess(number):
            break


if __name__ == "__main__":
    if "--play" in sys.argv:
        play_human()
    elif "--check" in sys.argv:
        raise NotImplementedError("need to implement this")
    else:
        print("use a flag plz")
