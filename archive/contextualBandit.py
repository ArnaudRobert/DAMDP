import numpy as np
from graphenv_v2 import GraphBandit
import networkx as nx
import matplotlib.pyplot as plt


class ThompsonSamplingContextualBandit():
    """
    Implementation of a Thompson sampling algoritm
    for contextual bandit in tabular case.
    """
    def __init__(self, env):
        """
        Set up the necessary variables for running the alg.
        The learner access all possible action through their index.
        """
        print("create bandit learner")
        self.env = env
        self.A = self.env.GF.A
        self.R = {}
        self.outcomes = self.env.GF.bandit_outcomes
        self.prior()

    def prior(self):
        """
        The prior is a is an indicator function that check for overflow
        and a categorical variable.
        The indicator is parameterized by Bernoulli fct.
        The reward is parametrise by a categorical.
        """
        for a in np.arange(len(self.A)):
            thetas = [1. for _ in self.outcomes]
            self.R[a] = thetas

    def update_obs(self, a, r):
        """
        Update the belief.
        Input:
            - a : the action
            - r : the reward
        """
        thetas = self.R[a]
        # update params of multinomial
        k = self.outcomes.index(r)
        thetas[k] = 1 + thetas[k]
        self.R[a] = thetas

    def sample(self):
        R_hat = {}
        for a, thetas in self.R.items():
            ps = np.random.dirichlet(thetas)
            # get model
            reward = np.random.choice(self.outcomes, size=1, p=ps)
            R_hat[a] = reward
        return R_hat

    def select_action(self):
        # find action that maximize
        best = 0
        a_stars = [] 
        a_star = np.random.choice(len(self.A))
        R_hat = self.sample()
        for a, r in R_hat.items():
            if r > best:
                a_stars = [a]
                best = r
            elif r == best:
                a_stars.append(a)
        return np.random.choice(a_stars, size=1)[0]

    def learn(self, T):
        rewards = []
        for t in range(T):
            self.env.reset()
            a = self.select_action()
            reward = self.env.act(a)
            self.update_obs(a, reward)
            rewards.append(reward)


