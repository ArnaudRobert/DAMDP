import gym
import numpy as np

from gym import Env, error, spaces, utils
from gym.utils import seeding


LEFT = 0
RIGHT = 1


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class RiverSwimEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nS=6, H=50):
        """
        Has the following members
        - nS: number of states
        - nA: number of actions
        - H: the horizon of the problem
        - P: transitions (*)
        - isd: initial state distribution (**)

        (*) dictionary of lists, where
        P[s][a] := [(probability, nextState, reward, done), ...]
        (**) list of nS probabilities
        """
        # Defining basic properties of the system
        self.nA = 2
        self.nS = nS
        self.lastaction = None
        self.t = 0
        self.H = H

        # Defining the reward system and dynamics of RiverSwim environment
        self.P, self.isd = self.__init_dynamics(self.nS, self.nA)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def __init_dynamics(self, nS, nA):
        # P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # Rewarded Transitions
        P[0][LEFT] = [(1., 0, 5/1000, 0)]
        P[nS-1][RIGHT] = [(0.9, nS-1, 1, 0), (0.1, nS-2, 1, 0)]

        # Left Transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1., max(0, s-1), 0, 0)]

        # RIGHT Transitions
        for s in range(1, nS - 1):
            P[s][RIGHT] = [(0.3, min(nS - 1, s + 1), 0, 0), (0.6, s, 0, 0), (0.1, max(0, s-1), 0, 0)]
        P[0][RIGHT] = [(0.3, 0, 0, 0), (0.7, 1, 0, 0)]

        # Starting State Distribution
        isd = np.zeros(nS)
        isd[0] = 1.

        return P, isd

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.t = 0
        return int(self.s), self.t

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, _ = transitions[i]
        self.t += 1
        d = int(self.t >= self.H)
        self.s = s
        self.lastaction = a
        return int(s), r, d, self.t, {"prob": p}

    def get_Q(self):
        """
        Return Q_0(s, a) for all s, a.
        """
        qVals = np.zeros((self.H, self.nS, self.nA))
        qMax = np.zeros((self.H+1, self.nS))
        for i in reversed(range(self.H)):
            for s in range(self.nS):
                for a in range(self.nA):
                    transitions = self.P[s][a]
                    ret = 0
                    for p, nextS, r, _ in transitions:
                        ret += p * (r + qMax[i+1, nextS])
                    qVals[i, s, a] = ret
                qMax[i, s] = np.max(qVals[i, s, :])

        return qVals, qMax

