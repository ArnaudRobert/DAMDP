import numpy as np
from utils import get_layers
from toyenv import FlowEnv 
import matplotlib.pyplot as plt

class GQlearning():

    def __init__(self, env, state_space=2, action_space=2,
                 alpha=0.3, gamma=0, eps=0.2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # q function is a list for each layer
        #  
        self.G = self.env.G
        self.layers = get_layers(self.G)
        self.Q = []
        for l, nodes in self.layers.items():
            a_size = action_space**len(nodes)
            s_size = state_space**len(nodes)
            self.Q.append(np.zeros((s_size, a_size)))
            print(f"layer {l} has dimension {self.Q[l].shape}")
    def idx2value(self, idx, num):
        s = bin(idx)[2:].zfill(num)
        v = [int(c) for c in s]
        return v

    def value2idx(self, v):
        s = [str(vv) for vv in v]
        s = "".join(s)
        idx = int(s, 2)
        return idx

    def select_action(self, layer, state, greedy=False):
        Q = self.Q[layer]
        s_id = self.value2idx(state)
        if np.random.uniform(0, 1) < self.eps and not greedy:
            a = np.random.binomial(1, 0.5, len(self.layers[layer]))
        else:
            a = np.argmax(Q[s_id, :])
            a = self.idx2value(a, len(self.layers[layer]))
        return a

    def get_V(self, g):
        pass

    def plot_policy(self, goal, room=None):
        pass

    def update(self, transition):
        s, action, r, nxt_s, done, layer = transition 
        a_id = self.value2idx(action)
        s_id = self.value2idx(s)
        nxt_s_id = self.value2idx(nxt_s)
        Q = self.Q[layer]
        Q_nxt = self.Q[layer + 1]
        q_sa = Q[s_id, a_id]
        max_nxt_q = np.max(Q_nxt[nxt_s_id, :])
        # make the update
        new_q_sa = q_sa + self.alpha * (r + (1 - int(done)) * self.gamma * max_nxt_q - q_sa)
        Q[s_id, a_id] = new_q_sa

    def train(self):
        episodes = 800
        rewards = []
        for ep in range(episodes):
            reward = 0
            s = self.env.reset()
            done = False
            while not done:
                a = self.select_action(self.env.t, s)
                nxt_s, r, done = self.env.step(a)
                t = (s, a, r, nxt_s, done, self.env.t - 1)
                self.update(t)
                s = nxt_s
                reward += r
            rewards.append(reward)

        return rewards


if __name__ == "__main__":
    env = FlowEnv()
    agent = GQlearning(env)
    rewards = agent.train()
    plt.plot(rewards)
    plt.show()

