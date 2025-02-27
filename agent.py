import numpy as np


class PSGRL:
    def __init__(self, env, epsilon=0.00):

        self.qVals = {}  # qVals[state, timestep] is the vector of Q values for each action
        self.qMax = {}   # qMax[timestep] is the vector of optimal values

        self.r_prior = {}
        self.p_prior = {}

        self.env = env
        self.epsilon = epsilon

        self.state_space = env.state_space
        self.action_space = env.action_space
        self.atomic_states = env.atomic_states
        self.atomic_actions = env.atomic_actions
        self.reward_Cs = env.r.keys()
        self.transition_Cs = env.p.keys()
        self.outcomes = env.outcomes

        self.X = len(self.atomic_states)
        self.Y = len(self.atomic_actions)

        # dirichlet prior on the reward distribution
        for c in self.reward_Cs:
            self.r_prior[c] = np.ones(len(self.outcomes))

        for c in self.transition_Cs:
            self.p_prior[c] = np.ones(self.X)

    def update_obs(self, rewards, new_states, done, contexts):
        """
        Given a set of atomic transition update
        """
        reward_Cs = contexts['rewards']
        transition_Cs = contexts['transitions']
        for nodeID, r in rewards.items():
            c = reward_Cs[nodeID]
            ri = self.outcomes.index(r)
            self.r_prior[c][ri] += 1

        if not new_states is None:
            for nodeID, state in new_states.items():
                c = transition_Cs[nodeID]
                si = self.atomic_states.index(state)
                self.p_prior[c][si] += 1

    def pick_action(self, state, timestep):
        """
        Epsilon greedy strategy.
        """
        si = self.state_space[timestep].index(state)
        Q = self.qVals[si, timestep]
        noise = np.random.uniform()
        if noise < self.epsilon:
            action = np.random.choice(len(self.action_space[timestep]))
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])
        return action

    def update_policy(self):
        """
        Compute qvals and qMax
        """
        # sample an DAMDP
        p_samp, r_samp = self.sample_mdp()

        # solve the DAMDP
        self.qVals, self.qMax = self.env.solve(p_samp, r_samp)

    def sample_mdp(self):
        """
        Return a set of atomic transition / reward functions.
        """
        p_samp = {}
        r_samp = {}

        for c in self.transition_Cs:
            p_samp[c] = np.random.dirichlet(self.p_prior[c])
        for c in self.reward_Cs:
            r_samp[c] = np.random.dirichlet(self.r_prior[c])
        return p_samp, r_samp


class PSRL:

    def __init__(self, env, epsilon=0.0):
        """
        Posterior sampling RL
        """
        self.epsilon = epsilon
        self.env = env
        self.state_space = self.env.state_space
        self.action_space = self.env.action_space

        self.R_outcomes = self.env.layer_outcomes
        self.H = env.H
        self.alpha0 = 1.
        self.beta0 = 1.

        self.P = {}
        self.R = {}
        self.prior()

    def prior(self):
        """
        """
        # set the prior for R
        for t in range(1, self.H):
            for state in self.state_space[t]:
                for action in self.action_space[t]:
                    R_thetas0 = [1. for _ in self.R_outcomes]
                    self.R[state, action] = R_thetas0
                    P_thetas0 = [1. for _ in self.state_space[t+1]]
                    self.P[state, action, t] = P_thetas0

    def update_policy(self):
        """
        Sample an MDP from posterior and solve
        """
        P_hat, R_hat = self.sample_mdp()
        # solve the MDP via value iteration
        qVals, qMax = self.env.solve_layer_wise(P_hat, R_hat, self.R_outcomes)
        # update the agent
        self.qVals = qVals
        self.qMax = qMax

    def sample_mdp(self):
        """
        """
        R_samp = {}
        P_samp = {}
        for t in range(1, self.H):
            for s in self.state_space[t]:
                for a in self.action_space[t]:
                    R_samp[s, a] = np.random.dirichlet(self.R[s, a])
                    P_samp[s, a, t] = np.random.dirichlet(self.P[s, a, t])

        return P_samp, R_samp

    def update_obs(self, state, action, r, nxt_state, done, t, skip_transition=False):
        """
        """
        ri = self.R_outcomes.index(r)
        self.R[state, action][ri] += 1

        if not skip_transition:
            nxt_i = self.state_space[t].index(nxt_state)
            self.P[state, action, t-1][nxt_i] += 1

    def pick_action(self, state, timestep):
        """
        Epsilon greedy strategy.
        """
        si = self.state_space[timestep].index(state)
        Q = self.qVals[si, timestep]
        noise = np.random.uniform()
        if noise < self.epsilon:
            action = np.random.choice(len(self.action_space[timestep]))
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])
        return action
