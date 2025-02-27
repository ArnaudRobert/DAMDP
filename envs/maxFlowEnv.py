import numpy as np
import networkx as nx
from networkx import traversal as T
from itertools import product


class MaxFlowEnv():
    """
    the rules are the following:

    the goal is to transport as much flow from a source node to a sink node.

    each edge has a capacity, if the flow exceed the capacity the amount transported is 0.
    each edge has probability of failure p, if the edge fails the amount transported is 0.
    lastly if the total amount of flow instructed to leave the node is larger than the amount of
    flow currently in the node - this result in an illegal action and the flow transported is also 0.

    """
    def __init__(self, G, c, fail, max=2):
        """
        Input:
            - G    : is the graph structure.
            - c    : is the capacity of each edge.
            - fail : is the failing probability of each edge.
        """
        self.G = G
        self.c = c
        self.fail = fail
        self.max = 2
        self.H = nx.dag_longest_path_length(self.G) + 1

        self.atomic_states = [0, 1, 2]
        self.atomic_actions = [0, 1, 2]
        self.X = len(self.atomic_states)
        self.Y = len(self.atomic_actions)
        self.outcomes = [0, 1, 2]

        self.p = {}
        self.r = {}

        # the parent node has two edges
        for s in self.atomic_states:
            for a1 in self.atomic_actions:  # the edge that connects to the node of interest.
                for a2 in self.atomic_actions:
                    if (s, a1, a2) in self.p:
                        continue
                    self.p[s, a1, a2] = np.zeros(self.X)
                    if a1 + a2 > s:  # we distribute more flow than available
                        self.p[s, a1, a2][0] = 1.0
                        continue
                    self.p[s, a1, a2][0] = self.fail
                    flow = a1 # a2 is not connected to the next node.
                    if flow > self.c:
                        flow = 0
                    self.p[s, a1, a2][flow] += 1 - self.fail

        for s1 in self.atomic_states:
            for a1 in self.atomic_actions:
                for s2 in self.atomic_states:
                    for a2 in self.atomic_actions:
                        # make sure we did not computed this entry
                        if (s1, a1, s2, a2) in self.p:
                            continue
                        # unrealist actions are not in the state space
                        if s1 + s2 > self.max:
                            continue
                        self.p[s1, a1, s2, a2] = np.zeros(self.X)
                        if a1 > s1:
                            flow1 = 0
                        elif a1 > self.c:
                            flow1 = 0
                        else:
                            flow1 = a1
                        self.p[s1, a1, s2, a2][0] += fail**2
                        if a2 > s2:
                            flow2 = 0
                        elif a2 > self.c:
                            flow2 = 0
                        else:
                            flow2 = a2
                        self.p[s1, a1, s2, a2][flow1] += fail*(1-fail)
                        self.p[s1, a1, s2, a2][flow2] += fail*(1-fail)
                        self.p[s1, a1, s2, a2][flow1+flow2] += (1-fail)*(1-fail)

        # single node with single edge
        for s in self.atomic_states:
            for a in self.atomic_actions:
                if (s, a) in self.r:
                    continue
                self.r[s, a] = np.zeros(len(self.outcomes))
                self.r[s,a][0] = fail
                if a > s:
                    flow = 0
                else:
                    flow = a
                if flow > self.c:
                    flow = 0
                self.r[s, a][flow] += 1 - fail


        for s in self.atomic_states:
            for a1 in self.atomic_actions:
                for a2 in self.atomic_actions:
                    # make sure we did not computed this entry
                    if (s, a1, a2) in self.r:
                        continue
                    # remove unrealistic actions
                    if a1 + a2 > self.max:
                        continue

                    self.r[s, a1, a2] = np.zeros(len(self.outcomes))
                    if a1 + a2 > s:   # try to distribute more flow than available
                        self.r[s, a1, a2][0] = 1
                    else:
                      self.r[s, a1, a2][0] += self.fail**2
                      flow1 = a1
                      if flow1 > self.c:
                          flow1 = 0
                      flow2 = a2
                      if flow2 > self.c:
                          flow2 = 0
                      self.r[s, a1, a2][flow1] += self.fail * (1 - self.fail)
                      self.r[s, a1, a2][flow2] += self.fail * (1 - self.fail)
                      self.r[s, a1, a2][flow1 + flow2] += (1 - self.fail)**2

        self.action_space, self.edge_layers = self._build_action_space()
        self.state_space, self.node_layers, self.valid_states = self._build_state_space()


    def _build_action_space(self):
        """
        Build the layer wise action space i.e. actions selected by the policy.
        """
        edge_layers = {}
        space = {}
        for t, layer in enumerate(T.bfs_layers(self.G, 1)):
            if t == self.H:
                continue
            edges = list(self.G.out_edges(layer))
            dim = len(edges)
            actions = product(self.atomic_actions, repeat=dim)
            space[t+1] = self._filter_action_space(actions)
            edge_layers[t+1] = edges
        return space, edge_layers

    def _build_state_space(self):
        """
        Build the layer wise state space i.e. the one observed by the policy.
        """
        space = {}
        node_layers = {}
        valid = {}
        for t, layer in enumerate(T.bfs_layers(self.G, 1)):
            states = product(self.atomic_states, repeat=len(layer))
            space[t+1], valid[t+1] = self._filter_state_space(states)
            node_layers[t+1] = list(layer)
        return space, node_layers, valid

    def _filter_action_space(self, actions):
        action_space = []
        for action in actions:
            if np.sum(action) > self.max:
                continue
            action_space.append(action)
        return action_space

    def _filter_state_space(self, states):
        state_space = []
        states = list(states)
        valid = np.zeros(len(states), dtype=bool)
        for i, state in enumerate(states):
            if np.sum(state) > self.max:
                continue
            state_space.append(state)
            valid[i] = 1
        return state_space, valid

    def act(self, a):
        """
        Input:
            - a : the action list of actions for each node
        Outpt:
            - s    : the state for all next nodes
            - r    : the reward for all current nodes
            - done : boolean flag if the episode is done
        """

        current_state = []
        nxt_state = {}
        for node in self.node_layers[self.t]:
            current_state.append(self.G.nodes[node]['flow'])

        act = self.action_space[self.t][a]
        edges = self.edge_layers[self.t]
        rewards = {}
        contexts = {}
        contexts['rewards'] = {}
        contexts['transitions'] = {}
        for k, edge in enumerate(edges):
            n1 = edge[0]
            n2 = edge[1]
            current = self.G.nodes[n1]['flow']
            flow = min(current, act[k])  # cannot transport more than available.
            if flow > self.c:            # if too much is transported it fails.
                flow = 0
            if np.random.uniform() < self.fail:  # the edge failed.
                flow = 0 
            self.G.nodes[n2]['flow'] += flow
            self.G.nodes[n1]['flow'] -= flow
            if n1 not in rewards:
                rewards[n1] = flow
            else:
                rewards[n1] += flow

        for node in self.node_layers[self.t+1]:
            nxt_state[node] = self.G.nodes[node]['flow']


        # build the contexts
        for node in self.node_layers[self.t]:
            contexts['rewards'][node] = self.get_reward_context(node, current_state, act, self.t)
        for node in self.node_layers[self.t+1]:
            contexts['transitions'][node] = self.get_transition_context(node, current_state, act, self.t) 
        self.t += 1
        done = False


        if self.t > self.H-1:
            done = True
        return nxt_state, rewards, done, self.t, contexts

    def get_reward_context(self, node, s, a, t):
        """
        given a node a layer wise state and action returns
        the reward context.
        """

        n = self.node_layers[t].index(node)
        context = (s[n],)
        for i, aa in enumerate(self.edge_layers[t]):
            if aa[0] == node:
                context += (a[i],)
        return context

    def get_transition_context(self, nxt_node, s, a, t):
        """
        """
        c = tuple()
        for ai, edge in enumerate(self.edge_layers[t]):
            if edge[1] == nxt_node:
                ni = self.node_layers[t].index(edge[0])
                c += (s[ni], a[ai])
                for a2i, edge2 in enumerate(self.edge_layers[t]):
                    if edge[0] == edge2[0] and (not edge2[1] == nxt_node):
                        c += (a[a2i],)
        return c

    def solve(self, p, r):
        """
        Given an atomic transition and reward function.
        Find the optimal policy and Q-function.
        """
        # qVals[state, timestep] is the vector of Q values for each action
        # qMax[timestep] is the vector of optimal values

        qVals = {}
        qMax = {}
        qMax[self.H] = np.zeros(len(self.state_space[self.H]))

        for i in range(self.H-1):
            j = self.H - i - 1
            qMax[j] = np.zeros(len(self.state_space[j]))

            for si, state in enumerate(self.state_space[j]):
                qVals[si, j] = np.zeros(len(self.action_space[j]))

                for ai, a in enumerate(self.action_space[j]):
                    # get the shape of all contexts... 
                    mean_r = 0
                    for atomic_n in self.node_layers[j]:
                        c = self.get_reward_context(atomic_n, state, a, j)
                        mean_r += np.dot(r[c], self.outcomes)
                    #print(f"the expected return {mean_r}")
                    nxt_V = 0
                    probs = []
                    for atomic_n in self.node_layers[j+1]:
                        c = self.get_transition_context(atomic_n, state, a, j)
                        probs.append(p[c])
                    transition_prob = np.array(list(product(*probs)))    # need to filter the transition probs
                    transition_prob = np.prod(transition_prob, axis=1)   # compute the probability that each state occur
                    T = transition_prob[self.valid_states[j+1]]          # only consider valid states
                    #print(f"the total transition mass: {np.sum(T)} for probs {probs}")
                    qVals[si, j][ai] = mean_r + np.dot(T, qMax[j+1])     # compute the qValue
                # store the value of best action at state si and timestep j. 
                qMax[j][si] = np.max(qVals[si, j])
        return qVals, qMax

    def solve_layer_wise(self, P, R, outcomes):
        """
        Given the non-atomic transition and reward function use dynamic programming
        """
        qVals = {}
        qMax = {}
        qMax[self.H] = np.zeros(len(self.state_space[self.H]))

        for i in range(self.H-1):
            j = self.H - i - 1
            qMax[j] = np.zeros(len(self.state_space[j]))

            for si, state in enumerate(self.state_space[j]):
                qVals[si, j] = np.zeros(len(self.action_space[j]))

                for ai, a in enumerate(self.action_space[j]):
                    r = np.dot(R[state, a], outcomes)
                    T = P[state, a]
                    qmax = qMax[j+1]
                    qVals[si, j][ai] =  r + np.dot(T, qmax)
                # store the value of best action at state si and timestep j. 
                qMax[j][si] = np.max(qVals[si, j])
        return qVals, qMax

    def reset(self):
        # reset the flow
        flow = {}
        for node in self.G.nodes:
            if node == 1:
                f = {'flow': self.max}
            else:
                f = {'flow': 0}
            flow[node] = f
        nx.set_node_attributes(self.G, flow)
        # reset the timestep
        self.t = 1
        return self.t, (self.G.nodes[1]['flow'],)

