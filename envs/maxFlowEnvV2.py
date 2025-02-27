import numpy as np
import networkx as nx
from networkx import traversal as T
from itertools import product


class MaxFlowEnv():
    """
    the rules are the following:

    the goal is to transport as much flow from a source node to a sink node.

    - each edge has a capacity, if the flow exceed the capacity the amount
    transported is 0.
    - each edge has probability of failure p, if the edge fails the amount
    transported is 0.
    - lastly if the total amount of flow instructed to leave the node is larger
    than the amount of flow currently in the node - this result in an illegal
    action and the flow transported is also 0.

    Work for any DAG.
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
        self.layer_outcomes = self.outcomes

        self.p = {}
        self.r = {}

        # compute the true atomic reward function
        self.r_templates = self._get_reward_context_templates()
        self.Cs_r, self.Cs_r_templates_idx = self._get_contexts_from_template(self.r_templates)
        self._compute_atomic_r()

        # compute the true atomic transition function
        self.p_templates = self._get_transition_context_templates()
        Cs_p, tidxs = self._get_contexts_from_template(self.p_templates)
        self.Cs_p, self.p_tidxs = self._filter_p_contexts(Cs_p, self.p_templates, tidxs)
        self.p = self._compute_all_atomic_p()

        self.action_space, self.edge_layers = self._build_action_space()
        self.state_space, self.node_layers, self.valid_states = self._build_state_space()

    def _get_reward_context_templates(self):
        """
        Method that goes over the graph, and identify
        what are the context templates for the reward function.
        """
        all_templates = set()
        for node in self.G.nodes:
            tmp = ('s',)
            for edge in self.G.out_edges(node):
                tmp += ('a',)
            all_templates.add(tmp)
        all_templates.remove(('s',))
        return list(all_templates)

    def _get_transition_context_templates(self):
        """
        Get the transition context templates.
        """
        all_templates = set()
        for node in self.G.nodes:
            C = tuple()
            parents = list(self.G.predecessors(node))
            for pn in parents:
                out = self.G.out_edges(pn)
                C += ('s',)
                C += len(out) * ('a',)
            if len(C) > 0:
                all_templates.add(C)
        return list(all_templates)

    def _get_contexts_from_template(self, templates):
        """
        Given a templates retrieve a list of contexts.
        Input:
            - templates : a template for a context ('s', 'a', 'a')
                          for example stipulates that some context will
                          contain a state and two actions.
       Output:
            - all_context  : a list of all contexts
            - template_ids : for each context maps the template from which it
              comes from.
        """
        all_contexts = []
        templatesID = []
        for t, template in enumerate(templates):
            l = []
            for elem in template:
                if elem == 's':
                    l.append(self.atomic_states)
                else:
                    l.append(self.atomic_actions)
            Cs = list(product(*l))
            all_contexts.extend(Cs)
            templatesID.extend(np.ones(len(Cs), dtype=int) * t)
        return all_contexts, templatesID

    def _filter_p_contexts(self, Cs, templates, tidxs):
        """
        """
        filtered_Cs = []
        filtered_idxs = []
        for i, cxt in enumerate(Cs):
            tmp = templates[tidxs[i]]
            tot = 0
            for elem, type in zip(cxt, tmp):
                if type == 's':
                    tot += elem
            if tot > self.max:
                continue
            filtered_Cs.append(cxt)
            filtered_idxs.append(tidxs[i])
        return filtered_Cs, filtered_idxs

    def _compute_atomic_r(self):
        """
        Compute the the atomic reward function.
        """
        for cxt in self.Cs_r:
            s = cxt[0]
            acts = cxt[1:]
            tot = np.sum(acts)
            if tot > self.max:
                continue
            self.r[cxt] = np.zeros((len(self.outcomes),))
            if tot > s:
                self.r[cxt][0] = 1
                continue
            actions = []
            probs = []
            for a in acts:
                if a > self.c:
                    a = 0
                actions.append([0, a])
                probs.append([self.fail, (1-self.fail)])
            actions = list(product(*actions))
            probs = list(product(*probs))
            actions = [np.sum(elem) for elem in actions]
            probs = [np.prod(elem) for elem in probs]

            for a, pp in zip(actions, probs):
                self.r[cxt][a] += pp

    def _compute_all_atomic_p(self):
        """
        """
        atomic_p = {}
        for cxtId, cxt in enumerate(self.Cs_p):
            tId = self.p_tidxs[cxtId]
            temp = self.p_templates[tId]
            i = 0
            curr = tuple()
            R = []
            while i < len(cxt):
                if temp[i] == 's':
                    if len(curr) > 0:
                        R.append(curr)
                    curr = (cxt[i],)
                else:
                    curr += (cxt[i],)
                i += 1
            R.append(curr)
            atomic_p[cxt] = self._compute_atomic_p(R)
        return atomic_p

    def _compute_atomic_p(self, cxt):
        """
        Context in the following form: ((s, a, a), (s, a), (s, a, a, a)...)
        """
        # compute the transition for each element idependently 
        ps = np.zeros((len(self.atomic_states),))
        actions = []
        probs = []
        for i, elem in enumerate(cxt):
            # check overflow
            s = elem[0]
            a = elem[1]
            if s < np.sum(elem[1:]):
                a = 0
            if a > self.c:
                a = 0
            actions.append([0, a])
            probs.append([self.fail, 1-self.fail])

        actions = [np.sum(elem) for elem in list(product(*actions))]
        probs = [np.prod(elem) for elem in list(product(*probs))]

        for a, pp in zip(actions, probs):
            ps[a] += pp
        return ps

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
        Return the Q-values for each action and the value of each state.

        Input:
            - p : an atomic transition function
            - r : an atomic reward function

        Output:
            - qVals : qVals[state, timestep] Q-values of each action
            - qMax  : qMax[timestep] each state value
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
                    T = P[state, a, j]
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

