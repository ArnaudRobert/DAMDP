import matplotlib.pyplot as plt
import networkx as nx
from networkx import traversal as T
import numpy as np
from itertools import product


class GraphFlow:
    """
    A class that maintains the graph representation used in GraphBandit
    and GraphMDP and provide the necessary functionalities.
    """
    def __init__(self, G, C, P, F=3):
        """
        Input:
            - G : a directed acyclic graph.
            - C : the edges' capacities dictionary {edge: c}
            - P : the edges' probability of failling {edge: p}
            - F : the inital flow in source node
        """
        self.G = G
        self.set_edge_capacities_and_probabilities(C, P)

        # store attributes
        self.N = self.G.nodes                      # list of nodes
        self.E = self.G.edges                      # list of edges
        self.source = 1                            # by convention
        self.sink = len(self.N)                    # by convention
        self.F = F                                 # initial flow
        self.Y = [int(f) for f in np.arange(F+1)]  # the atomic action space

        # build necessary spaces
        self.A = self._build_bandit_action_space()
        self.bandit_outcomes = [int(r) for r in np.arange(self.F+1)]
        print(f"source is {self.source} and sink is {self.sink}")

        self.layers = {}
        for t, layer in enumerate(T.bfs_layers(self.G, 1)):
            self.layers[t] = layer

        self.layers_edges = {}
        for t, layer in enumerate(T.bfs_layers(self.G, 1)):
            self.layers_edges[t] = []
            for node in layer:
                for (_, nxt_nodes) in T.bfs_successors(self.G, node, depth_limit=1):
                    for nxt_node in nxt_nodes:
                        self.layers_edges[t].append((node, nxt_node))

        print(self.layers_edges)

    def bandit_feedback(self):
        """
        Output:
            - r : the reward generated
        """
        for layer in T.bfs_layers(self.G, 1):
            for node in layer:
                for (_, nxt_nodes) in T.bfs_successors(self.G, node, depth_limit=1):
                    for nxt_node in nxt_nodes:
                        edge = self.G.edges[(node, nxt_node)]
                        flow = min(edge['action'], self.G.nodes[node]['flow'])
                        self.G.nodes[node]['flow'] -= flow
                        fail_prob = np.random.uniform()
                        if flow > edge['capacity'] or edge['p'] > fail_prob:
                            flow = 0
                        self.G.nodes[nxt_node]['flow'] += flow
        return self.G.nodes[self.sink]['flow']

    def advance(self, t):
        """
        Input:
            - t : the current timestep/layer

        Output:
            - r : the reward generated
        """
        active_nodes = self.layers[t]
        rewards = {}
        state = {}
        for node in active_nodes:
            for (_, nxt_nodes) in T.bfs_successors(self.G, node, depth_limit=1):
                for nxt_node in nxt_nodes:
                    edge = self.G.edges[(node, nxt_node)]
                    flow = min(edge['action'], self.G.nodes[node]['flow'])
                    self.G.nodes[node]['flow'] -= flow
                    fail_prob = np.random.uniform()
                    if flow > edge['capacity'] or edge['p'] > fail_prob:
                        flow = 0
                    self.G.nodes[nxt_node]['flow'] += flow
                    rewards[(node, nxt_node)] = flow
                    state[nxt_node] = self.G.nodes[nxt_node]['flow']
        return rewards, state

    def get_state(self, t):
        """
        Output:
            s : the state dictionary at the t^th time step {node id: flow}
        """
        pass

    def set_action(self, a):
        """
        Add the action to the edge attributes to allow.

        Input:
            a : action dictionary {edge: action}
        """
        action = {}
        for e, a in a.items():
            action[e] = {'action': int(a)}
        nx.set_edge_attributes(self.G, action)


    def set_edge_capacities_and_probabilities(self, C, P):
        """
        Input:
            - C : capacities dictionary {edge: capacity}
            - P : failing probability dictionary {edge: prob}
        """
        attr = {}
        for edge, cap in C.items():
            attr[edge] = {'capacity': int(cap), 'p': P[edge]}
        nx.set_edge_attributes(self.G, attr)

    def reset_flow(self):
        """
        Set the flow of the graph to be 0 everywhere
        except in the source node where it should be F.
        """
        flow = {}
        for node in self.N:
            if node == self.source:
                flow[self.source] = {'flow': self.F}
            else:
                flow[node] = {'flow': 0}
        nx.set_node_attributes(self.G, flow)

    def reset_actions(self):
        """
        Set to none all actions
        """
        action = {}
        for edge in self.E:
            action[edge] = {'action': None}
        nx.set_edge_attributes(self.G, action)

    def _build_bandit_action_space(self, filter=False):
        """
        Build the combinatorial action space for the bandit problem.
        Input:
            - filter : Specify if illegal actions are filtered (i. e. actions that request more flow than available)
        """
        all_actions = list(product(self.Y, repeat=len(self.E)))
        A = []
        for a in all_actions:
            action = {e:act for e,act in zip(self.E, a)}
            if self._is_action_valid(action) or not filter:
                A.append(action)
        return A

    def _build_layered_action_space(self):
        """
        """
        pass

    def _is_action_valid(self, a):
        """
        Check that an action is valid.

        Input:
            - a: action dictionary {edge: action}
        Output:
            - valid : a boolean idication if the action is valid
        """
        valid = True
        # prepare the network
        self.reset_flow()
        self.set_action(a)
        # simulate "deterministically" the action
        for layer in T.bfs_layers(self.G, 1):
            for node in layer:
                curr_flow = self.G.nodes[node]['flow']
                out_flow = 0
                for (_, nxt_nodes) in T.bfs_successors(self.G, node, depth_limit=1):
                    for nxt_node in nxt_nodes:
                        edge = self.G.edges[(node, nxt_node)]
                        tmp_flow = edge['action']
                        out_flow += tmp_flow
                        self.G.nodes[nxt_node]['flow'] += tmp_flow

                    if out_flow > curr_flow:
                        valid = False

        return valid
        
    def _get_CT(self, node):
        """
        Return the transition context.
        Input:
            - node: the node fro which we want to obtain the context
        Output:
            - C_T: (nodes, edges), a list of node ids and edges
        """
        nodes = list(self.G.predecessors(node))
        edges = []
        for n in nodes:
            edges.extend(self.G.out_edges(n))
        print(edges)
        return (nodes, edges)
        
    def _get_CR(self, node):
        """
        Return the transition context.
        Input:
            - node: the node fro which we want to obtain the context
        Output:
            - C_r: the edges exiting the node.
        """
        edges = list(self.G.out_edges(node))
        return edges

class GraphBandit:
    """
    Given a graph and its context determine the reward.
    """
    def __init__(self, G, C, P, F=3):
        """
        Input:
            - G : a directed acyclic graph.
            - C : the edges' capacities dictionary {edge: c}
            - F : the inital flow in source node
        """
        self.GF = GraphFlow(G, C, P,F)
        self.idx2act = {int(i):a for i, a in enumerate(self.GF.A)}

    def act(self, k):
        """
        Evaluate the action, return the corresponding reward.
        Note that actions are referred by their index.
        Input:
            - k : the index of the action
        """
        action = self.idx2act[k]
        self.GF.set_action(action)
        reward = self.GF.bandit_feedback()
        return reward

    def reset(self):
        """
        Reset the flow in the network and set the new context.
        Input:
            context : if specified determined the context for
                      the new graph.
        """
        self.GF.reset_flow()
        self.GF.reset_actions()

class GraphMDP:
    """
    Environment that cast the max flow problem as a graph MDP.
    """
    def __init__(self, G, C, P, F=3):
        """
        Input:
            - G : the graph encoding the problem.
            - C : the optional edge capacities if None randomly selectd.
            - P : the probability of each edge failling.
            - F : the initial amount of flow in the source node.
        """
        self.GF = GraphFlow(G, C, P, F)

        self.max_flow = F
        self.atomic_state = np.arange(0, F+1)
        self.outcomes = np.arange(0, F+1)
        self.atomic_action = np.arange(0, F+1)
        self.capacities = np.arange(0, F+1)

        self.N = len(self.GF.G.nodes)
        self.E = len(self.GF.G.edges)
        self.H = nx.dag_longest_path_length(self.GF.G)
        self.t = 0
        self.P = {}
        self.R = {}
        
        
        self.p_a = {}  # the atomic transition 
        self.r_a = {}  # the atomic reward

        for node in self.GF.G.nodes:
            pred = list(self.GF.G.predecessors(node))
            if len(pred) == 0:
                continue
            elif len(pred) == 1:  # single parent node
                for a in self.atomic_action:
                    for x in self.atomic_state:
                        nxt_x = np.zeros(len(self.atomic_state))
                        p = self.GF.G.edges[(pred[0], node)]['p']
                        c = self.GF.G.edges[(pred[0], node)]['capacity']
                        nxt_x[0] = p
                        flow = min(x, a)
                        if flow > c:
                            flow = 0
                        nxt_x[flow] += 1 - p
                        self.p_a[a,x] = nxt_x   

            elif len(pred) == 2:  # two parent nodes. 
                for a1 in self.atomic_action:
                    for a2 in self.atomic_action:
                        for x1 in self.atomic_state:
                            for x2 in self.atomic_state:
                                if a1 + a2 > self.max_flow:
                                    continue
                                nxt_x = np.zeros(len(self.atomic_state))
                                p1 = self.GF.G.edges[(pred[0], node)]['p']
                                p2 = self.GF.G.edges[(pred[1], node)]['p']
                                c1 = self.GF.G.edges[(pred[0], node)]['capacity']
                                c2 = self.GF.G.edges[(pred[1], node)]['capacity']

                                nxt_x[0] = p1*p2                                # both edges fails 
                                flow1 = min(x1, a1)
                                if flow1 > c1:
                                    flow1 = 0
                                flow2 = min(x2, a2)
                                if flow2 > c2:
                                    flow2 = 0
                                flow = flow1 + flow2
                                nxt_x[flow] += (1 - p1)*(1 - p2)                # both edges succeed
                                nxt_x[flow1] += (1 - p1) * p2                   # edge 2 fails edge 1 succeed
                                nxt_x[flow2] += p1 * (1 - p2)                   # edge 1 fails edge 2 succeed
                                self.p_a[a1, x1, c1, a2, x2, c2] = nxt_x

            else:
                print(f"Error: current implementation does not support connectivity pattern with {len(pred)} incoming edges")
                
        # the reward function
        for node in self.GF.G.nodes:
            succ = list(self.GF.G.successors(node))
            if len(succ) == 0:
                continue
            elif len(succ) == 1:
                for a in self.atomic_action:
                    for x in self.atomic_state:
                        for c in self.capacities:
                            r = np.zeros(len(self.outcomes))
                            p = self.GF.G.edges[node, succ[0]]['p']
                            r[0] += p
                            flow = min(x, a)
                            if flow > c:
                                flow = 0
                            r[flow] += 1 - p
                            self.r_a[a, x, c] = r

            elif len(succ) == 2:
                for a1 in self.atomic_action:
                    for a2 in self.atomic_action:
                        for x1 in self.atomic_state:
                            for x2 in self.atomic_state:
                                for c1 in self.capacities:
                                    for c2 in self.capacities:
                                        if a1 + a2 > self.max_flow:
                                            continue
                                        r = np.zeros(len(self.outcomes))                                
                                        p1 = self.GF.G.edges[node, succ[0]]['p']
                                        p2 = self.GF.G.edges[node, succ[1]]['p']
                                        r[0] = p1 * p2  # both edges fail
                                        flow1 = min(x1, a1)
                                        if flow1 > c1:
                                            flow1 = 0
                                        flow2 = min
            else:
                print(f"Error: current implementation does not support connectivity pattern")

        
        # build the time dependent state spaces
        self.state_space = {}
        for t, layer in self.GF.layers.items():
            tmp = list(product(self.atomic_action, repeat=len(layer)))
            tt = len(tmp)
            all_states = []
            for s in tmp:
                if np.sum(s) <= self.max_flow:
                    all_states.append(s)
            self.state_space[t] = all_states

        # build the time dependent action spaces
        self.action_space = {}  # {t: {k: (a1, a2, ...) } }
        for t, layer in self.GF.layers.items():
            tmp = list(product(self.GF.Y, repeat=len(self.GF.layers_edges[t])))
            self.action_space[t] = {i: a for i, a in enumerate(tmp)}

    def reset(self):
        """
        Reset the MDP, more specifically, reset the current
        layer / timestep to 0 and set all the flow in the
        unique source node.
        """
        self.t = 0
        self.GF.reset_flow()
        self.GF.reset_actions()

    def advance(self, action):
        """
        Move one step in the environment,
        hence, compute reward and next state.
        Input:
            - action : the action for each edge in the current layer.
                       dict {edge: action}
        """
        #print(self.action_space)
        action_dict = {edge: a for edge, a in zip(self.GF.layers_edges[self.t], self.action_space[self.t][action])}
        print(action_dict)
        self.GF.set_action(action_dict)
        rewards, nxt_state = self.GF.advance(self.t)
        self.t += 1
        return rewards, nxt_state

    def solve(self):
        """
        Solve a GraphMDP
        Outputs:
            - qVals : qVals[state, layer] is a vector of Q values for each action.
            - qMax  : qMax[layer] is the vector of optimal values at layer.
        """
        qVals = {}
        qMax = {}

        t = self.H
        qMax[t] = np.zeros(len())
        nodes = self.layers[t]

