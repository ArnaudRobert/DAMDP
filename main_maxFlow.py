import numpy as np
import networkx as nx
from networkx import traversal as T
from itertools import product
from envs.maxFlowEnv import MaxFlowEnv
from envs.maxFlowEnvV2 import MaxFlowEnv as MFE
from agent import PSGRL, PSRL
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


def build_graph(G, nodes, edges):
    """
    Function that adds a set of nodes and edges
    to a graph.

    Inputs:
        - G: the graph
        - nodes : the set of nodes
        - edges : the set of edges
    """
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    for sender, receiver in edges:
        G.add_edge(sender, receiver)


def create_graphs():

    G1 = nx.DiGraph()
    nodes = {1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (3, 1), 6: (3, -1), 7: (4, 0)}
    edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)]
    build_graph(G1, nodes, edges)

    G2 = nx.DiGraph()
    nodes = {1: (0, 0), 2: (1, 0), 3: (2, 0)}
    edges = [(1, 2), (2, 3)]
    build_graph(G2, nodes, edges)

    G3 = nx.DiGraph()
    nodes = {1: (0, 0), 2: (1, 1), 3: (1, 0), 4: (1,-1), 5: (2, 0)}
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]
    build_graph(G3, nodes, edges)

    G4 = nx.DiGraph()
    nodes = {1: (0, 0), 2: (1, 1), 3: (1, 0), 4: (1,-1), 5: (2, 1), 6: (2, 0), 7: (2, -1), 8: (3, 1), 9: (3, 0), 10: (3,-1), 11: (4, 0)}
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5,8), (6,9), (7,10), (8, 11), (9, 11), (10, 11)]
    build_graph(G4, nodes, edges)

    G5 = nx.DiGraph()
    nodes = {1: (0, 0), 2: (1, 1), 3: (1, 0), 4: (1,-1), 5: (2, 1), 6: (2, 0), 7: (2, -1), 8: (3, 1), 9: (3, 0), 10: (3,-1), 11: (4, 1), 12: (4, 0), 13: (4, -1), 14: (5,0)}
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5,8), (6,9), (7,10), (8, 11), (9, 12), (10, 13), (11, 14), (12, 14), (13, 14)]
    build_graph(G5, nodes, edges)

    G6 = nx.DiGraph()
    nodes = {1: (0,0), 2: (1,1), 3: (1,-1), 4: (2,2), 5: (2,0), 6: (2,-2), 7: (3,1), 8: (3,-1), 9: (4,0)}
    edges = [(1,2), (1,3), (2,4), (2,5), (3,5), (3,6), (4,7), (5,7), (5,8), (6,8), (7,9), (8,9)]
    build_graph(G6, nodes, edges)
    
    G7 = nx.DiGraph()
    nodes = {1: (0,0), 2: (1,1), 3: (1,-1), 4: (2,2), 5: (2,0), 6: (2,-2), 7: (3,3), 8: (3,1), 9: (3,-1), 10: (3,-3), 11: (4,2), 12: (4,0), 13: (4,-2), 14: (5,1), 15:(5,-1), 16: (6,0)}
    edges = [(1,2), (1,3), (2,4), (2,5), (3,5), (3,6), (4,7), (4,8), (5,8), (5,9), (6,9), (6,10), (7, 11), (8, 11), (8, 12), (9, 12), (9, 13), (10,13), (11, 14), (12, 14), (12, 15), (13, 15), (14, 16), (15, 16)]
    build_graph(G7, nodes, edges)

    return [G6, G7]


def run_psgrl(graphs, seeds, N):
    # compute the regret
    rw_psgrl = np.zeros((len(graphs), N, len(seeds)))
    regret_psgrl = np.zeros((len(graphs), N, len(seeds)))
    for gIdx, G in enumerate(graphs):
        for s in seeds:
            np.random.seed(s)
            env = MFE(G, 2, 0.1)
            agent_g = PSGRL(env)
            _, Q_star = env.solve(env.p, env.r)
            for k in range(N):
                agent_g.update_policy()
                t, state = env.reset()
                done = False
                while not done:
                    a = agent_g.pick_action(state, t)
                    nxt_state, rewards, done, t, contexts = env.act(a)
                    state = tuple(nxt_state.values())
                    step_reward = np.sum(list(rewards.values()))
                    agent_g.update_obs(rewards, nxt_state, done, contexts)
                rw_psgrl[gIdx, k, s] = step_reward
                regret_psgrl[gIdx, k, s] = max(Q_star[1]) - max(agent_g.qMax[1])
            print(f"Run PSGRL experiment - completed {s}/{len(seeds)}")
            np.save(f"./computations/maxFlow_diamonds_psgrl_regret", regret_psgrl)


def run_psrl(graphs, seeds, N):
    rw_psrl = np.zeros((len(graphs), N, len(seeds)))
    regret_psrl = np.zeros((len(graphs), N, len(seeds)))
    for gIdx, G in enumerate(graphs):
        for s in seeds:
            np.random.seed(s)
            env = MFE(G, 2, 0.1)
            agent = PSRL(env)
            _, Q_star = env.solve(env.p, env.r)
            for k in range(N):
                agent.update_policy()
                t, state = env.reset()
                done = False
                while not done:
                    a = agent.pick_action(state, t)
                    action = env.action_space[t][a]
                    nxt_state, rewards, done, t, _ = env.act(a)
                    r = np.sum(list(rewards.values()))
                    nxt_state = tuple(nxt_state.values())
                    agent.update_obs(state, action, r, nxt_state, done, t)
                    state = nxt_state
                rw_psrl[gIdx, k, s] = r
                regret_psrl[gIdx, k, s] = max(Q_star[1]) - max(agent.qMax[1])
            print(f"Run PSRL experiment - completed {s}/{len(seeds)}")
            np.save(f"./computations/maxFlow_diamonds_psrl_regret", regret_psrl)


if __name__ == "__main__":
    # experiment parameters
    N = 1000                # number of episodes
    seeds = np.arange(5)    # the seeds used for the experiments. 
    graphs = create_graphs()

    run_psgrl(graphs, seeds, N)
    run_psrl(graphs, seeds, N)

