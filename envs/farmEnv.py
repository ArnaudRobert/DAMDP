############################
# Python import
############################
import numpy as np
import networkx as nx
from networkx import traversal as T
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
from itertools import product

############################
# Floris imports
############################
import floris.tools as wfct

###############################################
# Helper function for creating Wind Farm Graph
###############################################


def speed2velocity(wind_speed: float, wind_direction: float,
                   angle_units: str = "deg"):
    """
    convert scalar "speed" to vector "velocity" i.e. wind speed
    in x and y directions.
    Args:
        wind_speed     : scalar wind speed
        wind_direction : wind direction in degree or radian
        angle_units    : unit of wind direction angle one of ["deg", "rad"]
    Returns:
        _type_: _description_
    """
    if angle_units == "deg":
        wind_direction = np.deg2rad(wind_direction)

    ws_x, ws_y = -wind_speed * np.sin(wind_direction), -wind_speed * np.cos(wind_direction)
    ws_xy = np.column_stack([ws_x, ws_y])
    return ws_xy


def dir2azimuth(wind_direction, angle_unit="deg"):
    """
    Converts wind direction in FLORIS to a format that is measured like azimuth 
    (i.e. from north ~ 0 deg)
    Args:
        wind_direction (float): wind direction
        angle_unit (string): unit of wind direction angle one of ["deg", "rad"]
    """
    [ws_x, ws_y] = speed2velocity(10., wind_direction)[0]
    azimuth = np.arctan(ws_x/ws_y)
    if ws_y < 0:
        azimuth += np.pi
    return np.rad2deg(azimuth)


def wake2edge(turbine_coord, wind_direction: float, cone_angle: float,
              max_dist: float):
    """
    Args:
        turbine_coord (_type_): _description_
        wind_direction (float, array like): wind direction in degrees
        cone_angle (float, optional): _description_. Defaults to 15.
    Returns:
        _type_: _description_
    """
    num_turbines = turbine_coord.shape[0]
    azimuth = dir2azimuth(wind_direction)
    angle_rot = np.deg2rad(azimuth - 90)
    # clockwise rotation matrix
    rot_cw = np.array([[np.cos(angle_rot), np.sin(angle_rot)],
                       [-np.sin(angle_rot), np.cos(angle_rot)]])

    # pair-wise relative displacement of vertices/turbines
    rel_disp = turbine_coord.reshape([1, num_turbines, 2]) - turbine_coord.reshape([num_turbines, 1, 2])
    rel_disp = rel_disp @ rot_cw

    # compute relative distance
    dist = np.sqrt(np.sum(np.square(rel_disp), axis=2))

    # angle relative to wind direction
    theta = np.rad2deg(np.arctan(rel_disp[:, :, 1]/rel_disp[:, :, 0]))
    theta[np.isnan(theta)] = 90
    directed_edge_indices = ((rel_disp[:, :, 0] >= 0) &
                             (abs(theta) < cone_angle) &
                             (abs(dist) < max_dist)).nonzero()
    sender_nodes, receiver_nodes =  directed_edge_indices[0],  directed_edge_indices[1]
    edge_attr = rel_disp[sender_nodes, receiver_nodes, :]
    return edge_attr, sender_nodes, receiver_nodes


class GraphWindFarm():
    """
    Environment to control the yaw angle of wind turbine in a wind farm.

    We assume 3 discrete actions: either deflect in one of the two directions or face the wind.

    The external conditions (turbulence intensity, wind speed, and wind direction) remain the same during an episode.
    """

    def __init__(self, config: str, wind: dict = None):
        """
        Input:
            config  : Floris configuration file.
            seed    : Random seed that governs randomness in
                      episode's execution.
        """
        super(GraphWindFarm, self).__init__()
        if wind is None:
            self.wind = {'speed': 10, 'dir': 270}
        else:
            self.wind = wind

        self.fi = wfct.floris_interface.FlorisInterface(config)

        # build DAG-MDP
        self.G = self.layout2graph(self.wind['dir'], cone_angle=15)
        self.H = nx.dag_longest_path_length(self.G) + 1
        self.node_layers = {}
        self.edge_layers = {}

        if 'grid_3' in config:
            self.node_layers[1] = [0]
            self.node_layers[2] = [1]
            self.node_layers[3] = [2]

            self.edge_layers[1] = [(0, 1)]
            self.edge_layers[2] = [(1, 2)]
            self.edge_layers[3] = []
            N_max = 1 # the max number of nodes per layer

        elif 'grid_6' in config:
            self.node_layers[1] = [0, 1]
            self.node_layers[2] = [2, 3]
            self.node_layers[3] = [4, 5]

            self.edge_layers[1] = [(0, 2), (1, 3)]
            self.edge_layers[2] = [(2, 4), (3, 5)]
            self.edge_layers[3] = []
            N_max = 2 # the max number of nodes per layer

        elif 'grid_9' in config:
            self.node_layers[1] = [0, 1, 2]
            self.node_layers[2] = [3, 4, 5]
            self.node_layers[3] = [6, 7, 8]

            self.edge_layers[1] = [(0, 3), (1, 4), (2, 5)]
            self.edge_layers[2] = [(3, 6), (4, 7), (5, 8)]
            self.edge_layers[3] = []
            N_max = 3 # the max number of nodes per layer

        elif 'grid_12' in config:
            self.node_layers[1] = [0, 1, 2, 3]
            self.node_layers[2] = [4, 5, 6, 7]
            self.node_layers[3] = [8, 9, 10, 11]

            self.edge_layers[1] = [(0, 4), (1, 5), (2, 6), (3, 7)]
            self.edge_layers[2] = [(4, 8), (5, 9), (6, 10), (7, 11)]
            self.edge_layers[3] = []
            N_max = 4 # the max number of nodes per layer

        else:
            print("Error: unknown layout")

        # store the number of nodes 
        self.numT = len(self.G.nodes)
        self.norm = 3298068  # to normalize power generated by a single wind turbine

        self.outcomes = np.arange(0.1,1.1, step=0.1).tolist()
        self.layer_outcomes = [l/10 for l in list(range(N_max,N_max*10+1,1))]
        self.atomic_actions = [-30, 0, 30]
        self.atomic_states = np.linspace(6, 10, 5).tolist()
        self.action_space = {}
        self.state_space = {}

        # all possible atomic state - atomic action compinaton
        self.Cr = list(product(self.atomic_states, self.atomic_actions))
        # assume only one type of transition fct for grid like farm.
        self.Ct = list(product(self.atomic_states, self.atomic_actions))
        for layer, nodes in self.node_layers.items():
            self.action_space[layer] = list(product(self.atomic_actions, repeat=len(nodes)))
            self.state_space[layer] = list(product(self.atomic_states, repeat=len(nodes)))

        # not computed...
        self.r = {}
        self.p = {}
        for cxt in self.Cr:
            self.r[cxt] = None
        for cxt in self.Ct:
            self.p[cxt] = None
    
    def plot_graph_layout(self, ax):
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, with_labels=True,
                pos=pos, ax=ax)
        return ax

    
    def step(self, action):
        """
        Perform one step in the environment and return the new state and reward.

        Input:
            - action   : the action profile to perform (an atomic action per node)
        Output:
            - n_state  : a dictionary {nodeId: atomic state}.
            - reward   : a dictionary {nodeId: atomic reward}.
            - done     : boolean that evaluate to True if the episode ends.
            - t        : the current timestep
            - contexts : 
        """
        assert len(action) == len(self.curr_action[self.t]) # check the dimension of the action profile
        # set up place holders
        done = False
        reward = {}
        contexts = {}
        contexts['rewards'] = {}
        contexts['transitions'] = {}
        
        # update action profile with new action
        self.curr_action[self.t] = action
        a = self.get_floris_action()
        
        # simulate performance
        self.fi.reinitialize(wind_speeds=[self.wind['speed']],
                             wind_directions=[self.wind['dir']])
        self.fi.calculate_wake(yaw_angles=a)

        # collect the power generated
        powers = self.fi.get_turbine_powers()[0, 0, :]
        tpw = [powers[n]/self.norm for n in self.node_layers[self.t]]
        tpw = self.discretize_r(tpw)
        
        for node, r in zip(self.node_layers[self.t], tpw):
            reward[node] = r
            contexts['rewards'][node] = self.get_reward_context(node, self.state, action, self.t)
 
        if self.t >= self.H:
            done = True
            n_state = None
        else :
            n_state = {}
            # get the wind speed at the next layer
            wst = self.fi.turbine_average_velocities[0, 0, :]
            new_s = [wst[n] for n in self.node_layers[self.t+1]]
            new_s_discrete = self.discretize_state(new_s)
            for node, s in zip(self.node_layers[self.t+1], new_s_discrete):
                n_state[node] = s
                contexts['transitions'][node] = self.get_transition_context(node, self.state, action, self.t)
            self.t += 1
            self.state = new_s_discrete
        return n_state, reward, done, self.t, contexts

    def reset(self):
        self.t = 1  # current time step
        self.fi.reinitialize(wind_speeds=[self.wind['speed']],
                             wind_directions=[self.wind['dir']])
        a = np.zeros((1, 1, self.numT))
        self.fi.calculate_wake(a)
        wst = self.fi.turbine_average_velocities[0, 0, :]
        state = [wst[n] for n in self.node_layers[self.t]]
        
        # set the action template to default
        self.curr_action = {}
        for layer, nodes in self.node_layers.items():
            self.curr_action[layer] = np.zeros((len(nodes),))
        self.state = self.discretize_state(state)
        return self.t, self.state

    def get_floris_action(self):
        """
        Convert the action dictionary into an array
        """
        florisA = np.zeros((1, 1, self.numT))
        curr = 0
        for acts in self.curr_action.values():
            florisA[0, 0, curr:curr+len(acts)] = acts
            curr += len(acts)
        return florisA

    def discretize_state(self, obs):
        discrete_state = tuple()
        for elem in obs:
            discrete_state += (self.atomic_states[min(range(len(self.atomic_states)), key = lambda i: abs(self.atomic_states[i]-elem))], )
        return discrete_state

    def discretize_r(self, rs, sum=False):
        discrete_reward = tuple()
        for r in rs:
            discrete_reward += (self.outcomes[min(range(len(self.outcomes)), key = lambda i: abs(self.layer_outcomes[i]-r))], )
        return discrete_reward

    def layout2graph(self, wd: float, cone_angle: float = 15.0,
                     max_dist: float = 601.0):
        """_summary_

        Args:
            wd                : wind direction
            layout_x (_type_) : _description_
            layout_y (_type_) : _description_
            radius (_type_)   : _description_
        """
        x, y = self.fi.get_turbine_layout()
        # reshape layout
        coord = np.zeros((len(x), 2))
        coord[:, 0] = x
        coord[:, 1] = y
        _, sender_nodes, receiver_nodes = wake2edge(coord, wd, cone_angle,
                                                            max_dist)
        # create directed graph
        G = nx.DiGraph()
   
        # add nodes
        for i, pos in enumerate(coord):
            G.add_node(i, pos=pos)
        # add edges
        for n1, n2 in zip(sender_nodes, receiver_nodes):
            G.add_edge(n1, n2)
        return G

    def get_reward_context(self, node, state, action, t):
        """
        Given a specfic node return the reward context form the full state and full action representation. 

        Input:
            - node   : 
            - state  : 
            - action : 
            - t      : 
        Output:
            - cxt    : 
        """
        ni = self.node_layers[t].index(node)
        cxt = (state[ni], action[ni])
        return cxt
        
    def get_transition_context(self, node, state, action, t):
        """
        Given a node (in the next layer) and the full state and action descritpion
        return the transition context.
        
        Input:
            - node   : 
            - state  : 
            - action : 
            - t      : 
        Output:
            - cxt    : 
        """
        cxt = tuple()
        for edge in self.edge_layers[t]:
            if edge[1] == node:
                ni = self.node_layers[t].index(edge[0])
                cxt += (state[ni], action[ni])
        return cxt
        
    def solve_layer_wise(self, P, R, outcomes):
        """
        Given the non-atomic transition and reward function use dynamic programming

        qVals(state, timestep) -> actions
        """
        qVals = {}
        qMax = {}
        qMax[self.H] = np.zeros(len(self.state_space[self.H])) 
        for si, state in enumerate(self.state_space[self.H]):
            qVals[si, self.H] = np.zeros(len(self.action_space[self.H]))
            for ai, a in enumerate(self.action_space[self.H]):
                r = np.dot(R[state, a], outcomes)
                qVals[si, self.H][ai] = r
            qMax[self.H][si] = np.max(qVals[si, self.H])
        
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

    def solve(self, p, r):
        """
        Given the atomic transition and reward function use dynamic programming 
        to find the value function and the Q-values.
        Input:
            - p : the atomic transition
            - r : the atomic reward
        Output:
            - qMax  : qMax[timestep] each state value 
            - qVals : qVals[state, timestep] Q-values of each action
        """
        qVals = {}
        qMax = {}
        qMax[self.H] = np.zeros(len(self.state_space[self.H]))

        for si, state in enumerate(self.state_space[self.H]):
            qVals[si, self.H] = np.zeros(len(self.action_space[self.H]))
            for ai, action in enumerate(self.action_space[self.H]):
                reward = 0
                for atomic_n in self.node_layers[self.H]:
                    cxt = self.get_reward_context(atomic_n, state, action, self.H)
                    reward = np.dot(r[cxt], self.outcomes)
                    qVals[si, self.H][ai] += reward
                qMax[self.H][si] = np.max(qVals[si, self.H])
        
        for i in range(self.H-1):
            j = self.H - i - 1
            qMax[j]= np.zeros(len(self.state_space[j]))
            for si, state in enumerate(self.state_space[j]):
                qVals[si, j] = np.zeros(len(self.action_space[j]))
                for ai, action in enumerate(self.action_space[j]):
                    for atomic_n in self.node_layers[j]:
                        cxt = self.get_reward_context(atomic_n, state, action, j)                        
                        reward = np.dot(r[cxt], self.outcomes)
                    nxtV = 0
                    probs = []
                    for atomic_n in self.node_layers[j+1]:
                        c = self.get_transition_context(atomic_n, state, action, j)
                        probs.append(p[c])
                    transition_prob = np.array(list(product(*probs)))
                    T = np.prod(transition_prob, axis=1)
                    #T = transition_prob[self.valid_states[j+1]]
                    qVals[si, j][ai] = reward + np.dot(T, qMax[j+1])
                qMax[j][si] = np.max(qVals[si, j])
        return qVals, qMax

   
