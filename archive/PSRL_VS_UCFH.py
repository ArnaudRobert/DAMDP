from tabular_algorithms import UCFH, PSRL
from riverSwim import RiverSwimEnv
import numpy as np
import time

def run_agent(run_name, alg_id, nState, nAction, H):
    stats = np.zeros((N, 2))
    env = RiverSwimEnv(nS=nState, H=H)
    if alg_id == 0:
        agent = UCFH(nState=nState, nAction=nAction, epLen=H)
        print("ucfh")
    else:
        agent = PSRL(nState=nState, nAction=nAction, epLen=H)

    start_time = time.time()
    for episode in range(N):
        state, step = env.reset()
        agent.update_policy()
        acc_r = 0
        done = False
        while not done:
            action = agent.pick_action(state, step)
            new_s, rew, done, new_step, _ = env.step(action)
            agent.update_obs(state, action, rew, new_s, done, step)
            state = new_s
            step = new_step
            acc_r += rew
        stats[episode, 0] = acc_r
        stats[episode, 1] = np.abs(start_time - time.time())*10**3 # in ms
    np.save("./results/"+run_name, stats)


if __name__ == "__main__":

    ######################
    # set up the experiment
    ######################
    n_state = 6
    H = n_state * 10
    n_action = 2
    N = 3000
    algorithm_id = 0
    algos = ['ucfh', 'psrl']

    name = f"{algos[algorithm_id]}_riverSwim_{n_state}"
    run_agent(name, algorithm_id, n_state, n_action, H)
