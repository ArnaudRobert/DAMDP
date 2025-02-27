import numpy as np
from agent import PSRL, PSGRL
from envs.farmEnv import GraphWindFarm

# experiment parameters
Ns = [int(1e3), int(1e3), int(1e4)]  # the number of episodes for each layout 
save = True
configs = ['grid_6_wt', 'grid_9_wt', 'grid_12_wt']
Ns = [int(1e3)]
configs = ['grid_3_wt']
seeds = np.arange(10)


for c, name in enumerate(configs):
    config = f"./configs/{name}.yaml"
    reward_psgrl = np.zeros((Ns[c], len(seeds)))
    reward_psrl = np.zeros((Ns[c], len(seeds)))

    # run PSGRL    
    for seed in seeds:
        env = GraphWindFarm(config)
        agent = PSGRL(env)
        for k in range(Ns[c]):
            agent.update_policy()
            t, state = env.reset()
            done = False
            ep_r = 0
            while not done:
                a = agent.pick_action(state, t)
                action = env.action_space[t][a]
                nxt_state, rs, done, t, contexts = env.step(action)
                agent.update_obs(rs, nxt_state, done, contexts)
                state = env.state
                ep_r += np.sum(list(rs.values()))
            reward_psgrl[k, seed] = ep_r
            if k % 10 == 0 and k > 0:
                print(f"episode {k}/{Ns[c]} - psrl max {np.max(reward_psrl)} - psgrl max {np.max(reward_psgrl)}")
        print(f"seed {seed+1}/{len(seeds)}")
        if save:
            np.save(f"./computations/windFarm_{name}_psgrl_reward", reward_psgrl)
        # to avoid segmentation fault
        del env
        del agent

    # run PSRL
    for seed in seeds:
        env = GraphWindFarm(config)
        agent = PSRL(env)
        for k in range(Ns[c]):
            agent.update_policy()
            t, state = env.reset()
            done = False
            ep_r = 0
            while not done:
                a = agent.pick_action(state, t)
                action = env.action_space[t][a]
                nxt_state, rs, done, t, _ = env.step(action)
                r = 0
                for v in rs.values():
                    r+=int(10*v)
                r /= 10
                if not nxt_state is None:
                    n_state = tuple(nxt_state.values())
                agent.update_obs(state, action, r, n_state, done, t, done)
                state = n_state
                ep_r += r
            reward_psrl[k, seed] = ep_r
            if k % 10 == 0 and k > 0:
                print(f"episode {k}/{Ns[c]} - psrl max {np.max(reward_psrl)} - psgrl max {np.max(reward_psgrl)}")
        print(f"seed {seed+1}/{len(seeds)}")
        if save:
            np.save(f"./computations/windFarm_{name}_psrl_reward", reward_psrl)
        # to avoid segmentation fault
        del env
        del agent

