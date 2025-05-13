import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
from nmppi_global_random import do_mppi_ais_mpc, gen_and_save_results
from branch_mppi.systems import Unicycle
import jax
import jax.numpy as jnp
import copy
from branch_mppi.jax_mppi.grid import OccupGrid
from branch_mppi.systems import reachability
from branch_mppi.jax_mppi.plot_utils import animate_simulation_with_sampled_states
import hj_reachability as hj
# from hj_reachability import get_reachable_set_hjr, check_reachability_multiple_hjr
from nmppi_global_random import get_reachable_set_hjr, check_reachability_multiple_hjr

jax.config.update('jax_platform_name', 'cpu')

def rerun_experiments(params, foldername, counter, do_mpc=True, do_ais=True, do_contingency=True):
    params_copy = copy.deepcopy(params)
    params_copy['nlmodel'] =  Unicycle({"lb":np.array([-np.pi, -3.0]),
                        "ub":np.array([np.pi, 3.0])}, dt=params['dt'])
    params_copy['start'] = np.array(params_copy['start'])
    params_copy['q_ref'] = np.array(params_copy['q_ref'])
    params_copy['obs'] = np.array(params_copy['obs'])
    params_copy['safe_zones'] = np.array(params_copy['safe_zones'])
    params_copy['sigma0'] = np.array(params_copy['sigma0'])
    params_copy['Q'] = np.array(params_copy['Q'])
    params_copy['QT'] = np.array(params_copy['QT'])
    params_copy['R'] = np.array(params_copy['R'])

    if do_contingency:
        outputs = do_mppi_ais_mpc(params_copy, jax.random.PRNGKey(0), do_mpc=do_mpc, do_ais=do_ais)
        if do_ais:
            alg = "mpc_ais" 
        else:
            alg="mpc"
    else:
        params_copy["n_mini"] = 0
        params_copy["N_mini"] = 0
        outputs = do_mppi_ais_mpc(params_copy, jax.random.PRNGKey(0), do_mpc=do_mpc, do_ais=do_ais)
        alg="mppi"

    gen_and_save_results(params_copy, outputs, foldername, counter, alg=alg)
    plt.close('all')
    pass

def gen_reachable_set_hjr(params):
    reachable_sets = []
    max_control = jnp.array([jnp.pi, 3.0])
    min_control = jnp.array([-jnp.pi, -3.0])
    # breakpoint() 
    dynamics = hj.systems.DubinsCarObs(np.array(params['obs']),
                max_control = max_control,
                min_control = min_control 
                 )

    scale = np.array((50,50,50))
    box_r = params['N_mini']*params['dt']*max_control[1]*0.8 / 2
    wh = np.array([box_r, box_r])*2
    target_time = -1.0*params['N_mini']*params['dt']
    for sz in params['safe_zones']:
        target_values = get_reachable_set_hjr(np.array(sz[:2]), dynamics, scale, wh, target_time, solver_settings=hj.SolverSettings.with_accuracy("high"))
        reachable_sets.append(target_values)
    return reachable_sets

    
def check_sz_reachability_hj(data, params, reachable_sets):
    states = data['states']
    max_control = jnp.array([jnp.pi, 3.0])
    scale = np.array((50,50,50))
    box_r = params['N_mini']*params['dt']*max_control[1]*0.8 / 2
    reachable_count=[]
    for state in states:
        reachable = check_reachability_multiple_hjr(reachable_sets, params['safe_zones'], state[:2], scale[:2], np.array([box_r*2, box_r*2]))
        if reachable:
            reachable_count.append(1)
        else:
            reachable_count.append(0)
    reachable_count = np.array(reachable_count)
    return np.sum(reachable_count)/len(states), reachable_count

def regen_graphs(data, params):
    obs = np.array(params['obs'])
    start = np.array(params['start'])
    q_ref = np.array(params['q_ref'])
    N_mini = np.array(params['N_mini'])
    safe_zones = np.array(params['safe_zones'])
    states = data['states']
    total_sampled_states = data['total_sampled_states']
    number_safe_hist = data['number_safe_hist']
    total_con_states = data['total_con_states']
    trial_hz = data['trial_hz']
    safety_hist = data['safety_hist']
    costs = data['costs']
    timestep_reached = data['timestep_reached']
    global_us = data['global_us']
    time_taken = data['time_taken']
    percent_safe_hist = data['percent_safe_hist']
    
    all_con_states = []
    for i in range(len(states)):
        found = False
        for con_states in total_con_states[i]:
            collided = False
            reached = False
            for state in con_states:
                if np.any(np.linalg.norm(state[:2] - obs[:,:2], axis=1) < obs[:,2]):
                    collided =True
                    break
                if np.any(np.linalg.norm(state[:2] - safe_zones[:,:2], axis=1)) < 0.4:
                    reached=True
            
            if reached and not collided:
                found = True
                print(i)
                break
        all_con_states.append(con_states)
    
    for con_states in all_con_states:
        plt.plot(con_states[:,0], con_states[:,1], 'g')
    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        plt.gca().add_artist(rect)

    plt.show()
    breakpoint()

def regen_gifs(data, params, name):
    obs = np.array(params['obs'])
    start = np.array(params['start'])
    q_ref = np.array(params['q_ref'])
    N_mini = np.array(params['N_mini'])
    safe_zones = np.array(params['safe_zones'])
    states = data['states']
    total_sampled_states = data['total_sampled_states']
    number_safe_hist = data['number_safe_hist']
    total_con_states = data['total_con_states']
    trial_hz = data['trial_hz']
    safety_hist = data['safety_hist']
    costs = data['costs']
    timestep_reached = data['timestep_reached']
    global_us = data['global_us']
    time_taken = data['time_taken']
    percent_safe_hist = data['percent_safe_hist']
    
    all_con_states = []
    N_mini=15
    num_safe = safe_zones.shape[0]

    for i in range(len(states)):
        # found = False
        tmp_con_states = []
        for con_states in total_con_states[i]:
            collided = False
            reached = False
            num_found= 0
            # if i > 5:
            #     breakpoint()
            for state in con_states:
                if np.any(np.linalg.norm(state[:2] - obs[:,:2], axis=1) < np.sqrt(obs[:,2])):
                    collided =True
                    break
                # if np.any(np.linalg.norm(state[:2] - safe_zones[:,:2], axis=1)) < 0.4:
                #     reached=True
            kron_safe_zones =np.kron(np.ones((N_mini,1)), safe_zones)
            kron_states = np.kron(con_states, np.ones((num_safe,1)))
            if np.any(np.linalg.norm(kron_safe_zones-kron_states, axis=1)<=1.0):
                reached=True
            
            if reached and not collided:
                print(i)
                num_found+=1
                tmp_con_states.append(con_states)
                if num_found >5:
                    break
            if i == len(states)-1:
                tmp_con_states.append(np.repeat([state], N_mini, axis=0))
                
        all_con_states.append(np.array(tmp_con_states))

    anim = animate_simulation_with_sampled_states (states, obs, safe_zones, all_con_states)
    anim.save(name, writer='imagemagick')

    for con_states in all_con_states:
        # con_states = con_states[0]
        plt.plot(con_states[:,0], con_states[:,1], 'g')
    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        plt.gca().add_artist(rect)

    plt.show()

def stats(algs, base_name):
    results_dict = {}
    aggregate_dict = {}

    for alg in algs:
        results_dict[alg] = {
            # "states":[],
            # "total_sampled_states":[],
            # "number_safe_hist":[],
            # "total_con_states":[],
            # "trial_hz":[],
            "safety_hist":[],
            # "costs":[],
            "timestep_reached":[],
            # "global_us":[],
            # "time_taken":[],
            "percent_safe_hist":[],
            "time_taken":[],
            "mpc_times":[],
            "mppi_times":[],
            "topo_times":[],
                            }
        aggregate_dict[alg] = {
            'reached':None
                            }

    for i in range(0,1  ):
        foldername = base_name + str(i)
        paramsname = os.path.join(foldername, f"params_{i}")

        # if i <= 90:
        #     continue
        with open(paramsname, 'r') as f:
            params = json.load(f)
        # if file of paramsname+_"rs" does not exist, create it:
        if os.path.exists(paramsname+"_rs.npz"):
            reachable_safe_sets_npz= np.load(paramsname+"_rs.npz")
            reachable_safe_sets = reachable_safe_sets_npz.f.arr_0
        else:
            reachable_safe_sets= gen_reachable_set_hjr(params)
            np.savez_compressed(paramsname+"_rs", np.array(reachable_safe_sets)) 

        for alg in algs:
            filename = os.path.join(foldername, f"sim_results_{alg}_{i}.npz")
            results = np.load(filename)

                # regen_graphs(results,params)
                # breakpoint()
            # percent, safe_hist = check_reachability(results, params)
            # percent, safe_hist = check_sz_reachability_hj(results, params, reachable_safe_sets)
            percent, safe_hist = check_sz_reachability_hj(results, params, reachable_safe_sets)
            print(f"{alg}: {percent}")
            # breakpoint()

            # rerun_experiments(params, foldername, i, do_mpc=True, do_ais=True, do_contingency=False)
            for kwd in results_dict[alg].keys():
                # if kwd == "percent_safe_hist":
                #     results_dict[alg][kwd].append(percent)
                if kwd == "safety_hist":
                    results_dict[alg][kwd].append(safe_hist)
                else:
                    results_dict[alg][kwd].append(results[kwd])


    
    sampled_stats = []
    reached = np.array([results_dict[alg]['timestep_reached'] for alg in algs] )
    index_reached_all = []
    for i in range(reached.shape[1]):
        # if np.all(reached[0:2,i] != -1):
        # if np.all(reached[1:3,i] != -1):
        if np.all(reached[:,i] != -1):
            index_reached_all.append(i)
            
    # breakpoint()

    print("***************************************************************************************************************************")
    for alg in algs:
        ts_reached = np.array(results_dict[alg]['timestep_reached'])
        reached = ts_reached!=-1
        print(f"{alg} number unsolved solved problems: {len(np.nonzero(1-reached))}")
        problem_safe = [1 if np.all(x) else 0 for x in results_dict[alg]['safety_hist']]
        print(f"{alg} safe problems: {np.sum(problem_safe)/len(problem_safe)}")
        number_reached = np.sum(reached)
        percent_reached = number_reached/len(reached)
        print(f"{alg} percent_reached: {percent_reached}")
        aggregate_dict[alg]['reached'] = reached
        safe_hist = np.array(results_dict[alg]['safety_hist'])
        percent_safe = np.count_nonzero(safe_hist) / np.size(safe_hist) 
        print(f"{alg} safety percentage: {percent_safe}")

        sampled_safe = np.array(results_dict[alg]['percent_safe_hist'])
        sampled_stats.append(sampled_safe.flatten())
        avg_safe = np.mean(sampled_safe)
        print(f"{alg} average safe samples percent: {avg_safe}")
        print(f"{alg} average time reached: {np.mean(ts_reached[reached])}")
        if alg == "ais_mpc":
            mpc_times = [time for xs in results_dict[alg]["mpc_times"] for time in xs]
            mppi_times = [time for xs in results_dict[alg]["mppi_times"] for time in xs]
            topo_times = [time for xs in results_dict[alg]["topo_times"] for time in xs]
            print(f"{alg} average computation time: {np.mean(results_dict[alg]['time_taken'])}")
            print(f"{alg} average mpc time: {np.mean(mpc_times)}")
            print(f"{alg} average mppi time: {np.mean(mppi_times)}")
            print(f"{alg} average topo time: {np.mean(topo_times)}")
        # avg_time_reached = 0
        # if True:
        # # if alg == "ais_mpc" or alg =="mpc":
        # # if alg == "base" or alg =="mpc":
        #     for index in index_reached_all:
        #         time_reached = np.array(results_dict[alg]['timestep_reached'][index])
        #         avg_time_reached += time_reached
        #     avg_time_reached = avg_time_reached / len(index_reached_all)
        #     print(f"{alg} average time reached: {avg_time_reached}")


    fig, ax = plt.subplots() 
    ax.boxplot(sampled_stats)
    plt.show()
    breakpoint()

def main(args):
    # base_name = "results_set0/sim_results"
    # base_name = "results_set1/sim_results"
    # base_name = "results_set2/sim_results"
    algs = ["mppi", "ais_mpc", "mpc", "base", "mppi_heuristic_3", "mppi_heuristic_30", "mppi_heuristic_30_mpc"]
    # algs = ["ais_mpc", "mpc", "base"]
    # base_name = "results_set0/sim_results"
    base_name = "sim_results"
    if args.stats:
        stats(algs, base_name)
    # if args.regen_graphs:
    #     regen_gifs(results, params, outpath)
    # base_name = "results_set2/sim_results"
    # experiment_name = 37
    # foldername = base_name + str(experiment_name)
    # paramsname = os.path.join(foldername, f"params_{experiment_name}")
    # filename = os.path.join(foldername, f"sim_results_ais_mpc_{experiment_name}.npz")
    # outpath = f"./regen_{experiment_name}.gif"
    # results = np.load(filename)
    # with open(paramsname, 'r') as f:
    #     params = json.load(f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-mppi', action='store_true')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--gen-gif', action='store_true')
    args = parser.parse_args()
    main(args)