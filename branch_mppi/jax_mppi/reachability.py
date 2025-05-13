import jax
import jax.numpy as jnp
import numpy as np

def get_reachability(system, environment, start, goals, ns, rng_key, K_mini, N_mini):
    '''
        Get Reachability thorugh Uniform Sampling on Control bounds
        K_mini: Safety Sampling number
        N_mini: Safety Horizion timesteps
    '''
    def mini_cost(carry, params):
        dcbf_alpha=0.01
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
    
        barrier_value = jax.vmap(system.h_x, in_axes=(None,0))(sim_state, environment)
        next_barrier_value =  jax.vmap(system.h_x, in_axes=(None,0))(new_state, environment)
        cbf_cost = -next_barrier_value + dcbf_alpha * barrier_value
        cbf_cost = jnp.max(cbf_cost) <= 0
        norm_dist = jnp.linalg.norm(sim_state[0:2] - goals[:,0:2], axis=1)
        safe = jnp.min(norm_dist)
        return (new_state), (sim_state, safe, cbf_cost)
    
    def cond_fun(carry):
        i, safe, u_seq = carry 
        break_condition = (safe<0.5)
        return ~break_condition & (i < N_mini)
    def body_fun(carry):
        i, safe, u_seq = carry

    def mini_rollout(state, rng_key):
        # Given a state and rollout and determines if rollout is safe
        u_seq = jax.random.uniform(rng_key, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
        term_state, state_and_safe = jax.lax.scan(mini_cost, state, u_seq)
        state_hist = state_and_safe[0]
        safe_hist = state_and_safe[1]
        cbf_hist = state_and_safe[2]
        safe = jnp.any(safe_hist<0.5)
        cbf_safe = jnp.all(cbf_hist)
        return (jnp.logical_and(safe,cbf_safe), state_hist)
    rng_keys = jax.random.split(rng_key, K_mini)
    safe_controls, state_hist = jax.vmap(mini_rollout, in_axes=(None, 0))(start.copy(), rng_keys)

    safe_ind = jnp.array(jnp.nonzero(safe_controls, size=1))
    safe_state_seq = jnp.take(state_hist, safe_ind, axis=0) 
    return jnp.any(safe_controls), safe_state_seq

def get_reachability_mppi(system, environment, start, goals, ns, rng_key, K_mini, N_mini, sigma, U_init, temperature=1.0, ais=3):
    m_elite = int((K_mini//ais)*0.25)
    p = 10
    def mini_cost(carry, params):
        dcbf_alpha=0.01
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
    
        barrier_value = jax.vmap(system.h_x, in_axes=(None,0))(sim_state, environment)
        next_barrier_value =  jax.vmap(system.h_x, in_axes=(None,0))(new_state, environment)
        cbf_cost = jnp.max(-next_barrier_value + dcbf_alpha * barrier_value) # max violation of obstacle avoidance cbf cost
        norm_dist = jnp.linalg.norm(sim_state[0:2] - goals[:,0:2], axis=1)
        sz_cost = jnp.min(norm_dist)

        return (new_state), (sim_state, sz_cost, cbf_cost)

    def eval_U_seq(u_seq):
        terminal_state, (state_seq, sz_cost_seq, cbf_cost_seq) = jax.lax.scan(mini_cost, start, u_seq)
        cost = jax.lax.cond(jnp.max(cbf_cost_seq) <= 0, lambda x: x, lambda x: np.inf + x, sz_cost_seq)
        return (cost, state_seq)

    def single_u_seq(rng_subkey, U, sigma):
        # noise_scaled = jax.random.multivariate_normal(rng_subkey, jnp.zeros(N_mini*2),sigma,  method='svd')
        noise_scaled = jax.random.normal(rng_subkey, shape=(N_mini*2,)) * jnp.diagonal(sigma)
        u_seq = (U + noise_scaled).reshape((-1,2))
        u_seq = jnp.clip(u_seq, system.control_bounds[0], system.control_bounds[1])
        # breakpoint()
        return u_seq 

    def calculate_new_means(costs, seq, original_seq):
        # lowest_ind = jnp.nanargmin(costs)
        exp_cost = jnp.exp(temperature*(jnp.nanmin(costs)-costs))
        denom =  jnp.nansum(exp_cost) + 1e-7
        best_u =  original_seq + jnp.nansum(exp_cost[..., None, None] * (seq-original_seq), axis=0) / denom
        return best_u

    def do_mppi(carry, xs):
        iter, rng_keys, U_mean, sigma, costs, state_seqs = carry
        u_mean = U_mean.reshape(((N_mini, 2)))
        u_seqs = jax.vmap(single_u_seq, (0, None, None))(rng_keys,U_mean, sigma)
        costs, state_seqs = jax.vmap(eval_U_seq, (0))(u_seqs)
        best_u = calculate_new_means(jnp.sum(costs, axis=1), u_seqs, u_mean)
        return (iter+1, rng_keys+1, best_u.reshape((N_mini*2,)), sigma, costs, state_seqs), xs
    
    def do_AIS(costs, u_seqs):
        ordered_costs = jnp.argsort(costs)
        elite =  u_seqs[ordered_costs[:m_elite], :]
        u_prime = jnp.mean(elite, axis=0)
        # u_prime = jnp.mean(cost)
        # breakpoint()
        sigma_prime = (jnp.cov(elite, rowvar=False) + jnp.eye(N_mini*2) * 10e-9)
        # sigma = jax.vmap(cov, (1, 0, None))(elite, u_prime, K_mini//ais)
        # sigma = jax.vmap(jnp.var, (1))(elite)
        # test = lambda carry, y: (carry, jnp.var(y))
        # _, sigma = jax.lax.scan(test, 0, elite.T)
        # sigma_prime = jnp.diag(sigma)
        # sigma_prime = sigma
        sigma_j  = sigma_prime.copy()
        sigma_j = sigma
        # breakpoint()
        # F = jnp.sum(sigma_prime) / p  * jnp.eye(N_mini*2)
        # F = jnp.trace(sigma_prime) / p  * jnp.eye(N_mini*2)
        # tr_ss = jnp.trace(sigma_j * sigma_prime)
        # rho = ((1-2.0/p) * tr_ss + jnp.trace(sigma_j)**2)/((m_elite + 1 - 2/p)*tr_ss+(1-m_elite/p)*jnp.trace(sigma_j)**2)
        # # jax.lax.while_loop(lambda x: x[1] - x[2] > 0.01, oas, (sigma_j, rho, 0.0,sigma_prime, F))
        # sigma_j, _, _, _, _=jax.lax.fori_loop(0,10,oas,(sigma_j, rho, 0.0,sigma_prime, F))
        return u_prime, sigma_j, ordered_costs

    def oas(i, carry):
        # carry, params):
        # u, noise_scaled, sigmai = params
        sigma_j, rho_j, rho_j_1, sigma_prime, F = carry
        rho_j_1 = rho_j
        tr_ss = jnp.trace(sigma_j * sigma_prime)
        rho_j = ((1-2.0/p) * tr_ss + jnp.trace(sigma_j)**2)/((m_elite + 1 - 2/p)*tr_ss+(1-m_elite/p)*jnp.trace(sigma_j)**2)
        sigma_j = (1-rho_j)*sigma_j+rho_j*F

        return (sigma_j, rho_j, rho_j_1, sigma_prime, F)


    def cond_fun(carry):
        iter, rng_keys, U_mean, sigma, costs, state_seqs = carry
        min_sz_distance = jnp.min(costs, axis=0)
        
        return (~(jnp.min(costs) <= 0.5)) & (iter <ais)

    rng_keys = jax.random.split(rng_key, K_mini//ais)
  
    min_costs = 10000
    u_mean = U_init.reshape(((N_mini, 2)))
    U_i  = U_init
    sigma_i = sigma
    
    # state_seqs_all = jnp.zeros((ais, K_mini//ais, N_mini, 3))
    state_seqs_all = jnp.zeros((ais, m_elite, N_mini, 3))
    state_seqs_means = jnp.zeros((ais, N_mini, 3))
    # for i in range(ais):
    #     (_, rng_keys, U_init, sigma, costs, state_seqs),_ = do_mppi((0, rng_keys, U_init, sigma, jnp.ones((K_mini//ais, N_mini,)), jnp.ones((K_mini//ais, N_mini, 3))), None)
    #     min_costs = jnp.minimum(min_costs, jnp.min(costs))
    #     state_seqs_all = state_seqs_all.at[i, :,:,:].set(state_seqs)
    #     state_seqs_means = state_seqs_means.at[i,:,:].set(state_rollouts(system, start, U_init.reshape((N_mini, 2))))

    for i in range(ais):
        u_seqs = jax.vmap(single_u_seq, (0, None, None))(rng_keys,U_i, sigma_i)
        costs, state_seqs = jax.vmap(eval_U_seq, (0))(u_seqs)
        min_costs = jnp.minimum(min_costs, jnp.min(costs))
        rollout_cost = jnp.sum(costs, axis=1)
        # rollout_cost = jnp.min(costs, axis=1)
        U_i, sigma_i, ordered_costs = do_AIS(rollout_cost, u_seqs.reshape((-1,N_mini*2)))
        u_i = U_i.reshape((N_mini,2))
        u_i = calculate_new_means(rollout_cost,  u_seqs, U_i.reshape((N_mini, 2)))
        U_i = u_i.reshape(N_mini*2,)
        # state_seqs_all = state_seqs_all.at[i, :,:,:].set(state_seqs)
        # state_seqs_all = state_seqs_all.at[i, :,:,:].set(state_seqs[ordered_costs[:m_elite], :])
        state_seqs_means = state_seqs_means.at[i,:,:].set(state_rollouts(system, start, u_i))

    # best_u = calculate_new_means(jnp.sum(costs, axis=1), u_seqs, u_mean)
    # costs, state_seqs = jax.vmap(eval_U_seq, (0))(best_u)
    # best_u = calculate_new_means(jnp.min(costs, axis=1), u_seqs, u_mean)

    safe_ind = jnp.nanargmin(costs[:1])
    safe_state_seq = jnp.take(state_seqs, safe_ind, axis=0)     
    # breakpoint() 
    return min_costs, safe_state_seq, state_seqs_all, state_seqs_means

def state_rollouts(system, start, u_seq):
    '''
        Get Reachability thorugh Uniform Sampling on Control bounds
        K_mini: Safety Sampling number
        N_mini: Safety Horizion timesteps
    '''
    def mini_cost(carry, params):
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
        return (new_state), (sim_state)

    term_state, state_hist = jax.lax.scan(mini_cost, start, u_seq)
    return state_hist

def get_reachability_general(system, environment, start, rng_key, K_mini, N_mini):
    '''
        Get Reachability thorugh Uniform Sampling on Control bounds
        K_mini: Safety Sampling number
        N_mini: Safety Horizion timesteps
    '''
    def mini_cost(carry, params):
        dcbf_alpha=0.01
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
        barrier_value = jax.vmap(system.h_x, in_axes=(None,0))(sim_state, environment)
        next_barrier_value =  jax.vmap(system.h_x, in_axes=(None,0))(new_state, environment)
        cbf_cost = -next_barrier_value + dcbf_alpha * barrier_value
        cbf_cost = jnp.max(cbf_cost) <= 0
        return (new_state), (sim_state, cbf_cost)
    
    def cond_fun(carry):
        i, safe, u_seq = carry 
        break_condition = (safe<0.5)
        return ~break_condition & (i < N_mini)
    def body_fun(carry):
        i, safe, u_seq = carry


    def mini_rollout(state, rng_key):
        # Given a state and rollout and determines if rollout is safe
        u_seq = jax.random.uniform(rng_key, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
        term_state, state_and_safe = jax.lax.scan(mini_cost, state, u_seq)
        state_hist = state_and_safe[0]
        cbf_hist = state_and_safe[1]
        cbf_safe = jnp.all(cbf_hist)
        return (cbf_safe, state_hist)

    rng_keys = jax.random.split(rng_key, K_mini)
    safe_controls, state_hist = jax.vmap(mini_rollout, in_axes=(None, 0))(start.copy(), rng_keys)

    # safe_ind = jnp.array(jnp.nonzero(safe_controls, size=K))
    # safe_state_seq = jnp.take(state_hist, safe_ind, axis=0) 
    return safe_controls, state_hist

# @functools.partial(jax.jit, static_argnames=('occup_value'))
def get_reachability_costmap(system, cost_map, origin, resolution, wh, tolerance, start, goals, footprint, rng_key, K_mini, N_mini, occup_value):
    '''
        Get Reachability thorugh Uniform Sampling on Control bounds
        K_mini: Safety Sampling number
        N_mini: Safety Horizion timesteps
    '''
    def mini_cost(carry, params):
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
        # barrier_value = system.h_x(sim_state, cost_map, origin, resolution)
        # clamped_state = jnp.maximum(sim_state[:2], origin)
        # clamped_state = jnp.minimum(clamped_state, wh)
        # ind = jnp.floor((clamped_state-origin)/resolution).astype(jnp.int32)
        # radius = 0.1
        # footprint = np.array([[radius, 0], 
        #                       [0,radius],
        #                       [-radius,0],
        #                       [0,-radius]])+sim_state[:2]
        pos = footprint+sim_state[:2]
        # ind = jnp.floor((sim_state[:2]-origin)/resolution)
        ind = jnp.floor((pos-origin)/resolution)
        ind1 = jnp.maximum(ind, np.array([0,0]))
        ind1 = jnp.minimum(ind1, wh).astype(jnp.int32)
        
        barrier_value = cost_map[ind1[0], ind1[1]]
        # barrier_value = jax.lax.cond(jnp.any(ind1!=ind), lambda x: 0, lambda x: x, barrier_value)
        # cbf_cost = barrier_value != occup_value
        cbf_cost = jnp.all(barrier_value <= occup_value)
        # cbf_cost = True
        norm_dist = jnp.linalg.norm(sim_state[0:2] - goals[:,0:2], axis=1)
        safe = jnp.min(norm_dist)

        return (new_state), (sim_state, safe, cbf_cost)
    def mini_rollout(state, rng_key):
        # Given a state and rollout and determines if rollout is safe
        # u_seq = jax.random.uniform(rng_key, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
        alpha = 0.7
        beta = 0.7
        u_seq = jax.random.beta(rng_key, alpha, beta, shape= (N_mini,2)) \
            * (system.control_bounds[1]-system.control_bounds[0]) \
            + system.control_bounds[0]
        # u_seq = u_seq.at[:, 1].set(system.control_bounds[1][1])
        term_state, state_and_safe = jax.lax.scan(mini_cost, state, u_seq)
        state_hist = state_and_safe[0]
        safe_hist = state_and_safe[1]
        cbf_hist = state_and_safe[2]
        safe = jnp.any(safe_hist<tolerance)
        cbf_safe = jnp.all(cbf_hist)
        # cbf_safe = True
        return (jnp.logical_and(safe,cbf_safe), state_hist, u_seq)
    rng_keys = jax.random.split(rng_key, K_mini)
    # mini_rollout(start.copy(), rng_keys[0])
    # mini_rollout(start.copy(), rng_keys[1])
    # breakpoint()

    safe_controls, state_hist, u_seqs = jax.vmap(mini_rollout, in_axes=(None, 0))(start.copy(), rng_keys)
    safe_ind = jnp.array(jnp.nonzero(safe_controls, size=1, fill_value =100000))
    safe_state_seq = jnp.take(state_hist, safe_ind, axis=0, mode="fill") 
    # safe_state_seq = state_hist[safe_ind]
    safe_u_seqs = u_seqs[safe_ind]
    return jnp.any(safe_controls), safe_state_seq, safe_u_seqs
    # return jnp.any(safe_controls), state_hist[0]
    # return jnp.any(safe_controls), state_hist[0]
    # return True, safe_state_seq
# @functools.partial(jax.jit, static_argnames=('occup_value'))
def get_reachability_mppi_costmap(system, cost_map, origin, resolution, wh, start, goals, footprint, rng_key, K_mini, N_mini, sigma, U_init, temperature=1.0, ais=3, occup_value=100):
    m_elite = int((K_mini//ais)*0.1)
    # m_elite = int((K_mini//ais)*1.0)
    p = 10
    def mini_cost(carry, params):
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
        # ind = jnp.floor((sim_state[:2]-origin)/resolution)
        # ind = jnp.maximum(ind, np.array([0,0]))
        # ind = jnp.minimum(ind, wh).astype(jnp.int32)
        # radius = 0.1
        # footprint = np.array([[radius, 0], 
        #                       [0,radius],
        #                       [-radius,0],
        #                       [0,-radius]])+sim_state[:2]
        pos = footprint + sim_state[:2]
        ind = jnp.floor((pos-origin)/resolution)
        ind = jnp.maximum(ind, np.array([0,0]))
        ind = jnp.minimum(ind, wh).astype(jnp.int32)
        # barrier_value = cost_map[ind[0] + origin[0], ind[1]+origin[1]]
        barrier_value = cost_map[ind[:, 0], ind[:,1]]
        # barrier_value = cost_map[ind[0], ind[1]]
        cbf_cost = jnp.all(barrier_value <= occup_value)
        # cbf_cost = True
        norm_dist = jnp.linalg.norm(sim_state[0:2] - goals[:,0:2], axis=1)
        safe = jnp.min(norm_dist)

        return (new_state), (sim_state, safe, cbf_cost, norm_dist)

    def eval_U_seq(u_seq):
        term_state, state_and_safe = jax.lax.scan(mini_cost, start, u_seq)
        state_hist = state_and_safe[0]
        safe_hist = state_and_safe[1]
        cbf_hist = state_and_safe[2]
        norm_dist = state_and_safe[3]
        # rollout_cost = jnp.sum(norm_dist, axis=1)
        rollout_cost = jnp.min(norm_dist, axis=1)
        rollout_cost = jnp.min(rollout_cost)

        cost = jax.lax.cond(jnp.all(cbf_hist), lambda x: x, lambda x: np.inf+x, safe_hist)
        rollout_cost = jax.lax.cond(jnp.all(cbf_hist), lambda x: x, lambda x: np.inf+x, rollout_cost)
        return (cost, state_hist, rollout_cost)

    def single_u_seq_uniform(rng_subkey):
        # u_seq = jax.random.uniform(rng_subkey, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
        alpha = 0.7
        beta = 0.7
        # alpha = 0.8
        # beta = 0.8
        # alpha = 1.0
        # beta = 1.0
        u_seq = jax.random.beta(rng_key, alpha, beta, shape= (N_mini,2)) \
            * (system.control_bounds[1]-system.control_bounds[0]) \
            + system.control_bounds[0]
        return u_seq

    def single_u_seq_normal(rng_subkey, U, sigma):
        # noise_scaled = jax.random.multivariate_normal(rng_subkey, jnp.zeros(N_mini*2),sigma,  method='svd')
        noise_scaled = jax.random.normal(rng_subkey, shape=(N_mini*2,)) * jnp.diagonal(sigma)
        u_seq = (U + noise_scaled).reshape((-1,2))
        u_seq = jnp.clip(u_seq, system.control_bounds[0], system.control_bounds[1])
        # breakpoint()
        return u_seq 

    def calculate_new_means(costs, seq, original_seq):
        # lowest_ind = jnp.nanargmin(costs)
        exp_cost = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        denom =  jnp.nansum(exp_cost) + 1e-7
        best_u =  original_seq + jnp.nansum(exp_cost[..., None, None] * (seq-original_seq), axis=0) / denom
        return best_u
    # @jax.jit
    def do_AIS(costs, u_seqs):
        ordered_costs = jnp.argsort(costs)
        elite =  u_seqs[ordered_costs[:m_elite], :]
        u_prime = jnp.mean(elite, axis=0)
        sigma_prime = (jnp.cov(elite, rowvar=False) + jnp.eye(N_mini*2) * 10e-9)
        sigma_j  = sigma_prime.copy()
        sigma_j = sigma
        return u_prime, sigma_j, ordered_costs

    def oas(i, carry):
        # carry, params):
        # u, noise_scaled, sigmai = params
        sigma_j, rho_j, rho_j_1, sigma_prime, F = carry
        rho_j_1 = rho_j
        tr_ss = jnp.trace(sigma_j * sigma_prime)
        rho_j = ((1-2.0/p) * tr_ss + jnp.trace(sigma_j)**2)/((m_elite + 1 - 2/p)*tr_ss+(1-m_elite/p)*jnp.trace(sigma_j)**2)
        sigma_j = (1-rho_j)*sigma_j+rho_j*F
        return (sigma_j, rho_j, rho_j_1, sigma_prime, F)

    rng_keys = jax.random.split(rng_key, K_mini//ais)
    # min_costs = 100000000000
    min_costs = jnp.inf
    u_mean = U_init.reshape(((N_mini, 2)))
    U_i  = U_init
    sigma_i = sigma
    
    # state_seqs_all = jnp.zeros((ais, K_mini//ais, N_mini, 3))
    state_seqs_all = jnp.zeros((ais, m_elite, N_mini, system.N_DIMS))
    state_seqs_means = jnp.zeros((ais, N_mini, system.N_DIMS))
    safe_state_seq = jnp.zeros((N_mini, system.N_DIMS))
    safe_u_seqs = jnp.zeros((N_mini, system.N_CONTROLS))
    # for i in range(ais):
    #     (_, rng_keys, U_init, sigma, costs, state_seqs),_ = do_mppi((0, rng_keys, U_init, sigma, jnp.ones((K_mini//ais, N_mini,)), jnp.ones((K_mini//ais, N_mini, 3))), None)
    #     min_costs = jnp.minimum(min_costs, jnp.min(costs))
    #     state_seqs_all = state_seqs_all.at[i, :,:,:].set(state_seqs)
    #     state_seqs_means = state_seqs_means.at[i,:,:].set(state_rollouts(system, start, U_init.reshape((N_mini, 2))))

    for i in range(ais):
        # breakpoint()
        if i == 0:
            u_seqs = jax.vmap(single_u_seq_uniform, (0))(rng_keys)
        else:
            u_seqs = jax.vmap(single_u_seq_normal, (0, None, None))(rng_keys,U_i, sigma_i)
        costs, state_seqs, rollout_cost = jax.vmap(eval_U_seq, (0))(u_seqs)
        min_dist = jnp.min(costs, axis=1)
        min_ind = jnp.argmin(min_dist)
        safe_state_seq = jax.lax.cond(min_costs<jnp.min(min_dist), lambda x: safe_state_seq, lambda x: state_seqs[min_ind], 0)
        safe_u_seqs = jax.lax.cond(min_costs<jnp.min(min_dist), lambda x: safe_u_seqs, lambda x: u_seqs[min_ind], 0)
        min_costs = jnp.minimum(min_costs, jnp.min(min_dist))
        # rollout_cost = jnp.sum(costs, axis=1)
        # rollout_cost = jnp.min(costs, axis=1)
        # U_i, sigma_i, ordered_costs = do_AIS(rollout_cost, u_seqs.reshape((-1,N_mini*2)))
        _, sigma_i, ordered_costs = do_AIS(rollout_cost, u_seqs.reshape((-1,N_mini*2)))
        u_i = U_i.reshape((N_mini,2))
        u_i = calculate_new_means(rollout_cost,  u_seqs, U_i.reshape((N_mini, 2)))
        U_i = u_i.reshape(N_mini*2,)
        # state_seqs_all = state_seqs_all.at[i, :,:,:].set(state_seqs)
        # state_seqs_all = state_seqs_all.at[i, :,:,:].set(state_seqs[ordered_costs[:m_elite], :])
        state_seqs_means = state_seqs_means.at[i,:,:].set(state_rollouts(system, start, u_i))

    # best_u = calculate_new_means(jnp.sum(costs, axis=1), u_seqs, u_mean)
    # costs, state_seqs = jax.vmap(eval_U_seq, (0))(best_u)
    # best_u = calculate_new_means(jnp.min(costs, axis=1), u_seqs, u_mean)

    # safe_ind = jnp.nanargmin(costs)
    # safe_state_seq = jnp.take(state_seqs, safe_ind, axis=0)     
    # safe_u_seqs = u_seqs[safe_ind]

    return min_costs, safe_state_seq, state_seqs_all, state_seqs_means, safe_u_seqs

def get_reachability_controls(system, environment, start, goals, ns, rng_key, K_mini, N_mini):
    '''
        Get Reachability thorugh Uniform Sampling on Control bounds
        K_mini: Safety Sampling number
        N_mini: Safety Horizion timesteps
    '''
    def mini_cost(carry, params):
        dcbf_alpha=0.01
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
    
        barrier_value = jax.vmap(system.h_x, in_axes=(None,0))(sim_state, environment)
        next_barrier_value =  jax.vmap(system.h_x, in_axes=(None,0))(new_state, environment)
        cbf_cost = -next_barrier_value + dcbf_alpha * barrier_value
        cbf_cost = jnp.max(cbf_cost) <= 0
        norm_dist = jnp.linalg.norm(sim_state[0:2] - goals[:,0:2], axis=1)
        safe = jnp.min(norm_dist)

        return (new_state), (sim_state, safe, cbf_cost)
    
    def cond_fun(carry):
        i, safe, u_seq = carry 
        break_condition = (safe<0.5)
        return ~break_condition & (i < N_mini)
    def body_fun(carry):
        i, safe, u_seq = carry


    def mini_rollout(state, rng_key):
        # Given a state and rollout and determines if rollout is safe
        u_seq = jax.random.uniform(rng_key, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
        term_state, state_and_safe = jax.lax.scan(mini_cost, state, u_seq)
        state_hist = state_and_safe[0]
        safe_hist = state_and_safe[1]
        cbf_hist = state_and_safe[2]
        safe = jnp.any(safe_hist<0.5)
        cbf_safe = jnp.all(cbf_hist)
        return (jnp.logical_and(safe,cbf_safe), state_hist, u_seq)
    rng_keys = jax.random.split(rng_key, K_mini)
    safe_controls, state_hist, u_seq = jax.vmap(mini_rollout, in_axes=(None, 0))(start.copy(), rng_keys)

    safe_ind = jnp.array(jnp.nonzero(safe_controls, size=1))
    safe_u_seq = jnp.take(u_seq, safe_ind, axis=0) 
    return jnp.any(safe_controls), safe_u_seq

def get_reachable_set(system, environment, start, rng_key, K_mini, N_mini):
    def mini_cost(carry, params):
        dcbf_alpha=0.01
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
    
        barrier_value = jax.vmap(system.h_x, in_axes=(None,0))(sim_state, environment)
        next_barrier_value =  jax.vmap(system.h_x, in_axes=(None,0))(new_state, environment)
        cbf_cost = -next_barrier_value + dcbf_alpha * barrier_value
        cbf_cost = jnp.max(cbf_cost) <= 0

        return (new_state), (sim_state, cbf_cost)
    def mini_rollout(state, rng_key):
            # Given a state and rollout and determines if rollout is safe
            u_seq = jax.random.uniform(rng_key, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
            term_state, state_and_safe = jax.lax.scan(mini_cost, state, u_seq)
            state_hist = state_and_safe[0]
            cbf_hist = state_and_safe[1]
            cbf_safe = jnp.all(cbf_hist)
            return (cbf_safe, state_hist)

    rng_keys = jax.random.split(rng_key, K_mini)
    safe_controls, state_hist = jax.vmap(mini_rollout, in_axes=(None, 0))(start.copy(), rng_keys)
    safe_ind = jnp.array(jnp.nonzero(safe_controls))
    safe_state_seq = jnp.array(jnp.take(state_hist, safe_ind, axis=0))
    return safe_state_seq.squeeze()