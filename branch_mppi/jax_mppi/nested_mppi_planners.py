import jax
import jax.numpy as jnp
import numpy as np
from branch_mppi.jax_mppi import reachability
import functools

class MPPI_Planner_Occup:
    def __init__(self,sigma, Q, QT, R, temperature, system, num_anci, n_samples, n_mini, N, N_mini, N_safe,  max_sz, tolerance, footprint=[[0.0,0.0]], occup_value=[100], alpha=1, heuristic_weight=0.0):
        self.system = system
        self.num_anci = num_anci
        self.n_samples = n_samples
        self.n_mini = n_mini
        self.N = N
        self.N_mini = N_mini
        self.N_safe = N_safe
        self.max_sz = max_sz
        self.sigma = sigma
        self.Q = Q
        self.QT = QT
        self.R = R
        self.temperature = temperature
        self.tolerance = tolerance
        self.footprint = np.array(footprint)
        self.occup_value = occup_value
        self.alpha = alpha
        self.heuristic_weight= heuristic_weight
        # print(f" self.system: {self.system}")
        # print(f"self.num_anci: {self.num_anci}")
        # print(f"self.n_samples: {self.n_samples}")
        # print(f"self.n_mini: {self.n_mini}")
        # print(f"self.N: {self.N}")
        # print(f"self.N_mini: {self.N_mini}")
        # print(f"self.N_safe: {self.N_safe}")
        # print(f"self.Q: {self.Q}")
        # print(f"self.QT: {self.QT}")
        # print(f"self.R: {self.R}")
        # print(f"self.temperature:{ self.temperature}")

    def eval_U_seq(self, u_seq, original_u, state, q_ref, safe_zones, cost_map, origin, resolution, wh):
        cost_and_term, (state_seq, min_sz_dist) = jax.lax.scan(self.single_sample_running_cost, 
                                            (1.0, state, q_ref, safe_zones, cost_map, origin, resolution, wh), 
                                            u_seq)
        cost = cost_and_term[0]
        terminal_state = cost_and_term[1]
        terminal_cost = jnp.dot((terminal_state - q_ref), jnp.dot(self.QT, (terminal_state - q_ref)))
        cost += terminal_cost 
        cost += u_seq.ravel().T @ jnp.diag(1.0 / jnp.diag(self.sigma)) @ (u_seq.ravel().T - original_u)* (1-self.alpha)*(self.temperature)
        
        return (cost, terminal_state, state_seq, jnp.sum(min_sz_dist))

    def single_sample_running_cost(self, carry, params):
        u = params
        # cost, sim_state, q_ref, safe_zones, cost_map, origin, resolution, wh, reached_goal, collided = carry
        cost, sim_state, q_ref, safe_zones, cost_map, origin, resolution, wh = carry
        # reached_goal = jax.lax.cond(not reached_goal and (np.linalg.norm(sim_state-q_ref)<0.5), lambda x: 1, lambda x: 0)
        new_state = self.system.jax_dynamics(sim_state, u, 0, self.system.dt, self.system.nominal_params)
        dist = sim_state - q_ref
        dx = jnp.dot(new_state-sim_state, jnp.dot(self.Q, new_state-sim_state))
        new_cost = cost + jnp.dot(dist, jnp.dot(self.Q, dist)) *(1+ jax.lax.cond(dx==0, lambda x: 0.0, lambda x: 1/x, dx))
        # new_cost = cost + jnp.dot(dist, jnp.dot(self.Q, dist))
            
        ind = jnp.floor((sim_state[:2]-origin)/resolution)
        ind1 = jnp.maximum(ind, np.array([0,0]))
        ind1 = jnp.minimum(ind1, wh).astype(jnp.int32)
        
        barrier_value = cost_map[ind1[0], ind1[1]] >= 10
        # barrier_value = jax.lax.cond(jnp.any(ind1!=ind), lambda x: 0, lambda x: x, barrier_value)

        # collided = jax.lax.cond(not collided and barrier_value, lambda x:1, lambda x:0)
        # new_cost += jax.lax.cond(barrier_value or collided, lambda x: 100000, lambda x: 0.0, 0) 
        new_cost += jax.lax.cond(barrier_value, lambda x: np.inf, lambda x: 0.0, 0) 
        # new_cost += barrier_value * 100000
        min_dist = jnp.min(jnp.linalg.norm(new_state[0:2] - safe_zones[:,0:2], axis=1))
        new_cost += (min_dist**2)*self.heuristic_weight
        # if self.heuristic:
        #     new_cost = new_cost + (min_dist**2)*3
        return (new_cost, new_state, q_ref, safe_zones, cost_map, origin, resolution, wh), (sim_state, min_dist)
        # return (new_cost, new_state, q_ref, safe_zones, cost_map, origin, resolution, wh, reached_goal, collided), (sim_state, min_dist)

    def mini_mppi(self, state_seq, rng_key, cost_map, origin, resolution, wh, safe_zones):
        """
        Given a state seq, does mini rollouts mapped across seqeunces 
        and determines if all states in sequence is safe
        """
        if self.N_mini > 0 and self.n_mini > 0:
            safe_flag, safe_state_seq, safe_u_seqs = jax.vmap(reachability.get_reachability_costmap, 
                            in_axes=(None, None, None, None, None, None, 0, None, None, None, None, None, None)) \
            (self.system, 
             cost_map,
             origin, 
             resolution, 
             wh, 
             self.tolerance, 
             state_seq, 
             safe_zones, 
             self.footprint, 
             rng_key, 
             self.n_mini, 
             self.N_mini,
             self.occup_value
             )
            return jnp.all(safe_flag), safe_state_seq.squeeze(), safe_flag[0], safe_u_seqs
        else:
            # return True, jnp.array([state_seq[0]]), True, np.zeros((1,self.system.N_DIMS, 1))
            # return True, jnp.expand_dims([state_seq[0]], 1), True, np.zeros((1,self.system.N_DIMS, 1))
            return True, jnp.expand_dims(state_seq[0],(0,1)), True, np.zeros((1,self.system.N_DIMS, 1))

    def mini_mppi_ais(self, state_seq, rng_key, cost_map, origin, resolution, wh, safe_zones):
        """
        Given a state seq, does mini rollouts mapped across seqeunces 
        and determines if all states in sequence is safe
        """
        if self.N_mini > 0 and self.n_mini > 0:
            ais = 3
            safe_flag, safe_state_seq, state_seqs_all, state_seqs_means, safe_u_seqs = jax.vmap(reachability.get_reachability_mppi_costmap, 
                in_axes=(None, None, None, None, None, 0, None, None, None, None, None, None, None,  None, None, None)) \
                (self.system,  
                 cost_map,  
                origin, 
                resolution, 
                wh,         
                state_seq, 
                safe_zones, 
                self.footprint, 
                rng_key, 
                self.n_mini, 
                self.N_mini,  
                self.sigma[:self.N_mini*2, :self.N_mini*2]*2, 
                jnp.kron(jnp.ones((self.N_mini,)), jnp.array([0,self.system.control_bounds[1][1]])), 
                # self.temperature, 
                self.temperature/1000, 
                ais,
                self.occup_value
                ) 
            current_safe_state_seq = safe_state_seq.squeeze()
            # current_safe_state_seq = jax.lax.cond(safe_flag[0] < self.tolerance, lambda x: current_safe_state_seq, lambda x: jnp.ones_like(current_safe_state_seq)*0.0, 0)
            # return jnp.all(safe_flag<self.tolerance, axis=0), safe_state_seq.squeeze()[0], safe_flag[0]<self.tolerance, safe_u_seqs
            return jnp.all(safe_flag<self.tolerance, axis=0), current_safe_state_seq, safe_flag[0]<self.tolerance, safe_u_seqs
        else:
            # jax.debug.breakpoint()
            return True, jnp.expand_dims(state_seq[0],(0,1)), True, np.zeros((1,self.system.N_DIMS, 1))

    def single_u_seq(self, rng_subkey, U, sigma):
        # noise_scaled = jax.random.multivariate_normal(rng_subkey, jnp.zeros(N*2), sigma,  method='svd')
        noise_scaled = jax.random.normal(rng_subkey, shape=(self.N*2,)) * jnp.diagonal(sigma)
        u_seq = (U + noise_scaled).reshape((-1,2))
        u_seq = jnp.clip(u_seq, self.system.control_bounds[0], self.system.control_bounds[1])
        return u_seq 

    # @jax.jit
    def ess(self, costs, temperature):
        w_i = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        ess = jnp.nansum(w_i)**2 / jnp.nansum(w_i **2)
        return ess


    def calculate_new_means(self, costs, seq, original_seq):
        lowest_ind = jnp.nanargmin(costs)
        temperature = self.temperature
        exp_cost = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        denom =  jnp.nansum(exp_cost) + 1e-7
        best_u =  original_seq + jnp.nansum(exp_cost[..., None, None] * (seq-original_seq), axis=0) / denom

        lowest_u = seq[lowest_ind]
        return best_u, lowest_u, temperature

    @functools.partial(jax.jit, static_argnames=('num_anci', 'n_samples'))
    def mppi(self, state, U, U_anci, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh, num_anci, n_samples):
        ais = 1
        m_elite = 5

        N = self.N
        N_safe = self.N_safe

        original_seqs = U.reshape((-1,2))
        rng_keys = jax.random.split(rng_key, n_samples+num_anci)
        generate_u = jax.vmap(self.single_u_seq, (0,None, None))

        sigma = self.sigma
        mini_guy = functools.partial(self.mini_mppi_ais, cost_map=cost_map, origin=origin, resolution=resolution, wh=wh, safe_zones=safe_zones)
        number_safe = 0

        u_seqs = generate_u(rng_keys[:n_samples], U, sigma)
        u_seqs = jnp.append(U_anci.reshape((-1,N,2)),u_seqs, axis=0)
        all_costs_and_state = jax.vmap(self.eval_U_seq,(0, None, None, None, None, None,None,None,None)) \
                                (u_seqs, U, state, q_ref, safe_zones, cost_map, origin, resolution, wh)
        costs = all_costs_and_state[0]
        all_state_seq = all_costs_and_state[2]
        min_sz_dist = all_costs_and_state[3]
        finite_cost_ind = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=n_samples+num_anci, fill_value=0))
        collision_free_state_seq = jnp.take(all_state_seq, finite_cost_ind,axis=0).squeeze()

        # test = mini_guy(collision_free_state_seq[0, :N_safe,:], rng_keys[0])

        mini_costs, all_con_state_seq, current_safe = jax.vmap(mini_guy, in_axes=[0,0])(collision_free_state_seq[:, :N_safe,:], rng_keys)
        number_safe += jnp.sum(mini_costs)
        safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(mini_costs, size=n_samples+num_anci, fill_value=1000009))).squeeze()
        min_cost = jnp.nanmin(jnp.take(costs, safe_ind, fill_value=np.inf))
        not_safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(1-mini_costs, size=n_samples+num_anci, fill_value=1000000))).squeeze()
        costs =  costs.at[not_safe_ind].set(np.inf)
        # costs = jax.lax.cond(jnp.any(current_safe), lambda x: costs.at[not_safe_ind].set(np.inf), lambda x: min_sz_dist, 0)
        # costs = jax.lax.cond(jnp.isfinite(min_cost), lambda x: costs.at[not_safe_ind].set(np.inf), lambda x: min_sz_dist, 0)
        ordered_costs = jnp.argsort(costs)
        elite =  u_seqs[ordered_costs[:m_elite], :].reshape((m_elite,self.N*2))
        U = jnp.mean(elite, axis=0)
        sigma = (jnp.cov(elite, rowvar=False) + jnp.eye(self.N*2) * 10e-9)

        safe_state_seq = jnp.take(all_state_seq, safe_ind, axis=0)
        best_u, lowest_u = self.calculate_new_means(costs, u_seqs, original_seqs)
        best_u = lowest_u
        best_costs_and_state = self.eval_U_seq(best_u, U, state, q_ref, safe_zones, cost_map, origin, resolution, wh)
        best_cost = best_costs_and_state[0]
        best_state_seq = best_costs_and_state[2]
        best_sz_dist = best_costs_and_state[3]
        mini_cost, _, _ = mini_guy(best_state_seq, rng_keys[0])

        best_cost = jax.lax.cond(mini_cost, lambda x: x, lambda x: np.inf, best_cost)

        new_u = jnp.roll(best_u, shift=-1, axis=0)
        # new_u = new_u.at[-1].set(new_u[-2])
        new_u = new_u.at[-1].set(np.zeros_like(new_u[-2]))
        new_U = new_u.reshape((1,-1)).squeeze(0)

        return best_u, new_u, new_U, min_cost, safe_state_seq ,all_con_state_seq[:,0,:][0], number_safe, jnp.any(current_safe), best_cost, best_sz_dist, all_state_seq, all_con_state_seq

    @jax.jit
    def mppi_mmodal(self, state, U_original, U_anci, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh):
        num_anci = self.num_anci
        n_samples = self.n_samples
        N = self.N
        N_safe = self.N_safe

        # each_controller_n = n_samples
        each_controller_n = int(n_samples/(num_anci+1))
        original_seqs = U_original.reshape((-1,2))

        rng_keys = jax.random.split(rng_key, (num_anci+1)*(each_controller_n)).reshape((num_anci+1, each_controller_n,2))
        generate_u = jax.vmap(self.single_u_seq, (0,None, None))
        mini_guy = functools.partial(self.mini_mppi_ais, cost_map=cost_map, origin=origin, resolution=resolution, wh=wh, safe_zones=safe_zones)
        sigma = self.sigma
        number_safe = 0
        # print(U_original.shape)
        # print(U_anci.shape)
        U_total = jnp.vstack((U_original, U_anci)).reshape((num_anci+1, N*2))
        u_seqs = jax.vmap(generate_u, in_axes=(0,0, None))(rng_keys, U_total, sigma).reshape((-1,N,self.system.N_CONTROLS))
        all_costs_and_state = jax.vmap(self.eval_U_seq,(0, None, None, None, None, None,None,None,None)) \
                                (u_seqs, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh)
        costs = all_costs_and_state[0]
        all_state_seq = all_costs_and_state[2]
        min_sz_dist = all_costs_and_state[3]
        finite_cost_ind = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=0))
        collision_free_state_seq = jnp.take(all_state_seq, finite_cost_ind,axis=0).squeeze()
        # jax.debug.breakpoint()
        mini_costs, all_con_state_seq, current_safe, safe_u_seqs = jax.vmap(mini_guy, in_axes=[0,0])(collision_free_state_seq[:, :N_safe,:], rng_keys.reshape((-1,2)))
        # print(safe_flags)
        # breakpoint()
        con_state_seq = all_con_state_seq[jnp.nonzero(mini_costs, size=n_samples, fill_value=100000), 0, :][0]
        number_safe += jnp.sum(mini_costs)
        safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(mini_costs, size=n_samples, fill_value=1000009))).squeeze()
        min_cost = jnp.nanmin(jnp.take(costs, safe_ind, fill_value=np.inf))
        not_safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(1-mini_costs, size=n_samples, fill_value=1000000))).squeeze()

        # uncomment to consider safety condition
        costs = costs.at[not_safe_ind].set(np.inf)
        # costs = jax.lax.cond(jnp.any(current_safe), lambda x: costs.at[not_safe_ind].set(np.inf), lambda x: min_sz_dist, 0)
        # costs = jax.lax.cond(jnp.any(jnp.isfinite(costs)), lambda x: costs, lambda x: min_sz_dist, 0)

        # costs =  costs.at[not_safe_ind].set(np.inf)
        # breakpoint()
        safe_state_seq = jnp.take(all_state_seq, safe_ind, axis=0)
        best_u, lowest_u, temperature = self.calculate_new_means(costs, u_seqs, original_seqs)
        best_u = lowest_u
        new_u = jnp.roll(best_u, shift=-1, axis=0)
        new_u = new_u.at[-1].set(new_u[-2])
        # new_u = new_u.at[-1].set(jnp.zeros_like(new_u[-2]))
        new_U = new_u.reshape((1,-1)).squeeze(0)
        return best_u, new_u, new_U, min_cost, safe_state_seq, con_state_seq, number_safe, jnp.any(current_safe), collision_free_state_seq, temperature, safe_u_seqs.squeeze(), all_con_state_seq, costs, all_state_seq 

    @jax.jit
    def mppi_mmodal_no_ais(self, state, U_original, U_anci, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh):
        num_anci = self.num_anci
        n_samples = self.n_samples
        N = self.N
        N_safe = self.N_safe

        # each_controller_n = n_samples
        each_controller_n = int(n_samples/(num_anci+1))
        original_seqs = U_original.reshape((-1,2))

        rng_keys = jax.random.split(rng_key, (num_anci+1)*(each_controller_n)).reshape((num_anci+1, each_controller_n,2))
        generate_u = jax.vmap(self.single_u_seq, (0,None, None))
        mini_guy = functools.partial(self.mini_mppi, cost_map=cost_map, origin=origin, resolution=resolution, wh=wh, safe_zones=safe_zones)
        sigma = self.sigma
        number_safe = 0

        U_total = jnp.vstack((U_original, U_anci)).reshape((num_anci+1, N*2))
        u_seqs = jax.vmap(generate_u, in_axes=(0,0, None))(rng_keys, U_total, sigma).reshape((-1,N,self.system.N_CONTROLS))
        all_costs_and_state = jax.vmap(self.eval_U_seq,(0, None, None, None, None, None,None,None,None)) \
                                (u_seqs, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh)
        costs = all_costs_and_state[0]
        all_state_seq = all_costs_and_state[2]
        min_sz_dist = all_costs_and_state[3]
        finite_cost_ind = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=0))
        collision_free_state_seq = jnp.take(all_state_seq, finite_cost_ind,axis=0).squeeze()
        mini_costs, all_con_state_seq, current_safe, safe_u_seqs = jax.vmap(mini_guy, in_axes=[0,0])(collision_free_state_seq[:, :N_safe,:], rng_keys.reshape((-1,2)))
        # print(safe_flags)
        # breakpoint()
        con_state_seq = all_con_state_seq[jnp.nonzero(mini_costs, size=n_samples, fill_value=100000), 0, :][0]
        number_safe += jnp.sum(mini_costs)
        safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(mini_costs, size=n_samples, fill_value=1000009))).squeeze()
        min_cost = jnp.nanmin(jnp.take(costs, safe_ind, fill_value=np.inf))
        not_safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(1-mini_costs, size=n_samples, fill_value=1000000))).squeeze()

        # uncomment to consider safety condition
        costs = costs.at[not_safe_ind].set(np.inf)
        # costs = jax.lax.cond(jnp.any(current_safe), lambda x: costs.at[not_safe_ind].set(np.inf), lambda x: min_sz_dist, 0)
        # costs = jax.lax.cond(jnp.any(jnp.isfinite(costs)), lambda x: costs, lambda x: min_sz_dist, 0)

        # costs =  costs.at[not_safe_ind].set(np.inf)
        # breakpoint()
        safe_state_seq = jnp.take(all_state_seq, safe_ind, axis=0)
        best_u, lowest_u, temperature = self.calculate_new_means(costs, u_seqs, original_seqs)
        new_u = jnp.roll(best_u, shift=-1, axis=0)
        new_u = new_u.at[-1].set(jnp.zeros_like(new_u[-2]))
        new_U = new_u.reshape((1,-1)).squeeze(0)
        return best_u, new_u, new_U, min_cost, safe_state_seq, con_state_seq, number_safe, jnp.any(current_safe), collision_free_state_seq, temperature, safe_u_seqs.squeeze(), all_con_state_seq, costs # u_applied, new control history, all control seqs, # # turn 

    @jax.jit
    def mppi_mmodal_individual(self, state, U_original, U_anci, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh):
        num_anci = self.num_anci
        n_samples = self.n_samples
        N = self.N
        N_safe = self.N_safe

        # each_controller_n = n_samples
        each_controller_n = int(n_samples/(num_anci+1))
        original_seqs = U_original.reshape((-1,2))
        # rng_keys = jax.random.split(rng_key, each_controller_n*(num_anci+1)).reshape((num_anci+1, each_controller_n, 2))
        rng_keys = jax.random.split(rng_key, num_anci+1).reshape((num_anci+1,  2))
        generate_u = jax.vmap(self.single_u_seq, (0,None, None))

        # best_us = jnp.zeros((num_anci, N, 2))
        # best_costs = jnp.zeros((num_anci))
        U_total = jnp.vstack((U_original, U_anci)).reshape((num_anci+1, N*2))
        mppi_fun = functools.partial(self.mppi, q_ref=q_ref, safe_zones=safe_zones,cost_map=cost_map, origin=origin, resolution=resolution, wh=wh, num_anci=1, n_samples=each_controller_n-1)
        outputs = jax.vmap(mppi_fun, (None, 0, 0, 0))(state, U_total, U_total, rng_keys, )
        # outputs = jax.vmap(mppi_fun, (None, 0, None, 0))(state, U_total, U_original, rng_keys, )

        # for i in range(num_anci):
        #     U=U_original
        #     sigma = self.sigma
        #     
        #     safe_state_seq = jnp.take(all_state_seq, safe_ind, axis=0)
        #     con_state_seq = jnp.take(con_state_seq, safe_ind, axis=0)

        #     # safe_costs = jnp.take(costs, safe_ind, fill_value=np.inf) 
        #     # safe_seq = jnp.take(u_seqs, safe_ind, axis=0)
        #     # best_u = self.calculate_new_means(safe_costs, safe_seq, original_seqs)
        #     # best_u, _ = self.calculate_new_means(costs, u_seqs, original_seqs)
        #     best_u, _ = self.calculate_new_means(costs, u_seqs, U_anci[i].reshape((-1,2)))
        #     best_costs_and_state = self.eval_U_seq(best_u, U, state, q_ref, safe_zones, cost_map, origin, resolution, wh)
        #     best_us = best_us.at[i,:,:].set(best_u)
        #     best_costs = best_costs.at[i].set(best_costs_and_state[0])
        best_us = jnp.array(outputs[0])
        min_costs = jnp.array(outputs[3])
        safe_state_seqs  = jnp.array(outputs[4])
        con_state_seqs = jnp.array(outputs[5])
        number_safe = jnp.array(outputs[6])
        current_safe = jnp.array(outputs[7])
        costs = jnp.array(outputs[8])
        best_sz_dists = jnp.array(outputs[9])
        all_state_seqs = jnp.array(outputs[10])
        all_con_state_seqs = jnp.array(outputs[11])

        # print(f"best_us {best_us}")
        
        best_costs =costs 
        # best_costs = jax.lax.cond(jnp.any(current_safe), lambda x: costs, lambda x: best_sz_dists, 0)
        min_cost = jnp.min(min_costs)
        safe_state_seq = safe_state_seqs.reshape((each_controller_n*(num_anci+1), N, 3))
        all_state_seq = all_state_seqs.reshape((each_controller_n*(num_anci+1), N, 3))
        con_state_seq = con_state_seqs.reshape((each_controller_n*(num_anci+1), self.N_mini, 3))
        all_con_state_seqs = all_con_state_seqs.reshape((each_controller_n*(num_anci+1), self.N_safe, self.N_mini, 3))
        number_safe = jnp.sum(number_safe)

        best_u, lowest_u = self.calculate_new_means(best_costs, best_us, original_seqs)
        # best_u, lowest_u = self.calculate_new_means(min_costs, best_us, original_seqs)
        new_u = jnp.roll(best_u, shift=-1, axis=0)
        new_u = new_u.at[-1].set(new_u[-2])
        new_U = new_u.reshape((1,-1)).squeeze(0)


        return best_u, new_u, new_U, min_cost, safe_state_seq, con_state_seq, number_safe, jnp.any(current_safe), all_state_seq, all_con_state_seqs, costs # u_applied, new control history, all control seqs, # # turn 

    def _tree_flatten(self):
        children = ( self.sigma, self.Q, self.QT, self.R, self.temperature,)
        aux_data = {'system': self.system, 
                    'num_anci': self.num_anci,
                    'n_samples': self.n_samples,
                    'n_mini':self.n_mini,
                    'N': self.N,
                    'N_mini': self.N_mini,
                    'N_safe': self.N_safe,
                    'max_sz': self.max_sz,
                    'tolerance': self.tolerance,
                    'footprint': self.footprint,
                    'occup_value': self.occup_value,
                    'alpha': self.alpha,
                    'heuristic_weight': self.heuristic_weight,
                    }
        return (children, aux_data)
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


from jax import tree_util 
tree_util.register_pytree_node(MPPI_Planner_Occup,
                               MPPI_Planner_Occup._tree_flatten,
                               MPPI_Planner_Occup._tree_unflatten)
print('Planners registered')
