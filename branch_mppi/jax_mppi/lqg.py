import numpy as np
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import casadi as ca
from branch_mppi.systems import NonlinerSystem
import functools
import time

class iLQRSolver:
    def __init__(self, system, dt, Q, R, QT, max_iter=100):
        # Initializations 
        # model, costFunction, X, U, F, derivatives 
        self.system: NonlinerSystem = system
        self.N_DIMS = system.N_DIMS
        self.N_CONTROLS = system.N_CONTROLS
        self.QT = QT
        self.Q = Q
        self.R = R
        self.dt = dt
        self.max_iter = max_iter
        # self.Xinit = np.zeros((model.Xdim, 1))
        # self.Xdes = np.zeros((2, 1))
        # self.szX = self.model.Xdim     # size of state vector:   4
        # self.szU = self.model.Udim     # size of control vector: 2
        # self.X = np.zeros((self.szX, 1)) 
        # self.U = np.zeros((self.szU, 1))
        # self.n = 50
        # self.dt = dt
        # self.iterMax = 200
        # self.uMin = np.tile(-np.inf, (self.szU, 1))
        # self.uMax = np.tile( np.inf, (self.szU, 1))

        # ---- user-adjustable parameters ----
        self.lambdaFactor = 10      # factor for multiplying or dividing lambda
        self.lambdaInit = 1.0      # initial value of Levenberg-Marquardt lambda
        self.lambdaMax = 1000       # exit if lambda exceeds threshold
        # self.epsConverge = 0.001    # exit if relative improvement below threshold
        self.epsConverge = 0.1    # exit if relative improvement below threshold
        # self.flgPrint = 1           # show cost- 0:never, 1:every iter, 2:final
        # self.maxValue = 1E+10       # upper bound on states and costs (Inf: none)
        # self.flgFix = 0             # clear large-fix global flag
    
    def linearization(self, x, u):
        fx = jnp.array(jax.jacfwd(self.system.ode, argnums=0)(x, u))
        fu = jnp.array(jax.jacfwd(self.system.ode, argnums=1)(x, u))
        return (fx, fu)
    


    def cost(self, x, u):
        x = (x + np.pi) % (2 * np.pi) - np.pi
        lb, ub = self.system.control_bounds
        high = (ub-u)
        low = (u-lb)
        log_cost =  - jnp.sum(jnp.log(high+1e-1)  - jnp.log(low+1e-1))
        # inv_cost = -jnp.sum(1/high) - jnp.sum(1/low)
        alpha = 0.1
        # smooth_abs = jnp.sqrt(x.T @ self.Q @ x/2+alpha*alpha)-alpha
        return x.T @ self.Q @ x/2 + u.T @ self.R @ u/2 + log_cost
        # return smooth_abs + u.T @ self.R @ u/2 + log_cost
        # return smooth_abs +inv_cost 

    def terminal_cost(self, x):
        x = (x + np.pi) % (2 * np.pi) - np.pi
        return x.T @ self.QT @ x/2

    # @jax.jit
    @functools.partial(jax.jit, static_argnums=(0,))
    def derivatives(self, x0, u, xbar, ubar): 
        (_, _), x = jax.lax.scan(self.step, (0.0, x0), (u, xbar, ubar))
        Nt = len(xbar)
        dx = x - xbar
        du = u - ubar
        fx, fu = jax.vmap(self.linearization, in_axes=[0,0])(xbar, ubar)
        Ak = fx*self.dt + jnp.kron(jnp.ones((Nt,1)), jnp.eye(self.N_DIMS)).reshape((Nt, self.N_DIMS, self.N_DIMS))
        Bk = fu*self.dt
        dfdx = jax.jacfwd(self.cost, argnums=0)
        dfdu = jax.jacfwd(self.cost, argnums=1)
        dfdxx = jax.jacfwd(jax.jacrev(self.cost, argnums=0), argnums=0)
        dfdux = jax.jacfwd(jax.jacrev(self.cost, argnums=1), argnums=0)
        dfduu = jax.jacfwd(jax.jacrev(self.cost, argnums=1), argnums=1)
        # l = jax.vmap(self.cost, in_axes=[0,0])(xbar, ubar)
        # lx = jax.vmap(dfdx, in_axes=[0,0])(xbar, ubar)
        # lu = jax.vmap(dfdu, in_axes=[0,0])(xbar, ubar)
        # lxx = jax.vmap(dfdxx)(xbar,ubar)
        # lux = jax.vmap(dfdux)(xbar,ubar)
        # luu = jax.vmap(dfduu)(xbar,ubar)
        l = jax.vmap(self.cost, in_axes=[0,0])(dx[:-1], du[:-1])
        lx = jax.vmap(dfdx, in_axes=[0,0])(dx[:-1], du[:-1])
        lu = jax.vmap(dfdu, in_axes=[0,0])(dx[:-1], du[:-1])
        lxx = jax.vmap(dfdxx)(dx[:-1], du[:-1])
        lux = jax.vmap(dfdux)(dx[:-1], du[:-1])
        luu = jax.vmap(dfduu)(dx[:-1], du[:-1])
        l = self.dt * l
        lx = self.dt * lx
        lxx = self.dt * lxx
        lu = self.dt * lu
        luu = self.dt * luu
        lux = self.dt * lux

        l = l.at[-1].set(self.terminal_cost(dx[-1]))
        lx = lx.at[-1].set(jax.jacfwd(self.terminal_cost, argnums=0)(dx[-1]))
        lxx = lxx.at[-1].set(jax.jacrev(jax.jacfwd(self.terminal_cost, argnums=0),argnums=0)(dx[-1]))
        return Ak, Bk, l, lx, lu, lxx, lux, luu
    
    def backward_step(self, carry, params):
        V_x, V_xx, lamb = carry 
        Ak, Bk, l, lx, lu, lxx, lux, luu = params
        Q_x = lx + jnp.dot(Ak.T, V_x) 
        Q_u = lu + jnp.dot(Bk.T, V_x)
        Q_xx = lxx + jnp.dot(Ak.T, jnp.dot(V_xx, Ak)) 
        Q_ux = lux + jnp.dot(Bk.T, jnp.dot(V_xx, Ak))
        Q_uu = luu + jnp.dot(Bk.T, jnp.dot(V_xx, Bk))
        Q_uu_evals, Q_uu_evecs = jnp.linalg.eigh(Q_uu)
        lz_ind = jnp.nonzero(Q_uu_evals<0, size=self.N_CONTROLS, fill_value=self.N_CONTROLS+1)
        Q_uu_evals = Q_uu_evals.at[lz_ind].set(0.0)
        Q_uu_evals += lamb
        Q_uu_inv = jnp.dot(Q_uu_evecs, 
                jnp.dot(jnp.diag(1.0/Q_uu_evals), Q_uu_evecs.T))
        k = -jnp.dot(Q_uu_inv, Q_u)
        K = -jnp.dot(Q_uu_inv, Q_ux)
        V_x = Q_x - jnp.dot(K.T, jnp.dot(Q_uu, k))
        V_xx = Q_xx - jnp.dot(K.T, jnp.dot(Q_uu, K))       
        return (V_x, V_xx, lamb), (k, K)

    @functools.partial(jax.jit, static_argnums=(0,))
    def backward(self, Ak, Bk, l, lx, lu, lxx, lux, luu, lamb): 
        N = len(Ak)
        V = l[-1].copy() # value function
        V_x = lx[-1].copy() # dV / dx
        V_xx = lxx[-1].copy() # d^2 V / dx^2
        # k = jnp.zeros((N, self.N_CONTROLS)) # feedforward modification
        # K = jnp.zeros((N, self.N_CONTROLS, self.N_DIMS)) # feedback gain
        (V_x, V_xx, lamb), (k, K) = jax.lax.scan(self.backward_step, (V_x, V_xx, lamb), (Ak[0:-1], Bk[0:-1], l, lx, lu, lxx, lux, luu), reverse=True)
        k = jnp.concatenate([k, jnp.zeros((1, self.N_CONTROLS))])
        K = jnp.concatenate([K, jnp.zeros((1, self.N_CONTROLS, self.N_DIMS))])
        # breakpoint()
        # for t in range(N-1, -1, -1):
        #     Q_x = lx[t] + jnp.dot(Ak[t].T, V_x) 
        #     # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
        #     Q_u = lu[t] + jnp.dot(Bk[t].T, V_x)

        #     # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
        #     # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
                
        #     # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
        #     Q_xx = lxx[t] + jnp.dot(Ak[t].T, jnp.dot(V_xx, Ak[t])) 
        #     # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
        #     Q_ux = lux[t] + jnp.dot(Bk[t].T, jnp.dot(V_xx, Ak[t]))
        #     # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
        #     Q_uu = luu[t] + jnp.dot(Bk[t].T, jnp.dot(V_xx, Bk[t]))

        #     # Calculate Q_uu^-1 with regularization term set by 
        #     # Levenberg-Marquardt heuristic (at end of this loop)
        #     # jnp.linalg.eigh(Q_uu)
        #     # breakpoint()
        #     Q_uu_evals, Q_uu_evecs = jnp.linalg.eigh(Q_uu)
        #     # lz = jnp.nonzeros((Q_uu_evals<0), size=Q_uu)
        #     # Q_uu_evals[Q_uu_evals < 0] = 0.0
        #     lz_ind = jnp.nonzero(Q_uu_evals<0, size=self.N_CONTROLS, fill_value=self.N_CONTROLS+1)
        #     Q_uu_evals = Q_uu_evals.at[lz_ind].set(0.0)
        #     Q_uu_evals += lamb
        #     Q_uu_inv = jnp.dot(Q_uu_evecs, 
        #             jnp.dot(jnp.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

        #     # 5b) k = -np.dot(Q_uu^-1, Q_u)
        #     # k[t] = -jnp.dot(Q_uu_inv, Q_u)
        #     k = k.at[t].set(-jnp.dot(Q_uu_inv, Q_u))
        #     # 5b) K = -np.dot(Q_uu^-1, Q_ux)
        #     K = K.at[t].set(-jnp.dot(Q_uu_inv, Q_ux))

        #     # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
        #     # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
        #     V_x = Q_x - jnp.dot(K[t].T, jnp.dot(Q_uu, k[t]))
        #     # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
        #     V_xx = Q_xx - jnp.dot(K[t].T, jnp.dot(Q_uu, K[t]))       
        # breakpoint()
        return k, K

    def step(self, carry, params):
        u, x_ref, u_ref = params
        cost, sim_state = carry
        new_state = self.system.jax_dynamics(sim_state, u, 0, self.dt, self.system.nominal_params)
        cost = cost + self.cost(sim_state-x_ref, u-u_ref)
        return (cost, new_state), (sim_state)

    def forward_step(self, carry, params):
        U, x, k, K, x_ref, u_ref = params
        cost, state = carry
        # Unew = U + k + jnp.dot(K, state-x)
        Unew = U + k + K @ (state-x)
        (cost, xnew), _ = self.step((cost, state), (Unew, x_ref, u_ref))
        return (cost, xnew), (state, Unew)


    @functools.partial(jax.jit, static_argnums=(0,))
    def forward(self, x0, x, U, k, K, x_ref, u_ref):
        # N = len(k)
        # cost = 0
        # Unew = np.zeros((N, self.N_CONTROLS))
        # states_new = np.zeros((N, self.N_DIMS))
        # xnew = x0.copy()
        # for i in range(N-1):
        #     Unew[i] = U[i] + k[i] + jnp.dot(K[i], xnew-x[i])
        #     states_new[i] = xnew
        #     (cost, xnew), _ = self.step((cost, xnew), (Unew[i], x_ref[i], u_ref[i]))
        (cost, xnew), (states_new, Unew) = jax.lax.scan(self.forward_step, (0.0, x0.copy()), (U[:-1], x[:-1], k[:-1], K[:-1], x_ref[:-1], u_ref[:-1]))
        # for i in range(N-1):
        #     (cost, xnew), (states_new[i], Unew[i]) = self.forward_step((cost, xnew), (U[i], x[i], k[i], K[i], x_ref[i], u_ref[i]))
        # states_new = states_new.at[-1].set(xnew)
        # Unew = Unew.at[-1].set(U[-1] + k[-1] + jnp.dot(K[-1], xnew -x[-1]))
        states_new = jnp.concatenate([states_new,xnew.reshape((1,self.N_DIMS))])
        Unew = jnp.concatenate([Unew, (U[-1] + k[-1] + jnp.dot(K[-1], xnew -x[-1])).reshape((1,self.N_CONTROLS))])
        # states_new[-1] = xnew
        # Unew[-1] = U[-1] + k[-1] + jnp.dot(K[-1], xnew -x[-1])
        term_state=xnew
        cost = cost + self.terminal_cost(term_state-x_ref[-1])
        # print(f"Forward cost: {cost}")
        # breakpoint()
        return cost, Unew, states_new
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def sim_new_trajectory(self, x0, U, x_ref, u_ref):
        Ak, Bk, l, lx, lu, lxx, lux, luu = self.derivatives(x0, U, x_ref, u_ref)
        (cost, _), states = jax.lax.scan(self.step, (0.0, x0), (U, x_ref, u_ref))
        cost += self.terminal_cost(states[-1]-x_ref[-1])
        return cost, Ak, Bk, l, lx, lu, lxx, lux, luu

    def ilqr(self, x0, x_ref, u_ref):
        x_ref = jnp.array(x_ref)
        u_ref = jnp.array(u_ref)
        U = u_ref
        x = x_ref
        N  = len(x0)
        lamb= self.lambdaInit
        sim_new_trajectory = True
        # Ak, Bk, l, lx, lu, lxx, lux, luu = self.derivatives(x0, U, x_ref, u_ref, lamb)
        # k, K = self.backward(Ak, Bk, l, lx, lu, lxx, lux, luu)
        # cost, U, x = self.forward(x0, x, U, k, K, x_ref, u_ref)
        for i in range(self.max_iter):
            st_time=time.time()
            if sim_new_trajectory:
                # Ak, Bk, l, lx, lu, lxx, lux, luu = self.derivatives(x0, U, x_ref, u_ref)
                # (cost, _), states = jax.lax.scan(self.step, (0.0, x0), (U, x_ref, u_ref))
                # cost += self.terminal_cost(states[-1]-x_ref[-1])
                cost, Ak, Bk, l, lx, lu, lxx, lux, luu = self.sim_new_trajectory(x0, U, x_ref, u_ref)
                old_cost = np.copy(cost)
                sim_new_trajectory = False

            k, K = self.backward(Ak, Bk, l, lx, lu, lxx, lux, luu, lamb)
            cost_new, Unew, states_new = self.forward(x0, x, U, k, K, x_ref, u_ref)

            if cost_new - cost < 1e-6:
                lamb /= self.lambdaFactor
                x = states_new
                U = Unew
                old_cost = cost
                cost = cost_new
                sim_new_trajectory = True
                if i>0 and ((abs(old_cost-cost)/cost)< self.epsConverge):
                    # print("Converged")
                    break
            else:
                lamb*=self.lambdaFactor
                if lamb>self.lambdaMax:
                    # print("Diverged")
                    break
            # print(f"Time for iter: {time.time()-st_time}")

        return x, U, cost



            
def main():
    dt = 0.2
    nlmodel = Unicycle({"lb":np.array([-3.1415, 0.0]), "ub":np.array([3.1415, 4.0])}, dt=dt)
    # nlmodel = Unicycle({"lb":np.array([-10, 0.0]), "ub":np.array([10, 4.0])}, dt=dt)
    # nlmodel = Unicycle({"lb":np.array([-20, 0.0]), "ub":np.array([20, 4.0])}, dt=dt)
    x_sol = np.load("x_sol.npz", allow_pickle=True)[:10]
    u_sol = np.load("u_sol.npz", allow_pickle=True)[:10]
    x_start = x_sol[0,:]+[0.01,0.03,np.pi/15]
    # x_start = x_sol[0,:]

    fac = 1
    dt = dt/fac
    new_x = np.interp(np.linspace(0,1,fac*len(x_sol)), np.linspace(0,1,len(x_sol)), x_sol[:,0],)
    new_y = np.interp(np.linspace(0,1,fac*len(x_sol)), np.linspace(0,1,len(x_sol)), x_sol[:,1],)
    new_z = np.interp(np.linspace(0,1,fac*len(x_sol)), np.linspace(0,1,len(x_sol)), x_sol[:,2],)
    new_states = np.vstack([new_x,new_y,new_z]).T

    new_v = np.interp(np.linspace(0,1,fac*len(x_sol)), np.linspace(0,1,len(x_sol)), u_sol[:,0],)
    new_w = np.interp(np.linspace(0,1,fac*len(x_sol)), np.linspace(0,1,len(x_sol)), u_sol[:,1],)
    new_u = np.vstack([new_v,new_w]).T


    # breakpoint()

    # x_bar = x_sol
    # x_bar[0,:] = x_start

    Q = np.eye(3)*1
    R = np.eye(2)*0.001
    solver = iLQRSolver(nlmodel, dt, Q, R, Q, max_iter=10)
    for i in range(2):
        st_time = time.time()
        x, U, cost  = solver.ilqr(x_start, new_states, new_u)
        print(f"Solving time: {time.time()-st_time}")
    # breakpoint()
    plt.plot(x[:,0], x[:, 1])
    plt.plot(x_sol[:,0], x_sol[:,1], "--")
    plt.show()
    breakpoint()
    print("ended")
    # Ak, Bk, l, lx, lu, lxx, lux, luu = solver.derivatives(x_bar, u_sol)
    # solver.forward(30, x_bar, u_sol)
    # solver.backward(x_bar, u_sol)

if __name__=="__main__":
    from branch_mppi.systems import Unicycle
    main()
