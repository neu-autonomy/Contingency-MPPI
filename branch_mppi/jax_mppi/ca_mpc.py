import functools
import jax
import jax.numpy as jnp
# from trajax import integrators
# from trajax.experimental.sqp import shootsqp, util
# from trajax.optimizers import ilqr
import time
from functools import partial

import numpy as np
from branch_mppi.systems import NonlinerSystem
# from branch_mppi.jax_mppi import plot_utils
import matplotlib.pyplot as plt
import casadi as ca
from casadi import SX, MX, DM
import pydecomp as pdc

def rotate_2dvectors(x, theta, center=np.array([0.0,0.0])):
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    for i in range(len(x)):
        x[i,:2] = R.T @ x[i,:2] - R.T @ center
    return x

def rotate_polyhedral(A, b, theta, center=np.array([0.0,0.0])):
    A_rot = rotate_2dvectors(A, theta)
    b_rot = b - A @center.reshape(-1,1)
    return A_rot, b_rot

def diff_angles(x1, x2):
    # v1 = MX([ca.cos(x1), ca.sin(x1)])
    # v2 = MX([ca.cos(x2), ca.sin(x2)])
    v1 = MX(2,1)
    v1[0][0] = ca.cos(x1)
    v1[1][0] = ca.sin(x1)
    v2 = MX(2,1)
    v2[0][0] = ca.cos(x2)
    v2[1][0] = ca.sin(x2)
    diff = ca.atan2(v1[0]*v2[1] - v1[1]*v2[0], ca.dot(v1, v2))
    return diff

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def linspace_theta_wrap_to_pi(start, end, steps=10) -> np.ndarray:
    start = wrap_to_pi(start)
    end = wrap_to_pi(end)
    if (np.sign(start) != np.sign(end)) and  (np.abs(start-end) > np.pi):
        res = wrap_to_pi(np.linspace(0, wrap_to_pi(end-start), steps) + start)
        return res
    return np.linspace(start, end, steps)

def find_Nonlin_Controls(path,start, solver, dis, system, Nt, occupied, box, planner, ns=15, que=None, U=None):
    st = time.time()
    path = planner.cutToMax(path, dis)
    path = planner.cutToSafe(path)
    if U is None:
        U =  np.kron(np.ones((1, Nt+1)), [0.0, system.control_bounds[1][1]]).ravel()
    u0 = U.reshape((-1,2))
    p = np.array(path)[:,0:2]
    if len(occupied) < 1:
        A = [np.zeros((ns,2)) for i in range(len(p))]
        b = [np.ones((ns,1))*1000 for i in range(len(p))]
    else:
        A, b = pdc.convex_decomposition_2D(occupied, p, box)
    X0, idx =  planner.discretizePath(p,Nt+1)
    thetas = np.ones((Nt+1, 1)) * start[2]
    X0_f = np.hstack((np.array(X0), thetas.reshape(-1,1)))
    rX0 = rotate_2dvectors(X0_f, start[2], start[:2])
    rX0[:,2] = wrap_to_pi(rX0[:,2] - start[2])
    
    obs_halfplane = [[A[i],b[i]] for i in idx]
    A_np = []
    b_np = []

    for Ab in obs_halfplane:
        # print(len(Ab[0]))
        Ai = np.vstack((Ab[0], np.zeros((ns-len(Ab[0]),2))))
        bi = np.vstack((Ab[1], np.zeros((ns-len(Ab[1]),1))))
        A_rot, b_rot = rotate_polyhedral(Ai,bi,start[2],start[:2])
        b_rot[np.argwhere(np.isnan(b_rot))] =  np.inf
        A_np.append(A_rot)
        b_np.append(b_rot)
    start_solve = time.time()
    x_sol, u_sol =solver(
    rX0.reshape(-1,), 
    u0.reshape(-1)[:], 
    np.array([0.0,0.0,0.0]),
    rX0[-1,:].reshape(1,-1), 
    np.array(A_np).reshape(-1,2), 
    np.array(b_np).reshape(-1),
    rX0.reshape(-1,))       
    x_sol = np.array(x_sol).reshape((-1,3))
    u_sol = np.array(u_sol).reshape((-1,2))
    if que is not None:
        que.put(u_sol)
    return x_sol, u_sol, (A,b)

def cas_shooting(system: NonlinerSystem, start, goal, obs, R, Q, QT, Nt, X0, u0, max_iter=1000):
    st1 = time.time()
    opti = ca.Opti()
    x = opti.variable(Nt + 1, 3)  # state (x, z, theta)
    u = opti.variable(Nt+1, 2)  # control (v, omega) 
    goal_vector = np.tile(goal[:2], (Nt,1))
    # goal_vector = X0[:-1,:2]
    delta = x[1:,:2] - goal_vector

    # opti.minimize(10*ca.sumsqr(delta[:,-1]) + 100*ca.sumsqr(delta[-1,:2]))
    # opti.minimize(ca.sumsqr(delta[:]) )
    # opti.minimize(ca.sumsqr(delta[Nt+1]) )
    # delta = x[1:,:2] - X0[1:,:2] 
    opti.minimize(ca.sumsqr(delta[:]) /100)
    # opti.minimize(ca.dot(delta[:]) )
    # breakpoint()
    
    opti.subject_to(x[0,0] == start[0])
    opti.subject_to (x[0,1] == start[1])
    opti.subject_to(x[0,2] == start[2])

    # for control_idx, bound in enumerate(system.control_bounds):
    for t in range(Nt):
        opti.subject_to(u[t, 0] <= np.pi/ 3)
        opti.subject_to(u[t, 0] >= -np.pi/3)
        opti.subject_to(u[t, 1] <= 4.0)
        opti.subject_to(u[t, 1] >= 0.0)
        # breakpoint()
        opti.subject_to(ca.DM(obs[t][0]) @ x[t,:2].T < ca.DM(obs[t][1]))

    for t in range(Nt):
        # Extract states and controls
        px_next = x[t + 1, 0]
        py_next = x[t + 1, 1]
        theta_next = x[t + 1, 2]
        px_now = x[t, 0]
        py_now = x[t, 1]
        theta_now = x[t, 2]
        v = u[t, 1]
        omega = u[t, 0]

        # These dynamics are smooth enough that we probably can get away with a simple
        # forward Euler integration.

        # x_dot = v * cos(theta)
        opti.subject_to(px_next == px_now + v * ca.cos(theta_now) * system.dt)
        # y_dot = v * sin(theta)
        opti.subject_to(py_next == py_now + v * ca.sin(theta_now) * system.dt)
        # theta_dot = omega
        opti.subject_to(theta_next == theta_now + ca.tan(omega) * v * system.dt)
    opti.set_initial(x, X0)
    opti.set_initial(u, u0)
    p_opts = {"expand": 1, "print_time": False, "verbose": False}
    s_opts = {"Minor print level":0, "Major print level":1, "Summary file": 0, "Solution": "no", "Suppress options listings":1, "Time Limit":0.01, "Timing level":2}
    opti.solver("snopt", p_opts, s_opts)
    st2 = time.time()
    try:
        sol1 = opti.solve()
    except:
        endtime = time.time()
        print(f"Solve Time: {endtime-st2}")
        print(f"Cas time: {st2-st1}")
        print(f"Total time: {endtime-st1}") 
        return opti.debug.value(x[:,:]), opti.debug.value(u[:,:])

    endtime = time.time()
    print(f"Solve Time: {endtime-st2}")
    print(f"Cas time: {st2-st1}")
    print(f"Total time: {endtime-st1}") 
    return sol1.value(x[:,:]), sol1.value(u[:,:])

def cas_coll(system: NonlinerSystem, start, goal, obs, R, Q, QT, Nt, X0, u0, max_iter=1000):
    st1 = time.time()
    opti = ca.Opti()
    x = opti.variable(Nt + 1, 3)  # state (x, z, theta)
    u = opti.variable(Nt+1, 2)  # control (v, omega) 
    # dt = opti.variable(1)
    goal_vector = np.tile(goal[:2], (Nt,1))
    delta = x[1:,:2] - goal_vector

    # opti.minimize(10*ca.sumsqr(delta[:,-1]) + 100*ca.sumsqr(delta[-1,:2]))
    # opti.minimize(ca.sumsqr(delta[:]) )
    # opti.minimize(ca.sumsqr(delta[Nt+1]) )
    # delta = x[1:,:2] - X0[1:,:2] 
    opti.minimize(ca.sumsqr(delta[:]) /1000)
    # opti.minimize(ca.dot(delta[:]) )
    # breakpoint()
    
    opti.subject_to(x[0,0] == start[0])
    opti.subject_to (x[0,1] == start[1])
    opti.subject_to(x[0,2] == start[2])
    # opti.subject_to(dt > 0)
    # for control_idx, bound in enumerate(system.control_bounds):
    for t in range(Nt):
        opti.subject_to(u[t, 0] <= np.pi/ 3)
        opti.subject_to(u[t, 0] >= -np.pi/3)
        opti.subject_to(u[t, 0] <= np.pi/2)
        opti.subject_to(u[t, 0] >= -np.pi/2)
        opti.subject_to(u[t, 1] <= 4.0)
        opti.subject_to(u[t, 1] >= 0.0)
        # breakpoint()
        opti.subject_to(ca.DM(obs[t][0]) @ x[t,:2].T < ca.DM(obs[t][1]))

    for t in range(Nt):
        # Extract states and controls
        px_next = x[t + 1, 0]
        py_next = x[t + 1, 1]
        theta_next = x[t + 1, 2]
        omega_next = u[t+1, 0]
        v_next = u[t+1, 1]

        px_now = x[t, 0]
        py_now = x[t, 1]
        theta_now = x[t, 2]
        omega_now = u[t, 0]
        v_now = u[t, 1]

        # Intermediate Nodes for Hermite-Simpson Collocation


        f_k_x = v_now * ca.cos(theta_now) 
        f_k_y = v_now * ca.sin(theta_now) 
        f_k_theta = ca.tan(omega_now)*v_now

        f_k_x1 = v_next * ca.cos(theta_next) 
        f_k_y1 = v_next * ca.sin(theta_next) 
        f_k_theta1 = ca.tan(omega_next)*v_next

        # px_12 = (px_next + px_now)/2 + system.dt/8 * (f_k_x-f_k_x1)
        # py_12 = (py_next + py_now)/2 + system.dt/8 * (f_k_y-f_k_y)
        # theta_12 = (theta_next + omega_next)/2 + system.dt/8 * (f_k_theta-f_k_theta1)

        # omega_12 = (omega_next + omega_now) / 2
        # v_12 = (v_next + v_now) / 2

        # f_k_x12 = v_12 * ca.cos(theta_12) 
        # f_k_y12 = v_12 * ca.sin(theta_12) 
        # f_k_theta12 = ca.tan(omega_12)*v_12
        
        # opti.subject_to((f_k_x + f_k_x1 + 4*f_k_x12)*system.dt / 6 == px_next-px_now)
        # opti.subject_to((f_k_y + f_k_y1 + 4*f_k_y12)*system.dt / 6 == py_next-py_now)
        # opti.subject_to((f_k_theta+f_k_theta1+4*f_k_theta12)*system.dt / 6 == theta_next-theta_now)

        opti.subject_to((f_k_x+f_k_x1)*system.dt*0.5 == px_next-px_now)
        opti.subject_to((f_k_y+f_k_y1)*system.dt*0.5 == py_next-py_now)
        opti.subject_to((f_k_theta+f_k_theta1)*system.dt*0.5 == theta_next-theta_now)

        # opti.subject_to((f_k_x+f_k_x1)*dt*0.5 == px_next-px_now)
        # opti.subject_to((f_k_y+f_k_y1)*dt*0.5 == py_next-py_now)
        # opti.subject_to((f_k_theta+f_k_theta1)*dt*0.5 == theta_next-theta_now)

        # opti.subject_to(px_next == px_now + v_now * ca.cos(theta_now) * system.dt)
        # opti.subject_to(py_next == py_now + v_now * ca.sin(theta_now) * system.dt)
        # opti.subject_to(theta_next == theta_now + ca.tan(omega_now) * v_now * system.dt)

    opti.set_initial(x, X0)
    opti.set_initial(u, u0)
    # opti.set_initial(dt, 0.2)
    p_opts = {"expand": 1, "print_time": False, "verbose": False}
    # p_opts = {"print_time": False, "verbose": False}
    s_opts = {"tol":1e-3, "max_iter": max_iter, "mu_strategy":"adaptive"}
    # opti.solver("ipopt", p_opts, s_opts)
    opti.solver("snopt", p_opts)
    # opti.solver("blocksqp", p_opts)
    # opti.solver("sleqp", p_opts)
    # opti.solver("alpaqa", p_opts)
    # opti.solver("dsdp", p_opts)
    s_opts = {}
    # opti.solver("worhp", p_opts, s_opts)

    st2 = time.time()
    try:
        sol1 = opti.solve()
    except:
        endtime = time.time()
        print(f"Solve Time: {endtime-st2}")
        print(f"Cas time: {st2-st1}")
        print(f"Total time: {endtime-st1}") 
        return opti.debug.value(x[:,:]), opti.debug.value(u[:,:])

    endtime = time.time()
    print(f"Solve Time: {endtime-st2}")
    print(f"Cas time: {st2-st1}")
    print(f"Total time: {endtime-st1}") 
    return sol1.value(x[:,:]), sol1.value(u[:,:])
 
def cas_shooting_solver(system: NonlinerSystem, Nt, ode, ns=6, dt=None):
    opti = ca.Opti()
    # dxdt, state, control = system.cas_ode()
    # ode = ca.Function('ode', [state, control], [dxdt])
    if dt is None:
        dt = system.dt
    x = opti.variable((Nt + 1)*3)  # state (x, z, theta)
    u = opti.variable((Nt+1)* 2)  # control (v, omega) 
    goal = opti.parameter(1,3)
    start = opti.parameter(1,3)
    obs_A = opti.parameter((Nt+1)*ns, 2)
    obs_b = opti.parameter((Nt+1)*ns)
    path = opti.parameter((Nt+1)*3)
    goal_vector = ca.repmat(goal[:2].T, Nt+1,1)
    indicies = []
    for t in range(Nt+1):
        indicies.append(3*t)
        indicies.append(3*t+1)
    delta = x[indicies]-goal_vector ### ORIINAL
    # delta = x[indicies]-path[indicies]
    # delta = x - path
    cost = ca.diag(np.kron(np.arange(0,Nt), [1,1]))
    opti.minimize(ca.sumsqr(delta) /100)
    # opti.minimize(delta.T @ DM.eye((Nt+1)*2) @ delta /100)
    # opti.minimize(delta.T @ cost @ delta /100)
    opti.subject_to(x[0] == start[0])
    opti.subject_to (x[1] == start[1])
    opti.subject_to(x[2] == start[2])

    # for control_idx, bound in enumerate(system.control_bounds):
    for t in range(Nt):
        opti.subject_to(u[2*t] < system.control_bounds[1][0])
        opti.subject_to(u[2*t] > system.control_bounds[0][0])
        opti.subject_to(u[2*t+1] < system.control_bounds[1][1])
        opti.subject_to(u[2*t+1] > system.control_bounds[0][1])
        opti.subject_to(obs_A[ns*t:ns*(t+1),:] @ x[3*t:3*t+2] < obs_b[ns*t:ns*(t+1)])
    # print(system.control_bounds)
    for t in range(Nt):
        # Extract states and controls
        # px_next = x[3*(t + 1)]
        # py_next = x[3*(t + 1)+1]
        # theta_next = x[3*(t + 1)+2]
        # px_now = x[3*t]
        # py_now = x[3*t+1]
        # theta_now = x[3*t+2]
        # v = u[2*t+1]
        # omega = u[2*t]
        fx = ode(x[3*t:3*(t+1)], u[2*t:2*(t+1)])

        ct = ca.cos(x[3*t+2])
        st = ca.sin(x[3*t+2])
        R = ca.MX(2,2)
        R_1 = ca.MX(2,2)
        w = ca.MX(2,2)
        R[0,0] = ct
        R[0,1] = -st
        R[1,0] = st
        R[1,1] = ct
        # breakpoint()

        ct_1 = ca.cos(x[3*(t+1)+2])
        st_1 = ca.sin(x[3*(t+1)+2])

        ct_w = ca.cos(fx[2]*dt)
        st_w = ca.sin(fx[2]*dt)
        w[0,0] = ct_w
        w[0,1] = -st_w
        w[1,0] = st_w
        w[1,1] = ct_w

        # opti.subject_to(x[3*(t+1):3*(t+2)] == x[3*t:3*(t+1)] + fx *dt)
        opti.subject_to(x[3*(t+1)] == x[3*t] + fx[0] *dt)
        opti.subject_to(x[3*(t+1)+1] == x[3*t+1] + fx[1] *dt)
        # opti.subject_to(x[3*(t+1)+2] == x[3*t+2] + fx[2] *dt)

        # opti.subject_to(ca.sin(x[3*(t+1)+2])**2 == 1 - ca.cos(x[3*t+2] + fx[2] *dt)**2)
        # opti.subject_to( ca.fmod(x[3*(t+1)+2]+np.pi, 2*np.pi) == ca.fmod(x[3*t+2] + fx[2] *dt+np.pi, 2*np.pi))
        R_1s = w @ R 
        opti.subject_to( (ca.arctan2(st_1, ct_1) - ca.arctan2(R_1s[1,0], R_1s[0,0])) == 0.0)
        # opti.subject_to(R_1s[0,0]==ct_1)
        # opti.subject_to(R_1s[1,0]==st_1)
        # opti.subject_to()

    jit_options = {"flags": ["-O3", "-march=native"], "compiler":"ccache gcc"}
    p_opts = {"expand": 1, "print_time": False, "verbose": False, "jit": True, "compiler":"shell", "jit_options":jit_options,'jit_cleanup':True, 'jit_temp_suffix':False}
    # p_opts = {"expand": 1, "print_time": True, "verbose": True, "jit": True, "compiler":"shell", "jit_options":jit_options}
    # p_opts = {"expand": 1, "print_time": False, "verbose": False}
    s_opts = {"tol":1e-2, "max_iter": 1000, "mu_strategy":"adaptive", "print_level":0}
    opti.solver("ipopt", p_opts, s_opts)
    # opti.solver("madnlp", p_opts)
    # s_opts = {"Minor print level":0, 
    #           "Major print level":0, 
    #           "Summary file": 0, 
    #           "Solution": "no", 
    #           "Suppress options listings":1, 
    #           "Timing level":0,
    #           "Scale option": 2,
    #           "Major iterations limit":1000
    #           }
    # opti.solver("snopt", p_opts, s_opts)
    # opti.solver("snopt", p_opts)
    # opti.solver("hpipm", p_opts)
    # opti.solver("ipopt", p_opts)
    # opti.solver("blocksqp", p_opts, {"print_header":False, "print_iteration":False, "globalization": False})
    # opti.solver('blocksqp', 
                # {"expand": 1, 
                #  "print_time": True, 
                #  "verbose": False, 
                #  "jit": True, 
                #  "compiler":"shell", 
                #  "jit_options":jit_options,
                #  'jit_cleanup':True, 
                #  'jit_temp_suffix':False,
                #     "print_header":False,
                #     "print_iteration":False,
                #     "max_iter":500,
                #     "globalization":True,
                #     "opttol":1e-2,
                #     "warmstart":True
                #  })
    # opti.solver("fatrop", 
    #             {"expand": 1, 
    #              "print_time": True, 
    #              "verbose": False, 
    #         "jit": True, 
    #              "compiler":"shell", 
    #              "jit_options":jit_options,
    #              'jit_cleanup':True, 
    #              'jit_temp_suffix':False},
    #              { "tol":1e-2, "max_iter": 500, "print_level":0})

    # endtime =time.time()
    # print(f"Cas time: {endtime-st1}")
    return opti.to_function('F', [x, u, start, goal,  obs_A, obs_b, path], [x, u], ['x', 'u', 'start', 'goal', 'obs_A', 'obs_b', 'path'], ['x_opt', 'u_opt'])
    # return opti

def cas_shooting_sparse(system: NonlinerSystem, start, goal, obs, R, Q, QT, Nt, X0, u0, max_iter=1000):
    st1 = time.time()
    opti = ca.Opti()
    # x = opti.variable(Nt + 1, 3)  # state (x, z, theta)
    # u = opti.variable(Nt+1, 2)  # control (v, omega) 
    x = opti.variable((Nt + 1)*3)  # state (x, z, theta)
    u = opti.variable((Nt+1)* 2)  # control (v, omega)
    goal_vector = np.tile(goal[:2], (Nt,1)).reshape(-1,)
    # goal_vector = X0[:-1,:2]
    # delta = x[:] - goal_vector
    indicies = []
    for t in range(Nt):
        indicies.append(3*t)
        indicies.append(3*t+1)
    delta = x[indicies]-X0.reshape(-1)[indicies]
    # delta = x[indicies]-goal_vector
    # opti.minimize(10*ca.sumsqr(delta[:,-1]) + 100*ca.sumsqr(delta[-1,:2]))
    # opti.minimize(ca.sumsqr(delta[:]) )
    # opti.minimize(ca.sumsqr(delta[Nt+1]) )
    # delta = x[1:,:2] - X0[1:,:2] 
    opti.minimize(ca.sumsqr(delta) /1000)
    # opti.minimize(ca.dot(delta[:]) )
    # breakpoint()
    
    opti.subject_to(x[0] == start[0])
    opti.subject_to (x[1] == start[1])
    opti.subject_to(x[2] == start[2])

    # for control_idx, bound in enumerate(system.control_bounds):
    for t in range(Nt):
        opti.subject_to(u[2*t] <= np.pi/ 3)
        opti.subject_to(u[2*t] >= -np.pi/3)
        opti.subject_to(u[2*t+1] <= 4.0)
        opti.subject_to(ca.DM(obs[t][0]) @ x[3*t:3*t+2] < ca.DM(obs[t][1]))

    for t in range(Nt):
        # Extract states and controls
        px_next = x[3*(t + 1)]
        py_next = x[3*(t + 1)+1]
        theta_next = x[3*(t + 1)+2]
        px_now = x[3*t]
        py_now = x[3*t+1]
        theta_now = x[3*t+2]
        v = u[2*t+1]
        omega = u[2*t]

        # These dynamics are smooth enough that we probably can get away with a simple
        # forward Euler integration.

        # x_dot = v * cos(theta)
        opti.subject_to(px_next == px_now + v * ca.cos(theta_now) * system.dt)
        # y_dot = v * sin(theta)
        opti.subject_to(py_next == py_now + v * ca.sin(theta_now) * system.dt)
        # theta_dot = omega
        opti.subject_to(theta_next == theta_now + ca.tan(omega) * v * system.dt)
    opti.set_initial(x, X0.reshape(-1,))
    opti.set_initial(u, u0.reshape(-1,))
    
    # jit_options = {"flags": ["-Ofast", "-march=native"], "verbose": True}
    p_opts = {"expand": 1, "print_time": False, "verbose": False}
    # s_opts = {"tol":1e-2, "max_iter": 1000, "mu_strategy":"adaptive", "print_level":0, "linear_solver":'ma97'}
    # opti.solver("ipopt", p_opts, s_opts)
    # opti.solver("ipopt", p_opts)
    s_opts = {"Minor print level":0, "Major print level":0, "Summary file": 0, "Solution": "no", "Suppress options listings":1}
    opti.solver("snopt", p_opts, s_opts)

    st2 = time.time()
    try:
        sol1 = opti.solve()
    except:
        endtime = time.time()
        print(f"Solve Time: {endtime-st2}")
        print(f"Cas time: {st2-st1}")
        print(f"Total time: {endtime-st1}") 
        # return opti.debug.value(x[:,:]), opti.debug.value(u[:,:])
        return opti.debug.value(x[:,:]).reshape((Nt+1,3)), opti.debug.value(u[:,:]).reshape((Nt+1,2))

    endtime = time.time()
    print(f"Solve Time: {endtime-st2}")
    print(f"Cas time: {st2-st1}")
    print(f"Total time: {endtime-st1}") 
    # return sol1.value(x[:,:]), sol1.value(u[:,:])
    return sol1.value(x[:,:]).reshape((Nt+1,3)), sol1.value(u[:,:]).reshape((Nt+1,2))  

def cas_coll_sparse(system: NonlinerSystem, start, goal, obs, R, Q, QT, Nt, X0, u0, max_iter=1000):
    st1 = time.time()
    opti = ca.Opti()
    # x = opti.variable(Nt + 1, 3)  # state (x, z, theta)
    # u = opti.variable(Nt+1, 2)  # control (v, omega) 
    x = opti.variable((Nt + 1)*3)  # state (x, z, theta)
    u = opti.variable((Nt+1)* 2)  # control (v, omega)
    goal_vector = np.tile(goal[:2], (Nt,1)).reshape(-1,)
    # goal_vector = X0[:-1,:2]
    # delta = x[:] - goal_vector
    indicies = []
    for t in range(Nt):
        indicies.append(3*t)
        indicies.append(3*t+1)
    delta = x[indicies]-goal_vector
    opti.minimize(ca.sumsqr(delta) /1000)

    opti.subject_to(x[0] == start[0])
    opti.subject_to (x[1] == start[1])
    opti.subject_to(x[2] == start[2])

    for t in range(Nt):
        opti.subject_to(u[2*t] <= np.pi/ 3)
        opti.subject_to(u[2*t] >= -np.pi/3)
        opti.subject_to(u[2*t+1] <= 4.0)
        opti.subject_to(u[2*t+1] >= 0.0)
        # breakpoint()        
        opti.subject_to(ca.DM(obs[t][0]) @ x[3*t:3*t+2] < ca.DM(obs[t][1]))

    for t in range(Nt):
        # Extract states and controls
        px_next = x[3*(t + 1)]
        py_next = x[3*(t + 1)+1]
        theta_next = x[3*(t + 1)+2]
        px_now = x[3*t]
        py_now = x[3*t+1]
        theta_now = x[3*t+2]
        v_now = u[2*t+1]
        omega_now = u[2*t]
        v_next = u[2*(t+1)+1]
        omega_next = u[2*(t+1)]

        f_k_x = v_now * ca.cos(theta_now) 
        f_k_y = v_now * ca.sin(theta_now) 
        f_k_theta = ca.tan(omega_now)*v_now

        f_k_x1 = v_next * ca.cos(theta_next) 
        f_k_y1 = v_next * ca.sin(theta_next) 
        f_k_theta1 = ca.tan(omega_next)*v_next

        opti.subject_to((f_k_x+f_k_x1)*system.dt*0.5 == px_next-px_now)
        opti.subject_to((f_k_y+f_k_y1)*system.dt*0.5 == py_next-py_now)
        opti.subject_to((f_k_theta+f_k_theta1)*system.dt*0.5 == theta_next-theta_now)

    opti.set_initial(x, X0.reshape(-1,))
    opti.set_initial(u, u0.reshape(-1,))
    p_opts = {"expand": 1, "print_time": False, "verbose": False}
    # s_opts = {"tol":1e-2, "max_iter": 1000, "mu_strategy":"adaptive", "print_level":0, "linear_solver":'ma57'}
    # opti.solver("ipopt", p_opts, s_opts)
    s_opts = {"Minor print level":0, "Major print level":0, "Print file": 0, "Summary file": 0, "Solution": "no", "Suppress options listings":1}
    opti.solver("snopt", p_opts, s_opts)

    st2 = time.time()
    try:
        sol1 = opti.solve()
    except:
        endtime = time.time()
        print(f"Solve Time: {endtime-st2}")
        print(f"Cas time: {st2-st1}")
        print(f"Total time: {endtime-st1}") 
        # return opti.debug.value(x[:,:]), opti.debug.value(u[:,:])
        return opti.debug.value(x[:,:]).reshape((Nt+1,3)), opti.debug.value(u[:,:]).reshape((Nt+1,2))

    endtime = time.time()
    print(f"Solve Time: {endtime-st2}")
    print(f"Cas time: {st2-st1}")
    print(f"Total time: {endtime-st1}") 
    # return sol1.value(x[:,:]), sol1.value(u[:,:])
    return sol1.value(x[:,:]).reshape((Nt+1,3)), sol1.value(u[:,:]).reshape((Nt+1,2))  

def qp_shooting_linearized(system, ode, jac_x, jac_u, start, goal, obs, Nt, X0, u0, ns=6):
    st_time = time.time()
    # dxdt, state, control = system.cas_ode()
    # ode = ca.Function('ode', [state, control], [dxdt], {'jit':True})
    # jac_x = ca.Function('jac_x', [state, control], [ca.jacobian(dxdt, state)])
    # jac_u = ca.Function('jax_u', [state, control], [ca.jacobian(dxdt, control)])
    x = SX.sym('x', (Nt + 1)*3)  # state (x, z, theta)
    u = SX.sym('u', (Nt+1)* 2)  # control (v, omega)
    goal_vector = np.tile(goal[:2], (Nt,1)).reshape(-1,)
    indicies = []
    for t in range(Nt):
        indicies.append(3*t)
        indicies.append(3*t+1)
    # delta = x[indicies]-goal_vector
    delta = (x[indicies]-X0.reshape(-1)[indicies])/10

    start_constraints = DM(3,(Nt+1)*3)
    start_constraints[0,0] = 1 
    start_constraints[1,1] = 1 
    start_constraints[2,2] = 1 

    control_cons_l = DM(2*(Nt+1),(Nt+1)*2).full()
    polytope_cons_l = DM(ns*(Nt+1), (Nt+1)*3).full()
    cons_A = DM(3*(Nt+1),3*(Nt+1)).full()
    cons_B = DM(3*(Nt+1),2*(Nt+1)).full()
    dyn_eq_cons = DM(3*(Nt+1), 3*(Nt+1)).full()
    dyn_ode = DM(3*(Nt+1), 1).full()

    control_cons_ub = DM(2*(Nt+1), 1)
    control_cons_lb = DM(2*(Nt+1), 1)
    polytope_cons_ub = DM.ones(ns*(Nt+1), 1)
    polytope_cons_lb = DM.ones(ns*(Nt+1), 1)*-100
    dx = x - X0.reshape((-1,1))
    du = u - u0 .reshape((-1,1))
    print(f"Before Loop: {time.time()-st_time}")
    before_loop_time = time.time()
    for t in range(Nt): 
        A = jac_x(X0[t,:], u0[t,:])
        B = jac_u(X0[t,:], u0[t,:])
        fxu = ode(X0[t,:], u0[t,:])
        control_cons_l[2*t, 2*t] = 1
        control_cons_l[2*t+1, 2*t+1] = 1
        control_cons_ub[2*t] = np.pi/3
        control_cons_ub[2*t+1] = 4.0
        control_cons_lb[2*t] = -np.pi/3
        control_cons_lb[2*t+1] = 0.0

        # (Axbu + fxu) * dt = x[n+1]-x[n]
        cons_A[3*t:3*(t+1), 3*t:3*(t+1)] = A 
        cons_B[3*t:3*(t+1), 2*t:2*(t+1)] = B 
        dyn_ode[3*t:3*(t+1),] = fxu
        dyn_eq_cons[3*t:3*(t+1),3*(t+1):3*(t+2)] = -DM.eye(3) # x[n+1]
        dyn_eq_cons[3*t:3*(t+1),3*(t):3*(t+1)] = DM.eye(3) # x[n]

        polytope_cons_l[ns*t:ns*(t+1),3*t:3*t+2] = ca.DM(obs[t][0])
        polytope_cons_ub[ns*t:ns*(t+1), 0] = ca.DM(obs[t][1])
    print(f"Loop time:{time.time()-before_loop_time} ")
    print(f'Problem Construction Time:{time.time()-st_time}')
    qp = {'x': ca.vertcat(x,u), 'f':ca.sumsqr(delta),'g':ca.vertcat(
            start_constraints @ x,
            control_cons_l@u,
            polytope_cons_l@x,
            system.dt*(cons_A @ dx + cons_B @ du + dyn_ode) + dyn_eq_cons@x 
            )}
    # p_opts = {"expand": 1, "print_time": False, "verbose": False, "error_on_fail":False}
    # opti.solver("osqp", p_opts)
    opts= {"expand":1, "error_on_fail":False, "print_time":False, "verbose":False,"osqp.verbose":False}
    solver = ca.qpsol("s",'osqp',qp, opts)
    start_solver = time.time()
    sol = solver(x0=np.vstack((X0.reshape((-1,1)), u0.reshape((-1,1)))), 
            ubg=ca.vertcat(
                start.reshape((-1,1)), 
                control_cons_ub,
                polytope_cons_ub,
                DM.zeros(3*(Nt+1),1)
                ),
            lbg=ca.vertcat(
                start.reshape((-1,1)),
                control_cons_lb,
                polytope_cons_lb,
                DM.zeros(3*(Nt+1),1)
                ))

    print(f"QP time:{time.time()-start_solver} ")
    print(f"Total time:{time.time()-st_time} ")
    return np.array(sol['x'][:3*(Nt+1)]).reshape((Nt+1,3)), \
            np.array(sol['x'][3*(Nt+1):]).reshape((Nt+1,2)) 

class QPShoot():
    def __init__(self, system, Nt, ns=6, dt=None):
        self.system = system
        nx = system.N_DIMS
        nu = system.N_CONTROLS
        if dt is None:
            self.dt = system.dt
        else:
            self.dt=dt
        self.x = SX.sym('x', (Nt + 1)*nx)  # state (x, z, theta)
        self.u = SX.sym('u', (Nt+1)* nu)  # control (v, omega)
        self.Nt = Nt
        self.ns = ns
        self.start_constraints = DM(nx,(Nt+1)*nx)
        self.start_constraints[0,0] = 1 
        self.start_constraints[1,1] = 1 
        self.start_constraints[2,2] = 1 
        self.control_cons_l = DM(nu*(Nt+1), (Nt+1)*nu)
        self.polytope_cons_l = DM(ns*(Nt+1), (Nt+1)*nx)
        self.cons_A = DM(nx*(Nt+1),nx*(Nt+1))
        self.cons_B = DM(nx*(Nt+1),nu*(Nt+1))
        self.dyn_eq_cons = DM(nx*(Nt+1), nx*(Nt+1))
        self.dyn_ode = DM(nx*(Nt+1), 1)
        self.control_cons_ub = DM(nu*(Nt+1), 1)
        self.control_cons_lb = DM(nu*(Nt+1), 1)
        self.polytope_cons_ub = DM.ones(ns*(Nt+1), 1)
        self.polytope_cons_lb = DM.ones(ns*(Nt+1), 1)*-100
        self.obj_indices = []
        for t in range(Nt):
            self.obj_indices.append(nx*t)
            self.obj_indices.append(nx*t+1)
            self.dyn_eq_cons[nx*t:nx*(t+1),nx*(t+1):nx*(t+2)] = -DM.eye(3) # x[n+1]
            self.dyn_eq_cons[nx*t:nx*(t+1),nx*(t):nx*(t+1)] = DM.eye(3) # x[n]
            self.control_cons_l[nu*t, nu*t] = 1
            self.control_cons_l[nu*t+1, nu*t+1] = 1
            self.control_cons_ub[nu*t] = system.control_bounds[1][0]
            self.control_cons_ub[nu*t+1] = system.control_bounds[1][1]
            self.control_cons_lb[nu*t] = system.control_bounds[0][0]
            self.control_cons_lb[nu*t+1] = system.control_bounds[0][1]
        dxdt, state, control = self.system.cas_ode()
        self.ode = ca.Function('ode', [state, control], [dxdt])
        self.jac_x = ca.Function('jac_x', [state, control], [ca.jacobian(dxdt, state)])
        self.jac_u = ca.Function('jax_u', [state, control], [ca.jacobian(dxdt, control)])

    def solve(self, start, goal, obs, Nt, X0, u0):
        st_time = time.time()
        nx = self.system.N_DIMS
        nu = self.system.N_CONTROLS
        x = self.x
        u = self.u
        goal_vector = np.tile(goal[:2], (Nt,1)).reshape(-1,)
        # delta = x[self.obj_indices]-goal_vector
        # delta = (x[self.obj_indices]-X0.reshape(-1)[self.obj_indices])
        delta = 100

        dx = x - X0.reshape((-1,1))
        du = u - u0 .reshape((-1,1))
        before_loop_time = time.time()- st_time
        # print(f"Before Loop: {before_loop_time}")
        loop_st = time.time()
        for t in range(Nt): 
            A = self.jac_x(X0[t,:], u0[t,:])
            B = self.jac_u(X0[t,:], u0[t,:])
            fxu = self.ode(X0[t,:], u0[t,:])

            # (Axbu + fxu) * dt = x[n+1]-x[n]
            self.cons_A[nx*t:nx*(t+1), nx*t:nx*(t+1)] = A 
            self.cons_B[nx*t:nx*(t+1), nu*t:nu*(t+1)] = B 
            self.dyn_ode[nx*t:nx*(t+1),0] = fxu
            self.polytope_cons_l[self.ns*t:self.ns*(t+1),nx*t:nx*t+2] = obs[t][0]
            self.polytope_cons_ub[self.ns*t:self.ns*(t+1), 0] = obs[t][1]
        loop_time = time.time()-loop_st
        # print(f"Loop time:{loop_time} ")
        qp = {'x': ca.vertcat(x,u), 'f':ca.sumsqr(delta),'g':ca.densify(ca.vertcat(
                self.start_constraints @ x,
                self.control_cons_l@u,
                self.polytope_cons_l@x,
                self.dt*(self.cons_A @ dx + self.cons_B @ du + self.dyn_ode) 
                    + self.dyn_eq_cons@x 
                ))}
        # print(f'Problem Construction Time:{time.time()-st_time}')
        opts= {"expand":1, "error_on_fail":False, "print_time":False, "verbose":False,"osqp.verbose":False}
        solver = ca.qpsol("s",'osqp',qp, opts)
        start_solver = time.time()
        sol = solver(x0=np.vstack((X0.reshape((-1,1)), u0.reshape((-1,1)))), 
                ubg=ca.vertcat(
                    start.reshape((-1,1)), 
                    self.control_cons_ub,
                    self.polytope_cons_ub,
                    DM.zeros(3*(Nt+1),1)
                    ),
                lbg=ca.vertcat(
                    start.reshape((-1,1)),
                    self.control_cons_lb,
                    self.polytope_cons_lb,
                    DM.zeros(3*(Nt+1),1)
                    ))
        # print(f"QP time:{time.time()-start_solver} ")
        # print(f"Total time:{time.time()-st_time} \n")
        return np.array(sol['x'][:3*(Nt+1)]).reshape((Nt+1,3)), \
                np.array(sol['x'][3*(Nt+1):]).reshape((Nt+1,2)) 

def cas_shooting_linearized_solver(system, Nt, ns=6):
    dxdt, state, control = system.cas_ode()
    jac_x = ca.jacobian(dxdt, state)
    jac_u = ca.jacobian(dxdt, control)

    opti = ca.Opti('conic')
    x = opti.variable((Nt + 1)*3)  # state (x, z, theta)
    u = opti.variable((Nt+1)* 2)  # control (v, o``mega)
    goal = opti.parameter(1,3)
    start = opti.parameter(1,3)
    obs_A = opti.parameter((Nt+1)*ns, 2)
    obs_b = opti.parameter((Nt+1)*ns)
    X0 = opti.parameter((Nt+1),3)
    u0 = opti.parameter((Nt+1),2)
    goal_vector = ca.repmat(goal[:2].T, Nt,1)
    indicies = []
    for t in range(Nt):
        indicies.append(3*t)
        indicies.append(3*t+1)
    # delta = x[indicies]-goal_vector
    delta = x[indicies]-X0.reshape((-1,1))[indicies]
    opti.minimize(delta.T @ DM.eye(Nt*2) @ delta /100)
    # opti.minimize(1000)
    opti.subject_to(x[0] == start[0])
    opti.subject_to (x[1] == start[1])
    opti.subject_to(x[2] == start[2])

    dx = x - X0.reshape((-1,1))
    du = u - u0.reshape((-1,1))
    for t in range(Nt):
        opti.subject_to(u[2*t] < np.pi/ 3)
        opti.subject_to(u[2*t] > -np.pi/3)
        opti.subject_to(u[2*t+1] < 4.0)
        opti.subject_to(u[2*t+1] > 0.0)
        opti.subject_to(obs_A[ns*t:ns*(t+1),:] @ x[3*t:3*t+2] < obs_b[ns*t:ns*(t+1),:1])
        A = ca.substitute([jac_x], [state, control], [X0[3*t:3*(t+1)], u0[2*t:2*(t+1)]])[0]
        # breakpoint() 
        # A = A.substitue(control, u[2*t:2*(t+1)])
        B = ca.substitute([jac_u], [state,control], [X0[3*t:3*(t+1)], u0[2*t:2*(t+1)]])[0]
        # B = B.substitue(control, u[2*t:2*(t+1)])
        # fxu = dxdt(X0[t,:], u0[t,:])
        fxu = ca.substitute([dxdt],[state,control], [X0[3*t:3*(t+1)], u0[2*t:2*(t+1)]])[0]
        # breakpoint()
        Axbu = (A @ dx[3*t:3*(t+1)] +  B @ du[2*t:2*(t+1)] + fxu)*system.dt
        x_next = (x[3*(t+1):3*(t+2)] - x[3*t:3*(t+1)])
        opti.subject_to(x_next  == Axbu)
    jit_options = {"flags": ["-O3", "-march=native"], "compiler":"ccache gcc"}
    p_opts = {"expand": 1, "print_time": False, "verbose": False, "jit": True,
              "compiler":"shell", "error_on_fail":False, "jit_options":jit_options}
    opti.solver("osqp", p_opts)
    return opti.to_function('F', [x, u, start, goal,  obs_A, obs_b, X0, u0], [x, u], ['x', 'u', 'start', 'goal', 'obs_A', 'obs_b', 'X0', 'u0'], ['x_opt', 'u_opt'])

def cas_shooting_linearized(system, start, goal, obs, Nt, X0, u0):
    st_time = time.time()
    opti = ca.Opti('conic')
    x = opti.variable((Nt + 1)*3)  # state (x, z, theta)
    u = opti.variable((Nt+1)* 2)  # control (v, o``mega)
    goal_vector = np.tile(goal[:2], (Nt,1)).reshape(-1,)
    indicies = []
    for t in range(Nt):
        indicies.append(3*t)
        indicies.append(3*t+1)
    # delta = x[indicies]-goal_vector
    delta = x[indicies]-X0.reshape(-1)[indicies]
    opti.minimize(1000)
    # opti.minimize(ca.sumsqr(delta) /1000)

    opti.subject_to(x[0] == start[0])
    opti.subject_to (x[1] == start[1])
    opti.subject_to(x[2] == start[2])

    print(f"First loop time: {time.time()-st_time}")
    for t in range(Nt):
        opti.subject_to(u[2*t] < np.pi/ 3)
        opti.subject_to(u[2*t] > -np.pi/3)
        opti.subject_to(u[2*t+1] < 4.0)
        opti.subject_to(u[2*t+1] > 0.0)
        opti.subject_to(ca.DM(obs[t][0]) @ x[3*t:3*t+2] < ca.DM(obs[t][1]))

    # start_x = X0[0,:]
    # start_u = u0[0,:]
    # A = ca.DM(np.array(jax.jacfwd(system.ode, argnums=0)(start_x, start_u)))
    # B = ca.DM(np.array(jax.jacfwd(system.ode, argnums=1)(start_x, start_u)))
    for t in range(0,Nt):
        start_x = X0[t,:]
        start_u = u0[t,:]
        A = ca.DM(np.array(jax.jacfwd(system.ode, argnums=0)(start_x, start_u)))
        B = ca.DM(np.array(jax.jacfwd(system.ode, argnums=1)(start_x, start_u)))
        
        xt = x[3*t:3*(t+1)]
        ut = u[2*t:2*(t+1)]
        dx = xt - start_x
        du = ut - start_u

        Axbu = (A @ dx +  B @ du + system.ode(start_x, start_u))*system.dt
        x_next = (x[3*(t+1):3*(t+2)] - xt)
        # Axbu = (A @ dx +  B @ du + system.jax_dynamics(start_x, start_u))
        # x_next = (x[3*(t+1):3*(t+2)])
        # breakpoint()

        opti.subject_to(x_next  == Axbu)

    opti.set_initial(x, X0.reshape(-1,))
    opti.set_initial(u, u0.reshape(-1,))
    p_opts = {"expand": 1, "print_time": False, "verbose": False, "error_on_fail":False}
    opti.solver("osqp", p_opts)
    start_solve_time = time.time()
    try:
        sol1 = opti.solve()
        x_sol = sol1.value(x[:,:]).reshape((Nt+1,3))
        u_sol = sol1.value(u[:,:]).reshape((Nt+1,2))   
    except:
        x_sol = opti.debug.value(x[:,:]).reshape((Nt+1,3))
        u_sol = opti.debug.value(u[:,:]).reshape((Nt+1,2)) 
    print(f"QP time: {time.time()-start_solve_time}")
    print(f"Total time: {time.time()-st_time}")
    return np.array(x_sol), np.array(u_sol)



def plot_simulation_result(states, obs, safe_zones, text="", max_arrows=30, ax=None):
    """
    Plot the trajectory and orientation of the car given the state history.

    Parameters:
        states (list of np.array): List of states [x, y, theta, v] at each time step.
    """
    x_vals = [state[0] for state in states]
    y_vals = [state[1] for state in states]
    theta_vals = [state[2] for state in states]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Generate circle for CBF
    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        ax.add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        ax.add_artist(rect)

    # Plot the trajectory
    ax.plot(x_vals, y_vals, '-o', label='Trajectory', markersize=4, alpha=0.5)

    # Plot the orientation at each point
    for i in range(0, len(states), int(len(states)/max_arrows)):  # 
        x, y, theta = states[i][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')

    # plot start and end point
    ax.scatter(3.5, 3.5, s=200, color="green", alpha=0.75, label="init. position")
    ax.scatter(safe_zones[0][0], safe_zones[0][1], s=200, color="purple", alpha=0.75, label="target position")

    ax.text(0.0,0.0, text, transform=plt.gca().transAxes,verticalalignment="bottom")

    ax.set_title('Simulation Result with Car Orientation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    # ax.legend()
    ax.grid(True)
    ax.axis('equal')
    x_range = np.max(safe_zones[:,0]) - np.min(safe_zones[:,0])
    y_range = np.max(safe_zones[:,1]) - np.min(safe_zones[:,1])
    x_low = np.min(safe_zones[:,0])- x_range/5
    y_low = np.min(safe_zones[:,1])- y_range/2
    # plt.xlim([x_low, 4.5])
    # plt.ylim([y_low, 4.5])
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False)
    plt.pause(0.1)
    # plt.savefig('asdf.png')
    return ax
