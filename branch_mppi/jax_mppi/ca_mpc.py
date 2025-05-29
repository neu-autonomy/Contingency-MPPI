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

def cas_shooting_solver(system: NonlinerSystem, Nt, ode, ns=6, dt=None,solver="ipopt"):
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
    p_opts = {"expand": 1, "print_time": False, "verbose": False,  "jit": True, "compiler":"shell", "jit_options":jit_options,'jit_cleanup':True, 'jit_temp_suffix':False}

    if solver.lower() == "ipopt":
        s_opts = {"tol":1e-2, "max_iter": 1000, "mu_strategy":"adaptive", "print_level":0}
        opti.solver("ipopt", p_opts, s_opts)

    elif solver.lower() == "snopt":
        s_opts = {"Minor print level":0, 
                "Major print level":0, 
                "Summary file": 0, 
                "Solution": "no", 
                "Suppress options listings":1, 
                "Timing level":1,
                "Scale option": 2,
                "Major iterations limit":1000
                }
        opti.solver("snopt", p_opts, s_opts)

    else:
        opti.solver(solver.lower(),p_opts)
    return opti.to_function('F', [x, u, start, goal,  obs_A, obs_b, path], [x, u], ['x', 'u', 'start', 'goal', 'obs_A', 'obs_b', 'path'], ['x_opt', 'u_opt'])

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
