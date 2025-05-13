import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm
import sys
from functools import partial
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_frame(frame, states, obs, safe_zones, all_total_costs, all_total_sampled_states, all_total_con_states, safe_hist=None, ):
    x_vals = [state[0] for state in states[:frame]]
    y_vals = [state[1] for state in states[:frame]]
    theta_vals = [state[2] for state in states[:frame]]

    fig = plt.figure(figsize=(6, 6))

    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        plt.gca().add_artist(rect)

    if safe_hist is not None and len(safe_hist)==len(states):
        not_safe_x = np.take(x_vals, np.nonzero(1-np.array(safe_hist[:frame])))
        not_safe_y = np.take(y_vals, np.nonzero(1-np.array(safe_hist[:frame])))
        plt.plot(not_safe_x, not_safe_y, 'ro',  markersize=6, alpha=1.0)

    # Plot the trajectory
    plt.plot(x_vals, y_vals, '-o', markersize=4, alpha=0.5)



    x, y, theta = states[frame-1][0:3]
    dx = 0.1 * np.cos(theta)
    dy = 0.1 * np.sin(theta)
    plt.arrow(x, y, dx, dy, head_width=1.0, head_length=1.0, fc='black', ec='black') 
    # sampled_states = np.insert(sampled_states, 0,states[frame][0:3], axis=0)
    # sampled_states = np.insert(sampled_states, 0,states[frame-1][0:3], axis=0)
    # breakpoint()
    sorted_idxs = np.argsort(all_total_costs[frame])

    mc_idx = np.nanargmin(all_total_costs[frame])    
    
    for i in range(3):
        mc_idx = sorted_idxs[i]
        sampled_states = all_total_sampled_states[frame][mc_idx]
        total_con_states = all_total_con_states[frame][mc_idx]
        plt.plot(sampled_states[:,0], sampled_states[:,1], '-ko')
        for i, seq in enumerate(total_con_states):
            # seq = np.insert(seq, 0, sampled_states[i][0:3], axis=0)
            plt.plot(seq[:,0], seq[:,1], 'g', alpha=0.5)

    # plt.scatter(states[0][0], states[0][1], s=200, color="green", alpha=0.75, label="Start")
    # plt.scatter(safe_zones[0][0], safe_zones[0][1], s=200, color="purple", alpha=0.75, label="Goal")
    # plt.legend()
    plt.grid(True)
    x_range = np.max(x_vals) - np.min(x_vals)
    y_range = np.max(y_vals) - np.min(y_vals)
    x_low = np.minimum(np.min(x_vals), np.min(safe_zones[:,0]))
    y_low = np.minimum(np.min(y_vals), np.min(safe_zones[:,1]))
    x_max = np.maximum(np.max(x_vals), np.max(safe_zones[:,0]))
    y_max = np.maximum(np.max(y_vals), np.max(safe_zones[:,1]))
    plt.xlim([x_low-1, x_max+1])
    plt.ylim([y_low-1, y_max+1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(0.5)
    return fig

def plot_cleaner(states, obs, safe_zones, total_con_states, safe_hist=None):
    x_vals = [state[0] for state in states]
    y_vals = [state[1] for state in states]
    theta_vals = [state[2] for state in states]

    fig = plt.figure(figsize=(6, 6))

    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        plt.gca().add_artist(rect)

    for i in range(0, len(states), int(len(states)/10)):  # Only plot 20 arrows for visibility
        x, y, theta = states[i][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=1.0, head_length=1.0, fc='black', ec='black')

    if safe_hist is not None and len(safe_hist)==len(states):
        not_safe_x = np.take(x_vals, np.nonzero(1-np.array(safe_hist)))
        not_safe_y = np.take(y_vals, np.nonzero(1-np.array(safe_hist)))
        plt.plot(not_safe_x, not_safe_y, 'ro',  markersize=6, alpha=1.0)

    # Plot the trajectory
    plt.plot(x_vals, y_vals, '-o', markersize=4, alpha=0.5)

    all_con_states = []
    for i in range(0, len(states), 1    ):
        found = False
        for con_states in total_con_states[i]:
            collided = False
            reached = False
            for state in con_states:
                if np.any(np.linalg.norm(state[:2] - obs[:,:2], axis=1) < np.sqrt(obs[:,2])):
                    collided =True
                    break
                if np.any(np.linalg.norm(state[:2] - safe_zones[:,:2], axis=1) < 0.4):
                    reached=True
            
            if reached and not collided:
                found = True
                all_con_states.append(con_states)
                break
    
    for con_states in all_con_states:
        plt.plot(con_states[:,0], con_states[:,1], 'g', alpha=0.5)
    plt.scatter(states[0][0], states[0][1], s=200, color="green", alpha=0.75, label="Start")
    plt.scatter(safe_zones[0][0], safe_zones[0][1], s=200, color="purple", alpha=0.75, label="Goal")
    # plt.legend()
    plt.grid(True)
    x_range = np.max(x_vals) - np.min(x_vals)
    y_range = np.max(y_vals) - np.min(y_vals)
    x_low = np.minimum(np.min(x_vals), np.min(safe_zones[:,0]))
    y_low = np.minimum(np.min(y_vals), np.min(safe_zones[:,1]))
    x_max = np.maximum(np.max(x_vals), np.max(safe_zones[:,0]))
    y_max = np.maximum(np.max(y_vals), np.max(safe_zones[:,1]))
    plt.xlim([x_low-1, x_max+1])
    plt.ylim([y_low-1, y_max+1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(0.5)
    return fig

def plot_simulation_result(states, obs, safe_zones, text="", max_arrows=30, safe_hist=None):
    """
    Plot the trajectory and orientation of the car given the state history.

    Parameters:
        states (list of np.array): List of states [x, y, theta, v] at each time step.
    """
    x_vals = [state[0] for state in states]
    y_vals = [state[1] for state in states]
    theta_vals = [state[2] for state in states]

    fig = plt.figure(figsize=(6, 6))

    # Generate circle for CBF
    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        plt.gca().add_artist(rect)

    # Plot the trajectory
    plt.plot(x_vals, y_vals, '-o', label='Trajectory', markersize=4, alpha=0.5)

    # Plot the orientation at each point
    for i in range(0, len(states), int(len(states)/max_arrows)):  # Only plot 20 arrows for visibility
        x, y, theta = states[i][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')
    
    if safe_hist is not None and len(safe_hist)==len(states):
        # breakpoint()
        not_safe_x = np.take(x_vals, np.nonzero(1-np.array(safe_hist)))
        not_safe_y = np.take(y_vals, np.nonzero(1-np.array(safe_hist)))
        # plt.plot
        plt.plot(not_safe_x, not_safe_y, 'ro',  markersize=4, alpha=0.5)


    # plot start and end point
    # plt.scatter(3.5, 3.5, s=200, color="green", alpha=0.75, label="init. position")
    plt.scatter(states[0][0], states[0][1], s=200, color="green", alpha=0.75, label="init. position")
    plt.scatter(safe_zones[0][0], safe_zones[0][1], s=200, color="purple", alpha=0.75, label="target position")

    plt.text(0.0,0.0, text, transform=plt.gca().transAxes,verticalalignment="bottom")

    plt.title('Simulation Result with Car Orientation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # plt.legend()
    plt.grid(True)
    # plt.axis('equal')
    # x_range = np.max(safe_zones[:,0]) - np.min(safe_zones[:,0])
    # y_range = np.max(safe_zones[:,1]) - np.min(safe_zones[:,1])
    # x_low = np.min(safe_zones[:,0])- x_range/2
    # y_low = np.min(safe_zones[:,1])- y_range/
    x_range = np.max(x_vals) - np.min(x_vals)
    y_range = np.max(y_vals) - np.min(y_vals)
    x_low = np.minimum(np.min(x_vals), np.min(safe_zones[:,0]))
    y_low = np.minimum(np.min(y_vals), np.min(safe_zones[:,1]))
    x_max = np.maximum(np.max(x_vals), np.max(safe_zones[:,0]))
    y_max = np.maximum(np.max(y_vals), np.max(safe_zones[:,1]))
    plt.xlim([x_low-1, x_max+1])
    plt.ylim([y_low-1, y_max+1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show(block=False)
    plt.pause(0.5)
    # plt.savefig('asdf.png')
    return fig

def get_predicted_trajectories(current_state, control_sequence, dynamics):
  predicted_state = [current_state]
  for ctrl in control_sequence:
    next_state = dynamics(predicted_state[-1], ctrl)
    predicted_state.append(next_state.copy())
  return np.array(predicted_state)

def animate_simulation(states, obs, safe_zones, sampled_us=[], optimal_us=None, dynamics=None):
    if dynamics is None:
        print("errorr")
    def update(frame, states):
        plt.gca().cla()  # Clear the current axes

        # Set axis limits
        x_low = np.min(safe_zones[:,0])- 1
        y_low = np.min(safe_zones[:,1])- 2
        plt.xlim([x_low, 4.5])
        plt.ylim([y_low, 4.5])

        # Set aspect ratio to be equal, so each cell will be square-shaped
        plt.gca().set_aspect('equal', adjustable='box')
        plt.text(0.0,0.0, f"Frame={frame}", transform=plt.gca().transAxes)

        # Generate circle for CBF
        # circle = plt.Circle((2, 2), np.sqrt(1), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        # plt.gca().add_artist(circle)
        for circ in obs:
            circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
            plt.gca().add_artist(circle)

        for zone in safe_zones:
            rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
            plt.gca().add_artist(rect)

        # Plot MPPI trajectories
        if len(sampled_us) > 0:
          # pred_trajs = [get_predicted_trajectories(states[frame], sampled_us[frame][i]) for i in range(len(sampled_us))]
          num_trajs_plotted = np.minimum(10,  sampled_us[frame].shape[0])  # plot maximum 50 trajs per step
          for idx in range(num_trajs_plotted):
            pred_traj = get_predicted_trajectories(states[frame], sampled_us[frame][idx], dynamics)
            x_pos, y_pos = pred_traj[:, 0], pred_traj[:, 1]
            plt.plot(x_pos, y_pos, color="k", alpha=0.1)

        # plot optimal predicted trajectory
        if optimal_us is not None:
          opt_pred_traj = get_predicted_trajectories(states[frame], optimal_us[frame], dynamics)
          x_pos, y_pos = opt_pred_traj[:, 0], opt_pred_traj[:, 1]
          plt.plot(x_pos, y_pos, color="orange", alpha=0.8, label="optimized traj.")


        x, y, theta  = states[frame][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)

        # Plot the trajectory up to the current frame
        plt.plot([state[0] for state in states[:frame+1]], [state[1] for state in states[:frame+1]], '-o', markersize=4, alpha=0.5)

        # Plot the orientation at the current frame
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')

        plt.scatter(3.5, 3.5, s=200, color="green", alpha=0.75, label="start")
        plt.scatter(safe_zones[0][0], safe_zones[0][1], s=200, color="purple", alpha=0.75, label="target position")

        plt.title('Simulation Result with Car Orientation')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.legend(loc="upper left")

    fig = plt.figure(figsize=(6, 6))
    anim = FuncAnimation(fig, update, frames=len(states), fargs=(states,), interval=100, blit=False)
    # anim.save('simulation.gif', writer='imagemagick')
    return anim

def animate_simulation_with_sampled_states(states, obs, safe_zones, sampled_xs=[], optimal_us=None, dynamics=None):
    if dynamics is None:
        print("errorr")
    def update(frame, states):
        plt.gca().cla()  # Clear the current axes
        # Set axis limits
        x_vals = [state[0] for state in states]
        y_vals = [state[1] for state in states]

        x_low = np.minimum(np.min(x_vals), np.min(safe_zones[:,0]))
        y_low = np.minimum(np.min(y_vals), np.min(safe_zones[:,1]))
        x_max = np.maximum(np.max(x_vals), np.max(safe_zones[:,0]))
        y_max = np.maximum(np.max(y_vals), np.max(safe_zones[:,1]))
        plt.xlim([x_low-1, x_max+1])
        plt.ylim([y_low-1, y_max+1])

        # x_low = np.min(safe_zones[:,0])-1
        # y_low = np.min(safe_zones[:,1])- 2
        # x_low = np.min(x_vals)
        # y_low = np.min(y_vals)
        # x_max = np.max(x_vals)
        # y_max = np.max(y_vals)
        # plt.xlim([x_low, x_max])
        # plt.ylim([y_low, y_max])
        # plt.xlim([x_low, 4.5])
        # plt.ylim([y_low, 4.5])
        # plt.xlim([-5, 4.5])
        # plt.ylim([-5, 4.5])

        # Set aspect ratio to be equal, so each cell will be square-shaped
        plt.gca().set_aspect('equal', adjustable='box')
        plt.text(0.0,0.0, f"Frame={frame}", transform=plt.gca().transAxes, verticalalignment="bottom")

        # Generate circle for CBF
        # circle = plt.Circle((2, 2), np.sqrt(1), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        # plt.gca().add_artist(circle)
        for circ in obs:
            circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
            plt.gca().add_artist(circle)

        for zone in safe_zones:
            rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
            plt.gca().add_artist(rect)

        # Plot MPPI trajectories
        # breakpoint()
        if len(sampled_xs) > 0:
          # pred_trajs = [get_predicted_trajectories(states[frame], sampled_us[frame][i]) for i in range(len(sampled_us))]
          num_trajs_plotted = np.minimum(100,  sampled_xs[frame].shape[0])  # plot maximum 50 trajs per step
          for idx in range(num_trajs_plotted):
            # pred_traj = get_predicted_trajectories(states[frame], sampled_us[frame][idx], dynamics)
            x_pos, y_pos = sampled_xs[frame][idx, :, 0], sampled_xs[frame][idx, :, 1]
            plt.plot(x_pos, y_pos, color="g", alpha=0.5)

        # plot optimal predicted trajectory
        if optimal_us is not None:
          opt_pred_traj = get_predicted_trajectories(states[frame], optimal_us[frame], dynamics)
          x_pos, y_pos = opt_pred_traj[:, 0], opt_pred_traj[:, 1]
          plt.plot(x_pos, y_pos, color="orange", alpha=0.8, label="optimized traj.")


        x, y, theta = states[frame][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)

        # Plot the trajectory up to the current frame
        plt.plot([state[0] for state in states[:frame+1]], [state[1] for state in states[:frame+1]], '-o', markersize=4, alpha=0.5)

        # Plot the orientation at the current frame
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')

        plt.scatter(states[0][0], states[0][1], s=200, color="green", alpha=0.75, label="init. position")
        plt.scatter(safe_zones[0][0], safe_zones[0][1], s=200, color="purple", alpha=0.75, label="target position")
        # plt.title('Simulation Result with Car Orientation')
        # plt.xlabel('X Position')
        # plt.ylabel('Y Position')
        plt.grid(True)
        # plt.legend(loc="upper left")

    fig = plt.figure(figsize=(6, 6))
    anim = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(states)), file=sys.stdout), fargs=(states,), interval=100, blit=False)
    # anim.save('simulation.gif', writer='imagemagick')
    return anim

def plot_simulation_result_with_sampled_states(states, obs, safe_zones, sampled_xs=[], text=""):
    """
    Plot the trajectory and orientation of the car given the state history.

    Parameters:
        states (list of np.array): List of states [x, y, theta, v] at each time step.
    """
    x_vals = [state[0] for state in states]
    y_vals = [state[1] for state in states]
    theta_vals = [state[2] for state in states]

    fig = plt.figure(figsize=(6, 6))

    # Generate circle for CBF
    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        plt.gca().add_artist(rect)

    # Plot the trajectory
    plt.plot(x_vals, y_vals, '-o', label='Trajectory', markersize=4, alpha=0.5)

    # Plot the orientation at each point
    for i in range(0, len(states), int(len(states)/30)):  # Only plot 20 arrows for visibility
        x, y, theta = states[i][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')

    # plot start and end point
    plt.scatter(3.5, 3.5, s=200, color="green", alpha=0.75, label="init. position")
    plt.scatter(safe_zones[0][0], safe_zones[0][1], s=200, color="purple", alpha=0.75, label="target position")

    plt.text(0.0,0.0, text, transform=plt.gca().transAxes,verticalalignment="bottom")

    plt.title('Simulation Result with Car Orientation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    # plt.axis('equal')
    x_range = np.max(safe_zones[:,0]) - np.min(safe_zones[:,0])
    y_range = np.max(safe_zones[:,1]) - np.min(safe_zones[:,1])
    x_low = np.min(safe_zones[:,0])- x_range/5
    y_low = np.min(safe_zones[:,1])- y_range/5
    plt.xlim([x_low, 4.5])
    plt.ylim([y_low, 4.5])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show(block=False)
    plt.pause(0.1)
    # plt.savefig('asdf.png')
    return fig
