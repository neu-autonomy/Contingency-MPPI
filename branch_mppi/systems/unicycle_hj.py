import jax.numpy as jnp
import jax

from hj_reachability import dynamics
from hj_reachability import sets


class Unicycle_HJ(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 obs=jnp.array([[0.0,0.0,0.0]]),
                 max_control = jnp.array([jnp.pi, 1.0]),
                 min_control = jnp.array([-jnp.pi, -0.1]),
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.obs = obs
        if control_space is None:
            control_space = sets.Box(max_control, min_control)
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0]), jnp.array([1]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, psi = state
        # return jnp.array([
        #     [0.],
        #     [0.],
        #     [0.],
        # ])
        return jnp.array([0.0,0.0,0.0])

    def control_jacobian(self, state, time):
        x, y, psi = state
        cpsi = jnp.cos(psi)
        spsi = jnp.sin(psi)

        obs_dist = jnp.linalg.norm((state[:2] - self.obs[:,:2]), axis=1)
        radius = jnp.sqrt(self.obs[:,2])
        in_collision = jax.lax.cond(jnp.min(obs_dist-radius) < 0, lambda x: 1, lambda x: 0, 0)
        # nominal =  jnp.array([[cpsi, -spsi, 0],
        #                         spsi, cpsi, 0,
        #                         0, 0, 1])
        nominal =  jnp.array([[0, cpsi],
                               [0, spsi],
                               [1, 0]])
        result = nominal* (1 - in_collision) + jnp.zeros((3,2,))
        return result

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [0.],
        ])

