from typing import Tuple, Optional, List, Dict
from .nonlinear_system import NonlinerSystem
import jax.numpy as jnp
import jax
import numpy as np
import casadi as ca
import functools

class HalfCar(NonlinerSystem):
    '''
    Represents a pointmass with 
    The system has state

        x = [x, y, theta]

    representing the position and heading of the unicycle, and it
    has control inputs

        u = [Delta, vel]

    representing the steering angle and velocity
    '''
    N_DIMS = 3
    N_CONTROLS = 2
    
    def __init__(
        self,
        nominal_params: Dict[str, float],
        dt: float = 0.1,
        controller_dt: Optional[float] = None,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_l1_norm: if True, use L1 norm for safety zones; otherwise, use L2
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt,
        )

    def validate_params(self, params: Dict[str, float]) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        valid = valid and "L" in params
        return valid

    @property
    def n_dims(self) -> int:
        return HalfCar.N_DIMS

    @property
    def n_controls(self) -> int:
        return HalfCar.N_CONTROLS

    @property
    def state_bounds(self) -> Tuple[jnp.array, jnp.array]:
        lb = np.array([-50, -50, -jnp.inf])
        ub = np.array([ 50,  50,  jnp.inf])
        return (lb, ub)

    @property
    def control_bounds(self) -> Tuple[jnp.array, jnp.array]:
        lb = np.array([-jnp.pi/3, 0.0])
        ub = np.array([jnp.pi/3, 4.0])
        # lb = jnp.array([-jnp.pi/2, 0.0])
        # ub = jnp.array([jnp.pi/2, 4.0])
        return (lb, ub)

    def dynamics(self, state, control, t=0, dt=None, params=None):
        """
        simplified kinematic bicycle car dynamics

        The state of the car is q = (x, y, theta, v), where (x, y) is the position,
        theta is the heading angle, and v is the velocity. The control input is
        the steering angle delta and the acceleration a. The dynamics are given by

          x' = v cos(theta)
          y' = v sin(theta)
          theta' = v tan(delta) / L
        #   v' =a
        """
        if params is None:
            params = self.nominal_params
        if dt is None:
           dt = self.dt 
        L = params["L"]
        x, y, theta  = state
        control = np.clip(control, self.control_bounds[0], self.control_bounds[1])
        delta, v = control
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v*np.tan(delta)/L
        next_state =np.clip(state + np.array([dx, dy, dtheta])*dt, self.state_bounds[0], self.state_bounds[1])
        return next_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def jax_dynamics(self, state, control, t=0, dt=None, params=None):
        """
        Simplified kinematic bicycle car dynamics with JAX-compatible code.
        """
        # jax.debug.breakpoint()
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
        L = params["L"]
        x, y, theta = state
        control = jnp.clip(control, self.control_bounds[0], self.control_bounds[1])
        delta, v = control
        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = v*jnp.tan(delta)/L
        next_state = jnp.clip(state + jnp.array([dx, dy, dtheta])*dt, self.state_bounds[0], self.state_bounds[1])
        return next_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def ode(self, state, control, t=0, dt=None, params=None):
        """
        Simplified kinematic bicycle car dynamics with JAX-compatible code.
        """
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
        L = params["L"]
        x, y, theta = state
        control = jnp.clip(control, self.control_bounds[0], self .control_bounds[1])
        delta, v = control
        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = v*jnp.tan(delta)/L
        # return state+jnp.array([dx, dy, dtheta])*dt
        return jnp.array([dx, dy, dtheta])
    
    def cas_ode(self: NonlinerSystem, dt=None, params=None):
        """
        Simplified kinematic bicycle car dynamics with JAX-compatible code.
        """
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
        L = params["L"]
        state = ca.MX.sym('state', 3,1)
        control = ca.MX.sym('control',2,1)
        # dxdt = ca.SX.sym('dxdt',3,1)
        dxdt = ca.MX.zeros(3,1)
        x = state[0]
        y = state[1]
        theta = state[2]
        # delta = control[0]
        # v = control[1]
        delta = ca.fmax(ca.fmin(control[0], 
                                self.control_bounds[1][0]), 
                                self.control_bounds[0][0])
        v = ca.fmax(ca.fmin(control[1], 
                                self.control_bounds[1][1]), 
                                self.control_bounds[0][1])
        dxdt[0] = v * ca.cos(theta)
        dxdt[1] = v * ca.sin(theta)
        dxdt[2] = v*ca.tan(delta)/L
        # return ca.Function('ode', [state, control], [dxdt]), state, control
        return dxdt, state, control

    def safe_mask(self, state: jnp.array, control: jnp.array):
        '''
        returns if current state/control is safe
        args:
            state
            control
        returns:
            boolean
        '''
        pass

    def h_x(self, state: jnp.array, obs: np.array):
        delta = state[:2] - jnp.array(obs[:2])
        return jnp.vdot(delta, delta) - obs[2]