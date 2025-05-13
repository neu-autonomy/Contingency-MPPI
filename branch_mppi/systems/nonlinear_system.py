"""Define an abstract base class for dymamical systems"""
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from typing import Callable, Tuple, Optional, List, Dict
from matplotlib.axes import Axes
import numpy as np
import jax.numpy as jnp


class NonlinerSystem(ABC):
    """
    Represents a general nonlinear system.

    Code structure taken from neural_clbf:
    @ARTICLE{dawson2022survey,
    author={Dawson, Charles and Gao, Sicun and Fan, Chuchu},
    journal={IEEE Transactions on Robotics}, 
    title={Safe Control With Learned Certificates: A Survey of Neural Lyapunov, Barrier, and Contraction Methods for Robotics and Control}, 
    year={2023},
    volume={},
    number={},
    pages={1-19},
    doi={10.1109/TRO.2022.3232542}}
    """

    def __init__(
        self,
        nominal_params: Dict[str, float],
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
    ):
        super().__init__()

        # Validate parameters, raise error if they're not valid
        # if not self.validate_params(nominal_params):
        #     raise ValueError(f"Parameters not valid: {nominal_params}")

        self.nominal_params = nominal_params

        # Make sure the timestep is valid
        # assert dt > 0.0
        self.dt = dt

        # if controller_dt is None:
        #     controller_dt = self.dt
        # self.controller_dt = controller_dt

    @abstractmethod
    def dynamics(self, state: np.array, control: np.array, t: float, dt: float, params: Dict[str, float]):
        """
        returns next state (non-jaxified)
        args:
            state
            control
            dt
            params
        returns:
            new state
        """
        pass

    @abstractmethod
    def jax_dynamics(self, state: jnp.array, control: jnp.array, t: float, dt: float, params: Dict[str, float]):
        pass

    @abstractmethod
    def ode(self, state: jnp.array, control: jnp.array, t: float, dt: float, params: Dict[str, float]):
        pass

    @abstractmethod
    def cas_ode(dt: float, params: Dict[str, float]):
        pass

    @abstractmethod
    def h_x(self, state: jnp.array, control: jnp.array):
        '''
        safety cbf
        args:
            state
            control
        returns:
            boolean
        '''
        pass
