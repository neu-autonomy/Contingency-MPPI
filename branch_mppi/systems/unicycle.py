from typing import Tuple, Optional, List
from .nonlinear_system import NonlinerSystem
import jax.numpy as jnp
import jax
import numpy as np
import functools
import casadi as ca
from jax import tree_util
class Unicycle(NonlinerSystem):
    N_DIMS = 3
    N_CONTROLS = 2
    
    def __init__(
        self,
        nominal_params,
        dt: float = 0.1,
        controller_dt: Optional[float] = None,
    ):
        
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt,
        
        )
        self.lb = nominal_params['lb']
        self.ub = nominal_params['ub']
        self.grid = np.array([])

    def validate_params(self, params) -> bool:
        valid = True
        return valid

    @property
    def n_dims(self) -> int:
        return Unicycle.N_DIMS

    @property
    def n_controls(self) -> int:
        return Unicycle.N_CONTROLS

    @property
    def state_bounds(self) -> Tuple[jnp.array, jnp.array]:
        lb = jnp.array([-50, -50, -jnp.inf])
        ub = jnp.array([ 50,  50,  jnp.inf])
        return (lb, ub)

    @property
    def control_bounds(self) -> Tuple[jnp.array, jnp.array]:
        # lb = jnp.array([-jnp.pi, -0.1])
        # ub = jnp.array([jnp.pi, 1.0])
        return (self.lb, self.ub)
        # return (lb, ub)

    def calcSlipxy(self,Vr,Vx,Vy,Ty):
        """ calcSlipxy(self,Vr,Vx,Vy)
        returns: kappa, alpha
        kappa is longitudinal slip fraction
        alpha is lateral slip angle
        takes into account whether cornering or braking. SAE axis system on wheel center. z down, x forward, y right.
        Vr is the "omega*r_effective" velocity where omega is the rotational velocity of the wheel. positive is forward velocity.
        Therefore, if Vr>Vx, it means that the tire is accelerating. If Vr<Vx, it is braking. Thus, kappa>0 is acceleraing.
        driving (Ty>0 in SAE):
        k = -(1+k)*Vx/Vr 
        k(1+Vx/Vr) = -1
        k = -1/(1+Vx/Vr)
        braking (Ty<0 in SAE):
        k = -Vx/Vr for brake """
        # if Ty>0:
        #     kappa = -1/(1+Vx/Vr)
        #     alpha = np.arctan((1+kappa)*-Vy/Vr)
        # else:
        #     kappa = (Vr-Vx)/Vx
        #     alpha = np.arctan(-Vy/Vx)
        kappa = Ty * (-1/(1+Vx/Vr)) + (1-Ty) * (Vr-Vx)/Vx
        alpha = Ty * (np.arctan((1+kappa)*-Vy/Vr)) + (1-Ty) * np.arctan(-Vy/Vx)
        return kappa,alpha
    

    def calcFxFy(self, Fz,kappa,alpha,mu,Ca,Ck,Ty):
        """ calcFxFy(Fz,kappa,alpha,mu,Ca,Ck,Ty=0):
            returns: Fx,Fy 
            -Uses Dugoff tire model with isotropic friction (one mu)
            -Does not explicitly account for changes in cornering stiffness with load
            -Does not account for changes in Mu with load. You can program these in or maybe we'll add them.
            """
        # if Ty<=0:
        #     sigy = np.tan(alpha)
        #     sigx = kappa
        # else:
        #     sigy = np.tan(alpha)/(1+kappa)
        #     sigx = kappa/(1+kappa)

        sigy = Ty * np.tan(alpha) + (1-Ty) * np.tan(alpha)/(1+kappa)
        sigx = Ty * kappa + (1-Ty) * kappa/(1+kappa)
        Fxlin = Ck*sigx
        Fylin = Ca*sigy

        lam = (mu*Fz/2)/np.sqrt(Fxlin**2+Fylin**2) #lambda parameter
        lin = jax.lax.cond(np.sqrt(Fxlin**2+Fylin**2)<=(mu*Fz/2), lambda x: 1, lambda x: 0, 0)
        Fx = lin * Fxlin + (1-lin) * Fxlin*(2*lam-lam**2) 
        Fy = lin * Fylin + (1-lin) * Fylin*(2*lam-lam**2) 
        # if np.sqrt(Fxlin**2+Fylin**2)<=(mu*Fz/2):
        #     Fx = Fxlin
        #     Fy = Fylin
        # else:
        #     lam = (mu*Fz/2)/np.sqrt(Fxlin**2+Fylin**2) #lambda parameter
        #     Fx = Fxlin*(2*lam-lam**2)
        #     Fy = Fylin*(2*lam-lam**2)
        return Fx,Fy

    def dynamics(self, state, control, t=0, dt=None, params=None):
        """
        simplified kinematic unicycle car dynamics
        """
        if params is None:
            params = self.nominal_params
        if dt is None:
           dt = self.dt 
        x, y, theta  = state
        ct = np.cos(theta)
        st = np.sin(theta)


        control = np.clip(control, self.control_bounds[0], self.control_bounds[1])
        delta, v = control
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        # dx = v * np.cos(theta)
        # dy = v * np.sin(theta)
        dtheta = delta # rad/s
        next_state =np.clip(state + np.array([dx, dy, dtheta])*dt, self.state_bounds[0], self.state_bounds[1])
        return next_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def jax_dynamics(self, state, control, t=0, dt=None, params=None):
        """
        Simplified kinematic unicycle car dynamics with JAX-compatible code.
        """
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
        x, y, theta = state
        control = jnp.clip(control, self.control_bounds[0], self.control_bounds[1])
        delta, v = control

        # Ty = jax.lax.cond((v> 0), lambda x: 1, lambda x: 0, 0)
        # kappa, alpha = self.calcSlipxy(v/0.1,v*np.cos(dtheta),v*np.sin(dtheta),Ty)
        # Fx, Fy = self.calcFxFy(20, kappa, alpha, 0.1, 100000, 1000000, Ty)
        
        # dx = v * jnp.cos(theta) 

        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = delta 
        next_state = jnp.clip(state + jnp.array([dx, dy, dtheta])*dt, self.state_bounds[0], self.state_bounds[1])
        return next_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def ode(self, state, control, t=0, dt=None, params=None):
        """
        Simplified kinematic unicycle car dynamics with JAX-compatible code.
        """
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
        x, y, theta = state
        control = jnp.clip(control, self.control_bounds[0], self .control_bounds[1])
        delta, v = control
        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = delta
        # return state+jnp.array([dx, dy, dtheta])*dt
        return jnp.array([dx, dy, dtheta])

    
    
    def cas_ode(self: NonlinerSystem, dt=None, params=None):
        """
        Simplified kinematic unicycle car dynamics with JAX-compatible code.
        """
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
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
        dxdt[2] = delta

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
    
    # def h_x(self, state: jnp.array, map_w: jnp.array, origin: jnp.array, resolution: float):
    #     floor_state = jnp.floor(state[:2]/resolution).astype(jnp.int32)
    #     return map_w[floor_state[0] + origin[0], floor_state[1]+origin[1]]

    def h_x(self, state: jnp.array, obs: np.array):
        delta = state[:2] - jnp.array(obs[:2])
        return jnp.vdot(delta, delta) - obs[2]


    @staticmethod
    @jax.jit
    def diff_flat_map(s, M,a, tf):
        pass
    @staticmethod
    def scannable_diff_flat_map(carry, s):
        pass

    @staticmethod
    @jax.jit
    def get_control_hist_scan(s_seq, M, a, tf):
        pass

    @staticmethod
    @jax.jit
    def get_control_hist(M,a,tf, s_seq):
        pass

def flatten_unicycle(obj):
    children = (obj.nominal_params,obj.dt,)
    aux_data = {}
    return children, aux_data

def unflatten_unicycle(aux_data, children):
    obj = Unicycle(children[0], children[1])
    return obj

tree_util.register_pytree_node(Unicycle, flatten_unicycle, unflatten_unicycle)