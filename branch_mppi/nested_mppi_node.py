import rclpy
from branch_mppi.jax_mppi.nested_mppi_planners import MPPI_Planner_Occup
from rclpy.node import Node
from branch_mppi.systems import Unicycle
from branch_mppi.jax_mppi.topo_prm import TopoPRM
from branch_mppi.jax_mppi.ca_mpc import find_Nonlin_Controls, cas_shooting_solver
from branch_mppi.jax_mppi.lqg import iLQRSolver
from typing import Callable, Optional
from jax._src.typing import Array, ArrayLike
import jax
import casadi as ca
import functools
import jax.numpy as jnp
import numpy as np
from tf2_py._tf2_py import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Pose, Quaternion, TransformStamped, Twist, Vector3, Polygon
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.time import Time
from rclpy.timer import Timer
from tf2_geometry_msgs import do_transform_pose_stamped
from rclpy.logging import get_logger
import multiprocessing as mp
from rclpy.lifecycle import State, TransitionCallbackReturn, Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_msgs.msg import ColorRGBA, Header, Float32MultiArray
import time
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
import copy
import cdd
from std_srvs.srv import Trigger



def cfg_rviz_mkr(
    m_id: int,
    m_frame_id: str,
    m_type: int = Marker.LINE_STRIP,
    m_action: int = Marker.ADD,
    m_position: Point = Point(x=0.0, y=0.0, z=0.0),
    m_orientation: Quaternion = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    m_scale: Vector3 = Vector3(x=0.01, y=0.0, z=0.0),
    m_duration: Duration = Duration(seconds=1).to_msg(),
    m_FrameLocked: bool = False,
) -> Marker:
    msg = Marker()
    msg.header.frame_id = m_frame_id
    msg.id = m_id
    msg.type = m_type
    msg.action = m_action
    msg.pose.position = m_position
    msg.pose.orientation = m_orientation
    msg.scale = m_scale
    msg.lifetime = m_duration
    msg.frame_locked = m_FrameLocked
    return msg

class MPPINode(Node):
    def __init__(self):
        super().__init__('mppi_node')

        self.log = self.get_logger()

        self.declare_parameters(
        namespace='',
        parameters=[
            ('n_samples', rclpy.Parameter.Type.INTEGER),
            ('n_mini', rclpy.Parameter.Type.INTEGER),
            ('Nt', rclpy.Parameter.Type.INTEGER),
            ('N_safe', rclpy.Parameter.Type.INTEGER),
            ('N_mini', rclpy.Parameter.Type.INTEGER),
            ('num_anci', rclpy.Parameter.Type.INTEGER),
            ('mpc_ratio', rclpy.Parameter.Type.INTEGER),
            ('dt', rclpy.Parameter.Type.DOUBLE),
            ('controller_dt', rclpy.Parameter.Type.DOUBLE),
            ('V_max', rclpy.Parameter.Type.DOUBLE),
            ('w_max', rclpy.Parameter.Type.DOUBLE),
            ('V_min', rclpy.Parameter.Type.DOUBLE),
            ('w_min', rclpy.Parameter.Type.DOUBLE),
            ('V_var', rclpy.Parameter.Type.DOUBLE),
            ('w_var', rclpy.Parameter.Type.DOUBLE),
            ('temp', rclpy.Parameter.Type.DOUBLE),
            ('Seed', rclpy.Parameter.Type.INTEGER),
            ('R', rclpy.Parameter.Type.DOUBLE_ARRAY),
            ('Q', rclpy.Parameter.Type.DOUBLE_ARRAY),
            ('QT', rclpy.Parameter.Type.DOUBLE_ARRAY),
            ('Map_Frame', rclpy.Parameter.Type.STRING),
            ('Robot_Frame', rclpy.Parameter.Type.STRING),
            ('Odom_Frame', rclpy.Parameter.Type.STRING),
            ('Solver', rclpy.Parameter.Type.STRING),
            ('Tolerance', rclpy.Parameter.Type.DOUBLE),
            ('Radius', rclpy.Parameter.Type.DOUBLE),
            ('Debug', rclpy.Parameter.Type.BOOL)
        ]
        )

        self.first = True
        self.stop = True
        self.plan = None
        self.contingency_plan = None
        self.iteration = 0
        self.plan_i = 0
        self.planned_time = None
        self.rollout_markers: list[Marker] = []
        self.safe_zones_markers: list[Marker] = []
        q_ref = np.array([7.0, -0.5, 0.0],)  # Reference state
        # self.safe_zones = np.array([q_ref,
        #                [0.5, 0.0, 0.0],
        #                [2.0, 1.0, 0.0],
        #                [2.0, -1.5, 0.0],
        #                [3.0, -1.5, 0.0],
        #                [3.5, -1.5, 0.0],
        #                [4.0, 1.5, 0.0],
        #                [5.0, -1.5, 0.0],
        #                [6.0, -1.5, 0.0],
        #                 ])
        self.safe_zones = np.array([q_ref,
                       [0.5, 1.0, 0.0],
                       [0.75, 0.0, 0.0],
                    #    [2.0, 1.0, 0.0],
                       [2.0, -1.5, 0.0],
                    #    [3.0, -1.5, 0.0],
                       [3.5, -1.5, 0.0],
                    #    [4.0, 1.5, 0.0],
                       [5.0, -1.5, 0.0],
                       [6.0, -1.5, 0.0],
                        ])
        # theta = np.pi/2
        # t = np.array([-4.6,-3.0])
        theta = 0.0
        t = np.array([-0.5,-1.0])
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s],
                    [s, c]])
        for sz in self.safe_zones:
            sz[:2] = R @ (sz[:2] + t)
        # self.safe_zones = R self.safe_zones - np.array([0.0, 1.0,0.0])
        # self.safe_zones = np.array([[-15.0, 0.0,0.0],
        #                [4.0, 0.0, 0.0],
        #                [2.0, -3.0, 0.0],
        #                [3.0, -3.0, 0.0],
        #                [3.5, -3.0, 0.0],
        #                [5.0, -3.0, 0.0],
        #                [6.0, -3.0, 0.0],
        #                [7.0, -2.0, 0.0],planned_time
        #                [2.0, -0.5, 0.0],
        #                [1.0, -1.0, 0.0], ])

        self.PARAM_CHANGED_FLAG: bool = True
        self.state_seq = None
        self.state_seq_odom = None
        self.sampled_states = []
        self.safe_sampled_states = []
        self.topo_paths = []
        self.x_sol = []
        self.contingency_states = []        
        self.polyhedrals = []
        self.use_mpc = True
        self.wait = True
        self.trigger_time = None


        # MPPI state variables.
        self.costmap: Optional[ArrayLike] = None # Occupied Cells in ecll frame (cell, cell)
        self.occupied: Optional[ArrayLike] = None # Occupied cells in world frame (m, m)
        self.res: Optional[float] = None
        self.origin: Optional[tuple[int, int]] = None
        self.wh: Optional[float] = None
        self.goal_msg: Optional[PoseStamped] = None
        self.goal_local_msg = None
        self.local_safe_zones  = None
        self.local_position = None
        self.ctrl_vel: list[float] = [0.0, 0.0]
        self.topo_planner = None
        self.mppi_controller = None
        self.mpc_solver = None
        self.log.info("[+] MPPI Node Initialized")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.ros_param_callback()
        self.add_on_set_parameters_callback(self._set_changed_param_flag)
        self.goal_tf_buffer: Buffer = Buffer()
        # self.goal_tf_listener = TransformListener(
        #     self.goal_tf_buffer, self, spin_thread=False)
        self.goal_tf_listener = TransformListener(
            self.goal_tf_buffer, self)
        self.publish_dt = 0.02
        # self.U_prev: ArrayLike = np.kron(np.ones((1,self.Nt)), [0.0,self.nlmodel.control_bounds[1][1]]).ravel()

        self.U_prev: ArrayLike = np.kron(np.ones((1,self.Nt)), [0.0,0.0]).ravel()

        self.con_trigger_service = self.create_service(Trigger, 'AAAAHHH', self.con_service_cb)

        self.cmd_vel_pub= self.create_publisher(Twist, "cmd_vel", 10)

        self.decomp_pub = self.create_publisher (
            MarkerArray, "decomp", 10
        )

        self.topo_paths_pub = self.create_publisher (
            MarkerArray, "topo_paths", 10
        )

        self.contingency_markers_pub = self.create_publisher (
            MarkerArray, "contingency_states", 10
        )

        self.anci_markers_pub = self.create_publisher(
            MarkerArray, "anci_states", 10
        )

        self.safe_big_guy_markers_pub = self.create_publisher(
            MarkerArray, "safe_sampled_states", 10
        )

        self.big_guy_markers_pub = self.create_publisher(
            MarkerArray, "sampled_states", 10
        )

        self.rollouts_marker_pub = self.create_publisher(
            MarkerArray, "rollout_state", 10
        )
        self.safe_zones_markers_pub = self.create_publisher(
            MarkerArray, "safe_zones", 10
        )

        self.local_goal_pub = self.create_publisher(
            PoseStamped, "goal", 10
        )
        self.masked_map_pub = self.create_publisher(
            OccupancyGrid, "masked_map", 10
        )
        self.costs_pub = self.create_publisher(Float32MultiArray, 'all_costs', 10)
        self.all_cont_pub = self.create_publisher(Float32MultiArray, 'all_cont', 10)

        self.config_change_timer: Timer = self.create_timer(
            1.0, self.ros_param_callback
        )

        pub_callback_group = MutuallyExclusiveCallbackGroup()
        map_callback_group = MutuallyExclusiveCallbackGroup()
        plan_callback_group = MutuallyExclusiveCallbackGroup()
        # plan_callback_group =ReentrantCallbackGroup()
        tf_callback_group = MutuallyExclusiveCallbackGroup()
        # self.local_goal_timer: Timer = self.create_timer(1e-2, self.publish_local_goal)
        # self.marker_timer: Timer = self.create_timer(1e-3, self.publish_markers)
        # self.replan_timer: Timer = self.create_timer(0.1, self.create_plan, callback_group=plan_callback_group)
        # self.cmd_vel_timer: Timer = self.create_timer(self.publish_dt, self.publish_ctrl)
        # self.cmd_vel_timer: Timer = self.create_timer(0.05, self.publish_ctrl, callback_group=pub_callback_group)
        self.update_tf_timer: Timer = self.create_timer(0.05, self.tf_timer_cb, callback_group=tf_callback_group)
        self.create_subscription(Odometry, "odom", self.publish_ctrl, 1, callback_group=pub_callback_group)
        self.create_subscription(Odometry, "odom", self.create_plan, 10, callback_group=plan_callback_group)

        # self.create_subscription(OccupancyGrid, "occupancy_grid", self.create_plan, 1, callback_group=plan_callback_group)
        self.create_subscription(OccupancyGrid, "occupancy_grid", self.map_callback, 1, callback_group=map_callback_group)
        self.create_subscription(Odometry, "odom", self.odom_callback, 5)
        self.create_subscription(PoseStamped, "goal_pose", self.goal_callback, 5)
        # self.create_subscription(PoseStamped, "/dlio/odom_node/pose", self.pose_callback, 10)
        self.get_logger().info(f"Node '{self.get_name()}' is in state '{state.label}'. Transitioning to 'configure'")

 
        
        self.topo_path_markers = [
                cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id=self.ODOM_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.02, y=0.0, z=0.0),
                )
        ]

        self.contingency_markers = [
                cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id=self.ODOM_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.005, y=0.0, z=0.0),
                )
        ]

        self.anci_markers = [
                cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id=self.ROBOT_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.05, y=0.0, z=0.0),
                )
        ]

        self.big_guy_markers = [
                cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id=self.ODOM_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.0025, y=0.0, z=0.0),
                )
        ]

        self.safe_big_guy_markers = [
                cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id=self.ODOM_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.005, y=0.0, z=0.0),
                )
        ]

        self.rollout_markers = [
                cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id=self.ODOM_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.05, y=0.0, z=0.0),
                )
            ]
        self.safe_zones_markers = [
                cfg_rviz_mkr(
                    m_id=1,
                    m_frame_id=self.ODOM_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.25, y=0.25, z=0.25),
                    m_type=Marker.SPHERE_LIST
                )
        ]

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(f"Node '{self.get_name()}' is in state '{state.label}'. Transitioning to 'Activate'")
        U = np.kron(np.ones((1,self.Nt)), [0.0,self.nlmodel.control_bounds[1][1]]).ravel()
        U_anci = np.tile(U, [self.num_anci, 1])
        start = np.array([0.0,0.0,0.0])
        goal = np.array([1.0,0.0,0.0])
        safe_zones = self.safe_zones
        wh = np.array([80,80])
        costmap = np.zeros((wh[0],wh[1]))
        res = 0.1
        origin = -res/2*np.array(wh)
        # self.mppi_controller.mppi_mmodal(
        #                                 state=start,
        #                                 U_original=U,
        #                                 U_anci=U_anci,
        #                                 rng_key=jax.random.PRNGKey(0),
        #                                 q_ref=goal,
        #                                 safe_zones=safe_zones,
        #                                 cost_map=costmap,
        #                                 resolution=res,
        #                                 origin=origin,
        #                                 wh=wh)
        return TransitionCallbackReturn.SUCCESS

    def ros_param_callback(self) -> None:
        if not self.PARAM_CHANGED_FLAG:
            return
        try:
            self.PARAM_CHANGED_FLAG = False
            self.n_samples: int = self.get_parameter("n_samples").value  # type: ignore
            self.n_mini: int = self.get_parameter("n_mini").value  # type: ignore
            self.num_anci: int = self.get_parameter("num_anci").value  # type: ignore
            self.Nt: int = self.get_parameter("Nt").value  # type: ignore
            self.N_safe: int = self.get_parameter("N_safe").value  # type: ignore
            self.N_mini: float = self.get_parameter("N_mini").value  # type: ignore
            self.dt: float = self.get_parameter("dt").value  # type: ignore
            self.controller_dt: float = self.get_parameter("controller_dt").value

            V_max = self.get_parameter("V_max").value
            w_max = self.get_parameter("w_max").value
            V_min = self.get_parameter("V_min").value
            w_min = self.get_parameter("w_min").value
            lb = [w_min, V_min]
            ub = [w_max, V_max]

            radius = self.get_parameter("Radius").value
            diag = np.sqrt(radius*radius/2)
            self.footprint = np.array([[radius, 0], 
                          [0,radius],
                          [-radius,0],
                          [0,-radius],
                          [diag,diag],
                          [diag,-diag],
                          [-diag,-diag],
                          [-diag,diag],
                          ])

            self.mpc_ratio = self.get_parameter("mpc_ratio").value
            self.temperature = self.get_parameter("temp").value
            V_var: float = self.get_parameter("V_var").value
            w_var: float = self.get_parameter("w_var").value
            sigma0 = np.diag(np.array([V_var, w_var]))
            self.sigma0 = np.kron(np.eye(self.Nt), sigma0)

            self.SEED: int = jax.random.PRNGKey(self.get_parameter("Seed").value)  # type: ignore
            R = np.reshape(self.get_parameter("R").value, (2,2))
            self.R = R
            Q = np.reshape(self.get_parameter("Q").value, (3,3))
            self.Q = Q
            QT = np.reshape(self.get_parameter("QT").value, (3,3))
            self.QT = QT

            self.MAP_FRAME: str = self.get_parameter("Map_Frame").value  # type: ignore
            self.ODOM_FRAME: str = self.get_parameter("Odom_Frame").value  # type: ignore
            self.ROBOT_FRAME: str = self.get_parameter("Robot_Frame").value  # type: ignore
            self.TOLERANCE: float = self.get_parameter("Tolerance").value  # type: ignore
            self.DEBUG: bool = self.get_parameter("Debug").value  # type: ignore
            self.solver= self.get_parameter("Solver").value

            self.topo_planner = TopoPRM(None, max_raw_path2=self.num_anci, 
                                        sample_sz_p=0.25, 
                                        ratio_to_short=3,
                                        footprint=self.footprint, 
                                        # occup_value=100
                                        occup_value=10
                                        # occup_value=0.0
                                        )

            self.nlmodel = Unicycle({"lb":jnp.array(lb), 
                                     "ub":jnp.array(ub) }, 
                                     dt=self.dt, 
                                     controller_dt=self.controller_dt)
            self.mppi_controller = MPPI_Planner_Occup(sigma=self.sigma0,
                                                Q=Q,
                                                QT=QT,
                                                R=self.R,
                                                temperature=self.temperature,
                                                system=self.nlmodel,
                                                num_anci = self.num_anci,
                                                n_samples=self.n_samples,
                                                n_mini=self.n_mini,
                                                N=self.Nt,
                                                N_mini=self.N_mini,
                                                N_safe=self.N_safe,
                                                max_sz=20,  # Note  obselete
                                                tolerance=self.TOLERANCE,
                                                # occup_value=[100]
                                                footprint=self.footprint,
                                                # occup_value=50
                                                occup_value=100,
                                                alpha=1.0
                                                )
            dxdt, state, control = self.nlmodel.cas_ode()
            ode = ca.Function('ode', [state, control], [dxdt])
            self.mpc_solver = cas_shooting_solver(self.nlmodel, 
                            int(self.Nt/self.mpc_ratio), 
                            ns=10, 
                            dt=self.nlmodel.dt*self.mpc_ratio, 
                            ode=ode,
                            solver=self.solver)
            self.ilqr = iLQRSolver(self.nlmodel, self.nlmodel.dt, 
                               jnp.array([[3.0, 0.0, 0.0],
                                          [0.0, 3.0, 0.0],
                                          [0.0, 0.0, 10.0]
                                            ]), 
                                jnp.array([[1.0, 0.0],
                                          [0.0, 1.0]]), 
                               jnp.array([[3.0, 0.0, 0.0],
                                          [0.0, 3.0, 0.0],
                                          [0.0, 0.0, 10.0]]),
                                max_iter=10)
            self.q = mp.Queue()
        except Exception as e:
            print(e)
        return

    def _set_changed_param_flag(self, _: list[Parameter]) -> SetParametersResult:
        self.PARAM_CHANGED_FLAG = True
        return SetParametersResult(successful=True)
    
    def get_local_sz(self):
        # to_frame: str = self.ROBOT_FRAME
        to_frame: str = self.ODOM_FRAME
        local_safe_zones = None
        from_frame: str = self.MAP_FRAME

        local_safe_zones = []
        request_time = Time()
        t: TransformStamped = self.goal_tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                request_time,
                Duration(seconds=1.0))
        for safe_zone in self.safe_zones:
            msg_header = Header(frame_id=self.MAP_FRAME, stamp=self.get_clock().now().to_msg())
            safe_zone_msg = PoseStamped(header=msg_header, 
                                        pose=Pose(position=Point(x=safe_zone[0], y=safe_zone[1])))
            # safe_zone_msg = self.goal_tf_buffer.transform(safe_zone_msg, self.ROBOT_FRAME)
            safe_zone_msg = do_transform_pose_stamped(safe_zone_msg, t)
            local_safe_zones.append([safe_zone_msg.pose.position.x, 
                                        safe_zone_msg.pose.position.y,0.0])
        self.local_safe_zones = np.array(local_safe_zones)
        # try:
        #     t: TransformStamped = self.goal_tf_buffer.lookup_transform(
        #         to_frame,
        #         from_frame,
        #         Time(clock_type=self.get_clock().clock_type),
        #     )
        #     local_safe_zones = []
        #     for safe_zone_msg in safe_zone_msgs:
        #         local_safe_zone = do_transform_pose_stamped(
        #                     pose=safe_zone_msg, transform=t
        #         )
        #         local_safe_zones.append([local_safe_zone.pose.position.x, 
        #                                     local_safe_zone.pose.position.y,0.0])
        #     local_safe_zones = jnp.array(local_safe_zones)
            
        # except TransformException as e:
        #     self.log.error(f"[-] TF Lookup Error: {e}")
        # return np.array(local_safe_zones)
        
    def get_local_goal(self) -> Optional[ArrayLike]:
        if self.goal_msg is None:
            self.log.info(f"[+] Waiting for a goal...", once=True)
            return None

        # to_frame: str = self.ROBOT_FRAME
        to_frame: str = self.ODOM_FRAME
        from_frame: str = self.goal_msg.header.frame_id
        try:
            # request_time = Time()
            request_time = self.get_clock().now()
            t: TransformStamped = self.goal_tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                request_time,
                Duration(seconds=0.5)
            )
            if self.DEBUG:
                last_time: Time = Time(clock_type=self.get_clock().clock_type).from_msg(
                    t.header.stamp
                )
                Time_diff: float = (
                    self.get_clock().now() - last_time
                ).nanoseconds * 1e-6
                self.log.warn(
                    f"[~] The Latest transform between {self.goal_msg.header.frame_id} and {self.ODOM_FRAME} frame was {Time_diff:.5f} ms. ago, requested at {request_time.to_msg()}"
                )
            self.goal_local_msg = do_transform_pose_stamped(
                pose=self.goal_msg, transform=t
            )
            # goal = np.array([self.goal_local_msg.pose.position.x,
            #                 self.goal_local_msg.pose.position.y,
            #                 0
            # ])

            # start = np.array([self.local_position.pose.pose.position.x,
            #                  self.local_position.pose.pose.position.y,
            #             ])
            # if np.linalg.norm(goal[:2]-start[:2]) <= self.TOLERANCE:
            #     if not self.stop:
            #         self.log.info("[+] Goal Reached")
            #         self.stop = True

        except TransformException as e:
            self.log.error(f"[-] TF Lookup Error: {e}")

    # def get_local_pos(self):
    #     to_frame: str = self.ODOM_FRAME
    #     from_frame: str = self.ROBOT_FRAME
    #     request_time = Time()
    #     t: TransformStamped = self.goal_tf_buffer.lookup_transform(
    #         to_frame,
    #         from_frame,
    #         request_time,
    #     )
    #     pose = PoseStamped(header=Header(frame_id=self.ROBOT_FRAME, stamp=Time().to_msg())
    #                        )
    #     self.local_position = do_transform_pose_stamped(
    #         pose=pose, transform=t
    #     )

    def publish_local_goal(self) -> None:
        if self.goal_local_msg is not None:
            self.local_goal_pub.publish(self.goal_local_msg)
        return

    @functools.partial(jax.jit, static_argnums=(0,))
    def scan_controls(self, sim_state, u):
        sim_state = self.nlmodel.jax_dynamics(sim_state, u, 0, self.nlmodel.dt, self.nlmodel.nominal_params)
        return sim_state, sim_state

    # def create_plan(self): 
    def create_plan(self, msg):
        # self.log.info("[+] Creating Plan")
        start_time = time.time()
        if self.wait:
            return
        if self.stop:
            # self.publish_ctrl()
            return


        if (self.local_safe_zones is None):
            print("[+] Trying to create plan, but local_safe_zones is None")
            return
        else:
            safe_zones = self.local_safe_zones

        if (self.local_position is None):
            print("[+] Trying to create plan, but local_position is None")
            return
        else:
            q = self.local_position.pose.pose.orientation
            yaw = (np.arctan2(2.0*(q.y*q.x + q.w*q.z), 1.0 - 2.0 * (q.z*q.z + q.x*q.x)))
            start = np.array([self.local_position.pose.pose.position.x,
                             self.local_position.pose.pose.position.y,
                             yaw])
        if (self.goal_local_msg is None):
            print("[+] Trying to create plan, but goal_local_msg is None")
            # return
            goal = start+np.random.uniform()*1e-5
        else:
            goal = np.array([self.goal_local_msg.pose.position.x,
                            self.goal_local_msg.pose.position.y,
                            0
            ])

        # if(yaw < 0):
        #     yaw +=2*np.pi
        # yaw = -yaw + 2*np.pi
        # start=np.array([0.0,0.0,0.0])
        paths = None
        if self.use_mpc:
            paths, _ = self.topo_planner.findTopoPaths([start[0], start[1], start[2]], goal, reset=True)
        """
        find_path_time = (time.time() - start_time)
        self.log.warn(f"[~] Path Time: {find_path_time:.5f} s")
        """

        U = np.kron(np.ones((1,self.Nt)), [0.0,self.nlmodel.control_bounds[1][1]]).ravel()
        U_anci = np.tile(self.U_prev, [self.num_anci, 1])
        # U_anci = np.tile(U, [self.num_anci, 1])
        if paths is not None:
            self.topo_paths = [np.array(path) for path in paths]
            find_nmpc_sol = functools.partial(find_Nonlin_Controls,
                                    start=start,
                                    solver=self.mpc_solver,
                                    dis=self.nlmodel.control_bounds[1][1]*self.nlmodel.dt*self.Nt,
                                    system=self.nlmodel,
                                    # Nt=int(self.Nt/2), 
                                    Nt=int(self.Nt/self.mpc_ratio), 
                                    occupied=self.occupied,
                                    # box=np.array([[2,2]]),
                                    box=np.array([[0.5,0.5]]),
                                    # box=np.array([[1,2]]),
                                    planner=self.topo_planner,
                                    ns=10
                                    )
            # # Sequential Functional# # 
            self.x_sol = []
            self.polyhedrals=[]
            for i,path in enumerate(paths):
                if i > self.num_anci:
                    break
                x_sol, u_sol, (A,b) = find_nmpc_sol(path)
                self.x_sol.append(x_sol)
                for j in range(len(A)):
                    self.polyhedrals.append([np.array(A[j]),np.array(b[j])])
                u_sol = np.repeat(u_sol,repeats=self.mpc_ratio,axis=0)
                if self.num_anci ==1:
                    U_anci = u_sol[:self.Nt,:].reshape((self.Nt*self.nlmodel.n_controls))
                else:
                    U_anci[i,:] = u_sol[:self.Nt,:].reshape((self.Nt*self.nlmodel.n_controls))
                # U_anci[i,:] = u_sol[:self.Nt,:].reshape((self.Nt))
        else:
            print(f"No paths found!")
        """
        find_mpc_time = time.time()-start_time-find_path_time
        self.log.warn(f"[~] MPC Time: {find_mpc_time:.5f} s")
        """
        output = self.mppi_controller.mppi_mmodal(
                                        state=start,
                                        U_original=self.U_prev,
                                        U_anci=U_anci,
                                        rng_key=self.SEED,
                                        q_ref=goal,
                                        safe_zones=safe_zones,
                                        cost_map=self.costmap,
                                        resolution=self.res,
                                        origin=self.origin,
                                        wh=self.wh)
        u_mppi = output[0]
        new_u = output[1]
        global_U = output[2]
        min_f_cost = output[3]
        safe_state_seq = output[4]
        con_state_seq = output[5]
        number_safe = output[6]
        current_safe = output[7]
        collision_free_state_seq = output[8]
        
        self.mppi_controller.temperature = output[9]
        contingency_controls = output[10]
        all_contingency_states = output[11]
        # self.log.warn(f"All Contingency_state shape: {all_contingency_states.shape}")
        all_costs = output[12]
        all_state_seq = output[13]
        contingency_controls = np.mean(contingency_controls, axis=0)
        # contingency_controls = contingency_controls[0,:]
        # print(contingency_controls.shape)
        self.contingency_plan = contingency_controls

        self.planned_time = self.get_clock().now()
        # self.plan = jnp.vstack((u_mppi, new_u))
        self.plan = u_mppi
        # u_mppi = U_anci[0,:].reshape(self.Nt,-1)[0,:]
        # new_u = U_anci[0,:].reshape(self.Nt,-1)[1:,:]
        # self.plan = U_anci[0,:].reshape(self.Nt,-1)
        # self.ctrl_vel = u_mppi.tolist()
        self.U_prev = global_U

        if jnp.isinf(min_f_cost):
            # print(start)
            pass
            # self.log.warn("Received Infinite Cost Trajectory")
        if not current_safe:
            pass
            # self.log.warn("Not Safe")
        sim_state = start
        state_seq = [sim_state]
        """
        find_mppi_time = time.time()-start_time-find_path_time-find_mpc_time
        self.log.warn(f"[~] MPPI Time: {find_mppi_time:.5f} s")
        """

        _, state_seq = jax.lax.scan(self.scan_controls,
                                         (sim_state), u_mppi)

        """
        find_states_time = time.time()-start_time-find_path_time-find_mpc_time-find_mppi_time
        self.log.warn(f"[~] Rollout Time: {find_states_time:.5f} s")
        """

        self.state_seq = np.array(state_seq)
        self.state_seq_odom = np.array(state_seq)

        self.sampled_states = all_state_seq
        self.safe_sampled_states = safe_state_seq
        self.contingency_states = np.array(con_state_seq)
        # self.publish_ctrl()

        """
        find_safe_sampled_states_time = time.time()-start_time-find_path_time-find_mpc_time-find_mppi_time-find_states_time
        self.log.warn(f"[~] Find Safe Sampled Time: {find_safe_sampled_states_time:.5f} s")
        """
        self.log.warn(f"[~] Total Time: {time.time()-start_time:.5f} s")
        # self.log.warn(f"[~] Total hz: {1/(time.time()-start_time):.5f} hz")
        self.data_dump(output, self.get_clock().now())
        # print(u_mppi[0])
        return

    def data_dump(self, output, stamp):
        all_contingency_states = output[11]
        all_costs = output[12]
        costs_msg = Float32MultiArray()
        costs_msg.data = [float(stamp.nanoseconds),  *(all_costs.flatten().tolist())]
        all_cont_msg = Float32MultiArray()
        all_cont_msg.data = [float(stamp.nanoseconds),
                             float(self.n_samples), 
                             float(self.N_safe),
                             float(self.N_mini), 
                             *(all_contingency_states.flatten().tolist())]
        self.costs_pub.publish(costs_msg) 
        self.all_cont_pub.publish(all_cont_msg) 


    def goal_callback(self, msg: PoseStamped) -> None:
        self.mppi_controller.temperature = self.temperature
        # self.plan = None
        self.goal_msg = msg
        self.log.info(f"[+] Goal Received")
        self.get_local_goal()
        self.run_time = self.get_clock().now()
        self.stop = False
        # if self.first:
            # self.create_plan()
            # self.first=False

        return
    def get_occupied(self, costmap, origin, resolution):
        occupied = np.argwhere(costmap > 0)
        occupied = occupied *resolution + origin
        
        return occupied
    
    def map_callback(self, msg: OccupancyGrid) -> None:
        while self.local_safe_zones is None:
            return
        self.res = msg.info.resolution
        self.origin = np.array([msg.info.origin.position.x,
                                msg.info.origin.position.y])
        self.wh = np.array([msg.info.width, msg.info.height])
        costmap = (np.array(msg.data))
        costmap = costmap.reshape((msg.info.height, msg.info.width)).T
        costmap[costmap == -1] = 50


        virtual_obs =  np.ones((msg.info.height, msg.info.width)).T
        for sz in self.local_safe_zones:
            radius = self.nlmodel.control_bounds[1][1]*self.nlmodel.dt*self.N_mini*1.0
            center = np.floor((sz[:2]-self.origin)/self.res)
            top_right = int(radius/self.res) + center
            bottom_left = center - int(radius/self.res)
            top_right = np.maximum(top_right, np.array([0,0]))
            bottom_left = np.maximum(bottom_left, np.array([0,0]))
            top_right = np.minimum(top_right, self.wh).astype(np.int32)
            bottom_left = np.minimum(bottom_left, self.wh).astype(np.int32)
            xx = np.arange(bottom_left[0], top_right[0]) - center[0]
            yy = np.arange(bottom_left[1], top_right[1]) - center[1]
            xs, ys = np.meshgrid(xx, yy) 
            temp = np.sqrt(xs**2 + ys**2) < radius/self.res
            # virtual_obs[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]] = 0
            virtual_obs[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]] = np.logical_and(
             virtual_obs[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]],           
                                        1-temp.T)
            # virtual_obs[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]] = 1-temp.T
        
        if self.local_position is not None:
            start = np.array([self.local_position.pose.pose.position.x,
                                self.local_position.pose.pose.position.y,
                        ])
            inds = ((start-self.origin)/self.res).astype(np.int32)
            if not (virtual_obs*10 + costmap)[inds[0], inds[1]]:
                costmap = virtual_obs*10 + costmap
                pass

        self.costmap = jnp.copy(costmap)
        self.occupied = self.get_occupied(costmap, self.origin, self.res)
        self.topo_planner.occup_grid = costmap
        self.topo_planner.origin = self.origin
        self.topo_planner.resolution = self.res
        self.topo_planner.wh =self.wh
        self.topo_planner.safe_zones= self.local_safe_zones
        values = np.unique(self.topo_planner.occup_grid)
        # print(values)
        msg.data = costmap.T.reshape((-1,)).astype(np.int8).tolist()
        self.masked_map_pub.publish(msg)
        self.wait = False
        return

    def odom_callback(self, msg) -> None:
        # self.log.info("[+] odom recieved")
        # # self.create_plan()
        # self.get_local_goal()
        # self.get_local_sz()
        # self.publish_local_goal()
        # self.publish_markers()
        self.local_position = msg
        return

    def tf_timer_cb(self) -> None:
        # self.create_plan()
        self.get_local_goal()
        self.get_local_sz()
        # self.get_local_pos()
        self.publish_local_goal()
        self.publish_markers()
        return
    
    def lqg_cb(self, i):
        # if self.plan is None:
        #     U = [[0.0,0.0]]
        q = self.local_position.pose.pose.orientation
        yaw = (np.arctan2(2.0*(q.y*q.x + q.w*q.z), 1.0 - 2.0 * (q.z*q.z + q.x*q.x)))
        start = np.array([self.local_position.pose.pose.position.x,
                             self.local_position.pose.pose.position.y,
                             yaw])
        controls = self.plan[i:np.min([i+5, len(self.plan)-1]), :]
        # controls = self.plan[i:i+2, :]
        states = self.state_seq[i:np.min([i+5, len(self.state_seq)-1]), :]
        # states = self.state_seq[i:i+2, :]
        states = states[:len(controls),:]
        x, U, cost = self.ilqr.ilqr(start, states, controls)      
        return U

    def con_service_cb(self, request, response):
        if self.contingency_plan is None:
            response.success = False
            response.message = "Contingency Plan is None"
            return response
        current_time = self.get_clock().now()
        dt = (current_time - self.planned_time).nanoseconds * 1e-9
        i = int(np.round(dt/self.dt))
        if i > self.N_safe-1:
            i = self.N_safe-1
        q = self.local_position.pose.pose.orientation
        yaw = (np.arctan2(2.0*(q.y*q.x + q.w*q.z), 1.0 - 2.0 * (q.z*q.z + q.x*q.x)))
        sim_state = np.array([self.local_position.pose.pose.position.x,
                            self.local_position.pose.pose.position.y,
                             yaw])
        if self.N_mini > 0 and self.n_mini > 0:
            contingency_controls = self.contingency_plan[i].ravel()
            contingency_controls = np.hstack([contingency_controls,
                                              np.zeros((self.Nt*self.nlmodel.N_CONTROLS)-len(contingency_controls))])
            goal = -1
            min_dist = np.inf
            sim_states = [sim_state]
            for u in contingency_controls.reshape(-1,2):
                sim_state = self.nlmodel.dynamics(sim_state, u, 0, dt=self.dt, params=self.nlmodel.nominal_params)
                sim_states.append(sim_state)
                dists = np.linalg.norm(self.local_safe_zones-sim_state, axis=1)
                i = np.argmin(dists)
                tmp_goal = self.local_safe_zones[i]
                tmp_dist = dists[i]
                if tmp_dist < self.TOLERANCE:
                    goal = tmp_goal
                    break
                if tmp_dist < min_dist:
                    min_dist = tmp_dist
                    goal = tmp_goal
            # self.use_mpc = not self.use_mpc
            self.U_prev = contingency_controls        
            self.plan = contingency_controls.reshape(-1,2)
            self.state_seq = np.array(sim_states)
        else:
            sim_states = [sim_state]
            dists = np.linalg.norm(self.local_safe_zones-sim_state, axis=1)
            i = np.argmin(dists)
            goal = self.local_safe_zones[i]

        self.goal_msg = PoseStamped(header=Header(frame_id=self.ODOM_FRAME) ,
                                        pose=Pose(position=Point(x=float(goal[0]), y=float(goal[1])))) 
        self.trigger_time = current_time
        response.success = True
        response.message = "Succesfully changed U_prev and goal_msg"
        return response

    def publish_ctrl(self, msg=None) -> None:
        # print(self.mppi_controller.mppi_mmodal._cache_size())
        # start_time=time.time()
        if (self.plan is None) or (self.state_seq is None):
            # or (self.mppi_controller.mppi_mmodal._cache_size()<2):
            return
        goal = np.array([self.goal_local_msg.pose.position.x,
                        self.goal_local_msg.pose.position.y,
                        0
        ])

        start = np.array([self.local_position.pose.pose.position.x,
                            self.local_position.pose.pose.position.y,
                    ])
        if np.linalg.norm(goal[:2]-start[:2]) <= self.TOLERANCE:
            self.use_mpc=True
            if not self.stop:
                self.log.info("[+] Goal Reached")
                self.stop = True
                if self.trigger_time:
                    dt = self.get_clock().now() -self.trigger_time
                    self.trigger_time = None
                    self.log.info(f"[+] Contingency time: {dt.nanoseconds*1e-9}")

        if self.stop:
            self.ctrl_vel = [0.0,0.0]
        else:
            current_time = self.get_clock().now()
            dt = (current_time - self.planned_time).nanoseconds * 1e-9
            i = int(dt/self.dt)
            if i >= self.Nt-1:
                i = self.Nt
                self.ctrl_vel = [0.0, 0.0]
            else:
                # self.ctrl_vel = self.plan.tolist()[i]
                U = self.lqg_cb(i).tolist()
                self.ctrl_vel = U[0]

        msg = Twist()
        msg.linear.x = self.ctrl_vel[1]
        msg.angular.z = self.ctrl_vel[0]
        self.cmd_vel_pub.publish(msg)
        # self.log.warn(f"Pub Ctrl time: {time.time()-start_time} s")
        return
    
        

    def _modify_marker(
        self,
        msg: Marker,
        m_id: int,
        x: list[float],
        y: list[float],
        color: tuple[float, float, float, float],
        stamp,
    ) -> Marker:
        msg.header.stamp = stamp
        msg.id = m_id
        cr, cg, cb, ca = color
        msg.color = ColorRGBA(r=cr, g=cg, b=cb, a=ca)
        msg.points = [Point(x=xi, y=yi, z=0.0) for (xi,yi) in zip(x,y)]
        return msg

    def publish_polyhedral(self):
        polyhedral_msg = MarkerArray()
        polyhedral_markers_base = cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id=self.ODOM_FRAME,  # type: ignore
                    m_scale=Vector3(x=0.1, y=0.0, z=0.0),
                )
        polyhedral_markers = []
        for i,polyhedral in enumerate(self.polyhedrals):
            A = np.array(polyhedral[0])
            b = np.array(polyhedral[1])
            try:
                # print(np.linalg.inv(A))
                # points = np.linalg.inv(A) @ b
                mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
                mat.rep_type=cdd.RepType.INEQUALITY
                poly = cdd.Polyhedron(mat)
                gen = poly.get_generators()
                verts = np.array(list(gen))[:,1:]
                inside = np.mean(verts, axis=0)
                # sorted_verts = np.zeros_like(verts)
                # angles = np.arctan2(verts[:,1]-verts[0,1], verts[:,0]-verts[0,0])
                angles = np.arctan2(verts[:,1]-inside[1], verts[:,0]-inside[0])
                
                sorted_verts = np.array(verts[np.argsort(angles)])
                # print("\n")
                # print(verts)
                # print(list(gen))
                x = np.append(sorted_verts[:,0], sorted_verts[0,0])
                y = np.append(sorted_verts[:,1], sorted_verts[0,1])
                # x = sorted_verts[:,0]
                # y = sorted_verts[:,1]
                # print(x)
                # print(y)
                polyhedral_marker = self._modify_marker(
                    m_id=i,
                    msg=copy.copy(polyhedral_markers_base),
                    x=x,
                    y=y,
                    color=(0.0, 0.0, 1.0, 0.5),
                    stamp=self.get_clock().now().to_msg()
                )
                polyhedral_markers.append(polyhedral_marker)
            except Exception as e:
                print(e)
                continue
        polyhedral_msg.markers = polyhedral_markers
        self.decomp_pub.publish(polyhedral_msg)

    def publish_markers(self) -> None:
        self.publish_polyhedral()
        # stamp=self.get_clock().now().to_msg()
        stamp=rclpy.time.Time().to_msg()
        sz_markers_msg = MarkerArray()
        if self.local_safe_zones is None:
            return
        self.safe_zones_markers = [
            self._modify_marker(
                msg=self.safe_zones_markers[0],
                m_id = 0,
                x=self.local_safe_zones[:,0].tolist(),
                y=self.local_safe_zones[:, 1].tolist(),
                color=(0.0, 0.0, 1.0, 1.0),
                stamp=stamp,
            )
        ]

        sz_markers_msg.markers = self.safe_zones_markers
        self.safe_zones_markers_pub.publish(sz_markers_msg)

        if self.state_seq_odom is None:
            return

        if np.isnan(self.state_seq).any():
            self.log.warn("No valid control sequences Found, not publishing markers!!!")
            return
        m_array_msg = MarkerArray()
        self.rollout_markers = [
            self._modify_marker(
                msg=self.rollout_markers[0],
                m_id = 0,
                x=self.state_seq_odom[:, 0].tolist(),
                y=self.state_seq_odom[:, 1].tolist(),
                color=(0.0, 0.0, 0.0, 0.2),
                stamp=stamp,
            )
        ]

        m_array_msg.markers = self.rollout_markers
        self.rollouts_marker_pub.publish(m_array_msg)

        if len(self.topo_paths) > 0:
            m_array_msg = MarkerArray()
            self.topo_path_markers = [
                self._modify_marker(
                    msg=copy.copy(self.topo_path_markers[0]),
                    m_id = i,
                    x=np.array(self.topo_paths[i])[:, 0].tolist(),
                    y=np.array(self.topo_paths[i])[:, 1].tolist(),
                    color=(0.01, 0.5, 0.5, 1.0),
                    stamp=stamp,
                )
                for i in range(len(self.topo_paths))
            ]
            m_array_msg.markers = self.topo_path_markers
            self.topo_paths_pub.publish(m_array_msg)
            

        if len(self.sampled_states) > 0:
            m_array_msg = MarkerArray()
            self.big_guy_markers = [
                self._modify_marker(
                    msg=copy.copy(self.big_guy_markers[0]),
                    m_id = i,
                    x=self.sampled_states[i, :, 0].tolist(),
                    y=self.sampled_states[i, :, 1].tolist(),
                    color=(1.0, 0.0, 0.0, 1.0),
                    stamp=stamp,
                )
                for i in range(len(self.sampled_states))
            ]
            m_array_msg.markers = self.big_guy_markers
            self.big_guy_markers_pub.publish(m_array_msg)

        if self.safe_sampled_states.shape[0] > 0:
            # print(self.safe_sampled_states.shape)
            # print(self.safe_sampled_states[0,:,1])
            try:
                m_array_msg = MarkerArray()
                self.safe_big_guy_markers = [
                    self._modify_marker(
                        msg=copy.copy(self.safe_big_guy_markers[0]),
                        m_id = i,
                        x=self.safe_sampled_states[i, :, 0].tolist(),
                        y=self.safe_sampled_states[i, :, 1].tolist(),
                        color=(1.0, 1.0, 0.0, 1.0),
                        stamp=stamp,
                    )
                    for i in range(len(self.safe_sampled_states))
                ]
                m_array_msg.markers = self.safe_big_guy_markers
                self.safe_big_guy_markers_pub.publish(m_array_msg)
            except Exception as e:
                pass

        if len(self.x_sol) > 0:
            m_array_msg = MarkerArray()
            self.anci_markers = [
                self._modify_marker(
                    msg=copy.copy(self.anci_markers[0]),
                    m_id = i,
                    x=self.x_sol[i][:,0].tolist(),
                    y=self.x_sol[i][:,1].tolist(),
                    color=(0.0, 1.0, 0.0, 1.0),
                    stamp=stamp,
                )
                for i in range(len(self.x_sol))
            ]
            m_array_msg.markers = self.anci_markers
            self.anci_markers_pub.publish(m_array_msg)

        if len(self.contingency_states) > 0:
            m_array_msg = MarkerArray()
            self.contingency_markers = [
                self._modify_marker(
                    msg=copy.copy(self.contingency_markers[0]),
                    m_id = i,
                    x=self.contingency_states[i,:,0].tolist(),
                    y=self.contingency_states[i,:,1].tolist(),
                    color=(0.0, 1.0, 1.0, 1.0),
                    stamp=stamp,
                )
                for i in range(len(self.contingency_states))
            ]
            m_array_msg.markers = self.contingency_markers
            self.contingency_markers_pub.publish(m_array_msg)
        return

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MPPINode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        # rclpy.spin(node)
        executor.spin()
    except KeyboardInterrupt:
        get_logger("Quitting MPPI Node").warn("[+] Shutting down MPPI Node.")
    node.destroy_node()
    return


if __name__ == "__main__":
    main()
