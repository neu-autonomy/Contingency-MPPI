import rclpy
from rclpy.node import Node
import jax.numpy as jnp
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_msgs.msg import ColorRGBA, Header, Float32MultiArray
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Pose, Quaternion, TransformStamped, Twist, Vector3, Polygon
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.lifecycle import State, TransitionCallbackReturn, Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.logging import get_logger
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import copy
from branch_mppi.systems import Unicycle
from branch_mppi.jax_mppi import reachability
import jax

def cfg_rviz_mkr(
    m_id: int,
    m_frame_id: str,
    m_type: int = Marker.LINE_STRIP,
    m_action: int = Marker.ADD,
    m_position: Point = Point(x=0.0, y=0.0, z=0.0),
    m_orientation: Quaternion = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    m_scale: Vector3 = Vector3(x=0.02, y=0.0, z=0.0),
    m_duration: Duration = Duration(seconds=1).to_msg(),
    m_FrameLocked: bool = False,
    ns=''
) -> Marker:
    msg = Marker()
    msg.header.frame_id = m_frame_id
    msg.id = m_id
    msg.ns=ns
    msg.type = m_type
    msg.action = m_action
    msg.pose.position = m_position
    msg.pose.orientation = m_orientation
    msg.scale = m_scale
    msg.lifetime = m_duration
    msg.frame_locked = m_FrameLocked
    return msg
    
def modify_marker(
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

class VisNode(Node):
    def __init__(self):
        super().__init__('vis_node')
        self.log = self.get_logger()
        self.costs = []
        self.timestamps = []
        self.traj_msgs = []

        self.create_subscription(Float32MultiArray, "all_cont", self.vis_all_cont_cb, 1)
        self.create_subscription(Float32MultiArray, "all_costs", self.costs_cb, 1)
        self.create_subscription(OccupancyGrid, "/costmap/costmap", self.map_callback, 1)
        self.create_subscription(MarkerArray, "rollout_state", self.rollout_callback, 5, callback_group=MutuallyExclusiveCallbackGroup())

        self.cont_traj_pub = self.create_publisher(MarkerArray, "selected_contingencies", 10)
        self.sampled_traj_pub = self.create_publisher(MarkerArray, "selected_sampled_traj", 10)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.safe_zone_sub = self.create_subscription(MarkerArray, "safe_zones", self.safe_zones_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.safe_zone_repub = self.create_publisher(MarkerArray, "safe_zones_repub", qos_profile=qos_profile)
        dt = 0.2
        self.system =  Unicycle({"lb":np.array([-0.5, -0.5]),
                        "ub":np.array([0.5, 1.5])}, dt=dt)

        # self.safe_zone_marker = cfg_rviz_mkr(
        #             m_id=0,
        #             m_frame_id="odom_dlio",  # type: ignore
        #             m_scale=Vector3(x=0.8, y=0.8, z=0.8),
        #         )
        # self.safe_zone_marker.lifetime = Duration().to_msg()

        self.main_markers = cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id="odom_dlio",  # type: ignore
                    m_scale=Vector3(x=0.075, y=0.0, z=0.0),
                )
        self.main_markers_dots = cfg_rviz_mkr(
                    m_id=0,
                    ns='dots',
                    m_frame_id="odom_dlio",  # type: ignore
                    m_type=7,
                    m_scale=Vector3(x=0.09, y=0.09, z=0.09),
                )
        
        self.cont_markers = [
                cfg_rviz_mkr(
                    m_id=0,
                    m_frame_id="odom_dlio",  # type: ignore
                    m_scale=Vector3(x=0.05, y=0.0, z=0.0),
                )
        ]
        self.safe_zones_published=True
    
    def safe_zones_cb(self, msg):
        self.safe_zones = []
        # safe_zone_marker = copy.deepcopy(self.safe_zone_marker)
        for marker in msg.markers:
                marker.scale.x = 0.8
                marker.scale.y = 0.8
                marker.scale.z = 0.8
                marker.color.a=0.3
                marker.lifetime=Duration().to_msg()
                for point in marker.points:
                    self.safe_zones.append([point.x, point.y])
                    # safe_zone_marker.points.append(point)

        # self.safe_zones_msg = msg
        # print(type(safe_zone_marker))
        # self.safe_zone_repub.publish(safe_zone_marker)
        self.safe_zone_repub.publish(msg)

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.res = msg.info.resolution
        self.origin = np.array([msg.info.origin.position.x,
                                msg.info.origin.position.y])
        self.wh = np.array([msg.info.width, msg.info.height])
        costmap = (np.array(msg.data))
        costmap = costmap.reshape((msg.info.height, msg.info.width)).T
        costmap[costmap == -1] = 50

        self.costmap = jnp.copy(costmap)
        msg.data = costmap.T.reshape((-1,)).astype(np.int8).tolist()
        return
    
    def rollout_callback(self, msg):
        rollout = msg.markers[0].points
        x = np.array([state.x for state in rollout])
        y = np.array([state.y for state in rollout])
        dx = x[1:] - x[0:-1]
        dy = y[1:] - y[0:-1]
        theta = np.arctan2(dy, dx)
        theta = np.append(theta, theta[-1])
        self.state_seq = jnp.array([x,y,theta]).T

    
    def vis_all_cont_cb(self, msg: Float32MultiArray):
        m_elite = 50
        if len(self.costs) == 0:
            return
        data = msg.data
        timestamp = data[0]
        n_samples = int(data[1])
        N_safe =int(data[2])
        N_mini =int(data[3])
        stamp=rclpy.time.Time().to_msg()

        try:
            if N_mini > 0:
                contingencies = np.array(data[4:]).reshape((n_samples, N_safe, N_mini, -1))
                idx = self.timestamps.index(timestamp)
                costs = self.costs[idx]
                sorted_idxs = np.argsort(costs)
                mini_guys = []
                main_state_seq = []
                for i in range(m_elite):
                    mc_idx = sorted_idxs[i]
                    total_con_states = contingencies[mc_idx, :, :, :]
                    for j in range(N_safe):
                        reached=False
                        kron_safe_zones =np.kron(np.ones((N_mini,1)), self.safe_zones)
                        kron_states = np.kron(total_con_states[j,:,:2], np.ones((7,1)))
                        if np.any(np.linalg.norm(kron_safe_zones-kron_states, axis=1)<=0.5):
                            reached=True
                        if not reached:
                            break
                    if j >= N_safe-1:
                        for k in range(N_safe):
                            mini_guys.append(total_con_states[k,:,:])
                        main_state_seq.append(total_con_states[:,0,:])
                        break

                main_markers = [
                    modify_marker(
                    msg=copy.deepcopy(self.main_markers),
                    m_id = i,
                    x=main_state_seq[i][:, 0].tolist(),
                    y=main_state_seq[i][:, 1].tolist(),
                    color=(0.0, 0.0, 0.0, 0.5),
                    stamp=stamp,
                ) for i in range(len(main_state_seq))
                ]
                dot_markers = [
                    modify_marker(
                    msg=copy.deepcopy(self.main_markers_dots),
                    m_id = i,
                    x=main_state_seq[i][:, 0].tolist(),
                    y=main_state_seq[i][:, 1].tolist(),
                    color=(0.0, 0.0, 0.0, 1.0),
                    stamp=stamp,
                ) for i in range(len(main_state_seq))
                ]
                self.sampled_traj_pub.publish(MarkerArray(markers=[*main_markers, *dot_markers]))

                cont_markers = [
                    modify_marker(
                        msg=copy.deepcopy(self.cont_markers[0]),
                        m_id=i,
                        x=mini_guys[i][:,0].tolist(),
                        y=mini_guys[i][:,1].tolist(),
                        color=(0.027,0.42,0.082,0.9),
                        stamp=stamp
                    ) for i in range(len(mini_guys))
                ]
                self.cont_traj_pub.publish(MarkerArray(markers=cont_markers))

            else:
                ais=3 
                N_mini = 8
                sigma0 = np.diag(np.array([0.5, 0.25]))
                sigma0 = np.kron(np.eye(N_mini), sigma0)
                safe_flag, safe_state_seq, state_seqs_all, state_seqs_means, safe_u_seqs = jax.vmap(reachability.get_reachability_mppi_costmap, 
                    in_axes=(None, None, None, None, None, 0, None, None, None, None, None, None, None,  None, None, None)) \
                    (self.system,  
                    self.costmap,  
                    self.origin, 
                    self.res, 
                    self.wh,         
                    self.state_seq, 
                    np.array(self.safe_zones), 
                    jnp.array([[0.0,0.0]]), 
                    jax.random.PRNGKey(2), 
                    100, 
                    N_mini,  
                    jnp.array(sigma0)*2, 
                    jnp.kron(jnp.ones((N_mini,)), jnp.array([-0.5,1.5])), 
                    1.0/1000, 
                    ais,
                    10
                    ) 
                main_markers = [
                    modify_marker(
                    msg=copy.deepcopy(self.main_markers),
                    m_id = 0,
                    x=self.state_seq[:, 0].tolist(),
                    y=self.state_seq[:, 1].tolist(),
                    color=(0.0, 0.0, 0.0, 0.5),
                    stamp=stamp,
                )
                ]
                dot_markers = [
                    modify_marker(
                    msg=copy.deepcopy(self.main_markers_dots),
                    m_id = 0,
                    x=self.state_seq[:, 0].tolist(),
                    y=self.state_seq[:, 1].tolist(),
                    color=(0.0, 0.0, 0.0, 1.0),
                    stamp=stamp,
                )
                ]
                self.sampled_traj_pub.publish(MarkerArray(markers=[*main_markers, *dot_markers]))
                cont_markers = [
                    modify_marker(
                        msg=copy.deepcopy(self.cont_markers[0]),
                        m_id=i,
                        x=safe_state_seq[i][:,0].tolist(),
                        y=safe_state_seq[i][:,1].tolist(),
                        color=(0.027,0.42,0.082,0.9),
                        stamp=stamp
                    ) for i in range(len(safe_state_seq))
                ]
                self.cont_traj_pub.publish(MarkerArray(markers=cont_markers)) 
        except Exception as e:
            self.log.warn(f"{e}")


    def costs_cb(self, msg: Float32MultiArray):
        data = msg.data 
        self.timestamps.append(data[0])
        self.costs.append(data[1:])

        while len(self.timestamps)>10:
            self.timestamps.pop(0)
            self.costs.pop(0)
        pass

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        # rclpy.spin(node)
        executor.spin()
    except KeyboardInterrupt:
        get_logger("Quitting MPPI Node").warn("[+] Shutting down MPPI Node.")
    node.destroy_node()
    return