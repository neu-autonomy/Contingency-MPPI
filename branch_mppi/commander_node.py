import rclpy
from rclpy.node import Node
from tf2_geometry_msgs import do_transform_pose_stamped
from rclpy.duration import Duration
from rclpy.time import Time
import numpy as np
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Pose, Quaternion, TransformStamped, Twist, Vector3, Polygon
from tf2_ros.transform_listener import TransformListener
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import Trigger
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.logging import get_logger
from tf2_ros.buffer import Buffer

class CommanderNode(Node):

    def __init__(self):
        super().__init__('mppi_node')

        self.log = self.get_logger()
        q_ref = np.array([7.0, -0.5],)  # Reference state
        self.potential_goals = [np.array([0.5,1.0]), q_ref]

        # theta = np.pi/2
        # t = np.array([-5.0,-3.0])
        theta = 0.0
        t = np.array([-0.5,-1.0])
        # t = np.array([0.0,-1.0])
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s],
                    [s, c]])
        for i in range(len(self.potential_goals)):
            self.potential_goals[i] = R @ (np.array(self.potential_goals[i]) + t)
        print(self.potential_goals)
        self.index = 1
        self.publish = True

        client_cb_group =  MutuallyExclusiveCallbackGroup()
        goal_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Odometry, "odom", self.set_goal, 1)
        self.create_subscription(PoseStamped, "goal", self.local_goal_cb, 1, callback_group=goal_cb_group)
        self.goal_tf_buffer: Buffer = Buffer()
        # self.goal_tf_listener = TransformListener(
        #     self.goal_tf_buffer, self, spin_thread=False)
        self.goal_tf_listener = TransformListener(
            self.goal_tf_buffer, self)

        self.goal_pub = self.create_publisher(PoseStamped, "goal_pose", 1)
        self.mppi_goal = None

        self.con_trigger_client = self.create_client(Trigger, 'AAAAHHH', callback_group=client_cb_group)
        self.con_trgger_srv = self.create_service(Trigger, 'trigger_contingency', self.trigger_contingency)

    
    def trigger_contingency(self, request, response):
        if self.publish:
            # while not self.con_trigger_client.service_is_ready(timeout_sec=1.0):
            #     self.log.info("Contingency Service not available, waiting")

            _ = self.con_trigger_client.call(Trigger.Request())
        self.publish = not self.publish
        response.success = True
        response.message="Changed to "

        return response
    
    def local_goal_cb(self, msg):
        to_frame = "map"
        from_frame = msg.header.frame_id
        request_time = Time()
        t: TransformStamped = self.goal_tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                request_time,
                Duration(seconds=0.5))
        
        safe_zone_msg = do_transform_pose_stamped(msg, t)
        self.mppi_goal = np.array([safe_zone_msg.pose.position.x, safe_zone_msg.pose.position.y])

    
    def set_goal(self, msg):
        to_frame = "map"
        from_frame = msg.header.frame_id
        request_time = Time()
        t: TransformStamped = self.goal_tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                request_time,
                Duration(seconds=0.5))
        odom_msg = do_transform_pose_stamped(msg.pose, t)
        position = odom_msg.pose.position

        state = np.array([position.x,
                            position.y,
                    ])
        goal = self.potential_goals[self.index]
        prev_index = self.index
        # print(np.linalg.norm(state-goal))
        # print(state)
        # print(goal)
        # print(self.mppi_goal)

        # print(self.potential_goals)
        # print(state)
        if np.linalg.norm(state-goal)<1.0:
            self.index = (self.index+ 1) % 2
        # if self.publish and (self.index != prev_index):
        if self.publish and (np.any(self.mppi_goal != self.potential_goals[self.index])):
        # if self.publish:
            self.goal_msg = PoseStamped(header=Header(frame_id="map") ,
                                            pose=Pose(position=Point(x=float(goal[0]), y=float(goal[1])))) 
            self.goal_pub.publish(self.goal_msg)
        
def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommanderNode()
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