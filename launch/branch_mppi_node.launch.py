import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
import launch
import launch_ros
import lifecycle_msgs.msg
from datetime import datetime
import sys

for p in sys.path:
    print(p)

def generate_launch_description():
    param_file = os.path.join(get_package_share_directory("branch_mppi"), "config", "default.yaml")
    # remappings = [("goal_pose","/move_base_simple/goal")]
    remappings = [("occupancy_grid", "/costmap/costmap"),
                    ('odom', '/dlio/odom_node/odom'),
                #   ('cmd_vel', 'mppi/cmd_vel')
                  ]
    customMppiNode =launch_ros.actions.LifecycleNode(
        name="nested_mppi_node",
        namespace='',
        package="branch_mppi",
        executable="nested_mppi_node",
        output="screen",
        parameters=[param_file],
        emulate_tty=True,
        remappings=remappings,
    )

    commanderNode =launch_ros.actions.Node(
        name="commander_node",
        namespace='',
        package="branch_mppi",
        executable="commander_node",
        output="screen",
        parameters=[param_file],
        emulate_tty=True,
        remappings=remappings,
    )

    to_inactive = launch.actions.EmitEvent(
        event=launch_ros.events.lifecycle.ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(customMppiNode),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    from_inactive_to_active = launch.actions.RegisterEventHandler(
            launch_ros.event_handlers.OnStateTransition(
                target_lifecycle_node=customMppiNode,
                start_state = 'configuring',
                goal_state='inactive',
                entities=[
                    launch.actions.LogInfo(msg="-- Inactive --"),
                    launch.actions.EmitEvent(event=launch_ros.events.lifecycle.ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(customMppiNode),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )),
                    launch.actions.LogInfo(msg="-- Active --"),
                ],
            )
        )
    cur_time = datetime.now().strftime("%a_%b-%d-%Y_%I-%M-%S_%p")
    directory = f"~/bagfiles/lj_mppi/{cur_time}"
    Qos_override = launch.substitutions.PathJoinSubstitution(
        [launch_ros.substitutions.FindPackageShare("nail_utils"), "cfg", "qos_depth_override.yaml"]
    )
    rosbag_record = launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            directory,
            "--qos-profile-overrides-path",
            Qos_override,
            "--max-cache-size",
            "1000000000",  # storing 40kb/s of data for 1 mins coz the QoS can only store last 10000 msgs
            "--max-bag-size",
            "3221225472",  # 3Gb of raw data (~11x less after compression)
            "--max-bag-duration",
            "1800",  # seconds
            "-x",  # exclude camera topics and record everything else
            "'(/camera/.*|/ouster/.*)'", 
            # "-x",
            # "/ouster/.*",
            # "-x",
            # "/ouster/scan",
            "-a",
        ],
        shell=True,
    )
    return LaunchDescription(
        [
            to_inactive, 
            from_inactive_to_active, 
            customMppiNode, 
            # commanderNode,
            # launch.actions.TimerAction(period=1.0,actions=[rosbag_record])
        ]
    )
