# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml, ReplaceString
import launch
import launch_ros
import lifecycle_msgs.msg

def generate_launch_description():
    # Get the launch directory
    # bringup_dir = get_package_share_directory('nav2_bringup')
    nail_utils_dir = get_package_share_directory('branch_mppi')
    # launch_dir = os.path.join(bringup_dir, 'launch')

    # Create the launch configuration variables
  
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(nail_utils_dir, 'config', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes',
    )

    declare_map_yaml_cmd = DeclareLaunchArgument(
        # 'map', default_value="/branch_mppi_ws/src/autonomy-stack/src/nail_utils/maps/highbay.yaml",
        'map', default_value="",
          description='Full path to map yaml file to load'
    )
    # Only it applys when `use_namespace` is True.

    # '<robot_namespace>' keyword shall be replaced by 'namespace' launch argument
    # in config file 'nav2_multirobot_params.yaml' as a default & example.
    # User defined config file should contain '<robot_namespace>' keyword for the replacements.

    configured_params = ParameterFile(
        param_file=LaunchConfiguration('params_file'),
        allow_substs=True)

    costmap_node = launch_ros.actions.LifecycleNode(
        name='costmap',
        namespace='costmap',
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        parameters=[configured_params],
        remappings=[('/map', '/map_server/initial_map')],
        output="screen",
        )
    map_server_node = launch_ros.actions.LifecycleNode(
        name='map_server',
        namespace='map_server',
        package='nav2_map_server',
        executable='map_server',
        emulate_tty=True,  # https://github.com/ros2/launch/issues/188
        parameters=[configured_params, {'yaml_filename': LaunchConfiguration('map')}],
        remappings=[('map', 'initial_map')],
        output="screen",
    )

    costmap_to_inactive = launch.actions.EmitEvent(
        event=launch_ros.events.lifecycle.ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(costmap_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    costmap_from_inactive_to_active = launch.actions.RegisterEventHandler(
            launch_ros.event_handlers.OnStateTransition(
                target_lifecycle_node=costmap_node,
                start_state = 'configuring',
                goal_state='inactive',
                entities=[
                    launch.actions.LogInfo(msg="-- Inactive --"),
                    launch.actions.EmitEvent(event=launch_ros.events.lifecycle.ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(costmap_node),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )),
                ],
            )
        )
        
    map_server_to_inactive = launch.actions.EmitEvent(
        event=launch_ros.events.lifecycle.ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(map_server_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    map_server_from_inactive_to_active = launch.actions.RegisterEventHandler(
            launch_ros.event_handlers.OnStateTransition(
                target_lifecycle_node=costmap_node,
                start_state = 'configuring',
                goal_state='inactive',
                entities=[
                    launch.actions.LogInfo(msg="-- Inactive --"),
                    launch.actions.EmitEvent(event=launch_ros.events.lifecycle.ChangeState(
                        lifecycle_node_matcher=launch.events.matches_action(map_server_node),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )),
                ],
            )
        )

    # Create the launch description and populate
    ld = LaunchDescription()
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(costmap_to_inactive)
    ld.add_action(costmap_from_inactive_to_active)
    ld.add_action(map_server_to_inactive)
    ld.add_action(map_server_from_inactive_to_active)
    ld.add_action(costmap_node)
    ld.add_action(map_server_node)
    return ld
