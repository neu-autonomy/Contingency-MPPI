import rosbags
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg,  \
                    register_types, \
                    Stores, \
                    get_typestore 
from rosbags.rosbag2 import Writer, WriterError
import numpy as np
import shutil

MarkerArrayMsg = """
visualization_msgs/Marker[] markers
"""
MarkerMsg ="""
uint8 ARROW=0
uint8 CUBE=1
uint8 SPHERE=2
uint8 CYLINDER=3
uint8 LINE_STRIP=4
uint8 LINE_LIST=5
uint8 CUBE_LIST=6
uint8 SPHERE_LIST=7
uint8 POINTS=8
uint8 TEXT_VIEW_FACING=9
uint8 MESH_RESOURCE=10
uint8 TRIANGLE_LIST=11
uint8 ADD=0
uint8 MODIFY=0
uint8 DELETE=2
uint8 DELETEALL=3
std_msgs/Header header
string ns
int32 id
int32 type
int32 action
geometry_msgs/Pose pose
geometry_msgs/Vector3 scale
std_msgs/ColorRGBA color
duration lifetime
bool frame_locked
geometry_msgs/Point[] points
std_msgs/ColorRGBA[] colors
string text
string mesh_resource
bool mesh_use_embedded_materials
"""
CostmapMsg = """
std_msgs/Header header
CostmapMetaData metadata
uint8[] data
"""
CostmapMetaDataMsg = """
builtin_interfaces/Time map_load_time
builtin_interfaces/Time update_time
string layer
float32 resolution
uint32 size_x
uint32 size_y
geometry_msgs/Pose origin
"""
typestore = get_typestore(Stores.ROS2_HUMBLE)
# typestore.register(get_types_from_msg(MarkerArrayMsg, 'visualization_m
# sgs/msg/MarkerArray'))typestore.register(get_types_from_msg(MarkerMsg, 'visualization_msgs/msg/Marker'))
typestore.register(get_types_from_msg(CostmapMsg, 'nav2_msgs/msg/Costmap'))
typestore.register(get_types_from_msg(CostmapMetaDataMsg, 'nav2_msgs/msg/CostmapMetaData'))


# path = "/media/jungle/T7/lj_mppi/succesful_2ms/"
# path = "/media/jungle/T7/lj_mppi/output/bag1"
path = "/media/jungle/T7/working_15/Mon_Nov-25-2024_01-29-27_AM"
# out_path = "/media/jungle/T7/lj_mppi/output/edited_bag1"
out_path = f"{path}_edited"


if Path(out_path).exists() and Path(out_path).is_dir():
    shutil.rmtree(Path(out_path))

safe_zones = None
topics = []
connections={}

with AnyReader([Path(path)]) as reader:
    with Writer(out_path, version=5) as new_bag:
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)
            if not connection.topic in topics:
                topics.append(connection.topic)
                # breakpoint()e
                try:
                    connections[connection.topic] = new_bag.add_connection(connection.topic, connection.msgtype, typestore=typestore)
                except Exception as e:
                    pass

        for connection, timestamp, rawdata in reader.messages():
            try:
                # msg = rosbags.serde.deserialize_cdr(rawdata, connection.msgtype)
                if connection.msgtype == "nav2_msgs/msg/Costmap":
                    continue
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                if connection.topic=="/safe_zones":
                 #   'odom' frame
                    points = msg.markers[0].points
                    msg.markers[0].type=7
                    msg.markers[0].scale.x=0.8
                    msg.markers[0].scale.y=0.8
                    msg.markers[0].scale.z=0.8
                    safe_zones = np.array([[a.x, a.y, a.z] for a in points])
                
                if connection.topic=="/contingency_states":
                    new_arr = []
                    for arr in msg.markers:
                        reached=False
                        points = arr.points
                        # path = np.array([[a.x, a.y, a.z] for a in points])
                        for a in points:
                        # 'odom' frame
                            state = np.array([a.x, a.y, a.z])
                            # if not np.isnan(state[0]):
                            #     breakpoint()
                            # if np.any(np.linalg.norm(state[:2] - obs[:,:2], axis=1) < obs[:,2]):
                            #     collided =True
                            #     break
                            if np.any(np.linalg.norm(state[:2] - safe_zones[:,:2], axis=1) < 0.5):
                                reached=True
                        if reached:
                            print("REACHED")
                            arr.color.g=1.0
                            arr.color.b=0.0
                            arr.scale.x=0.05
                            new_arr.append(arr)

                        if len(new_arr) > 3:
                            break
                    msg.markers = new_arr

                new_bag.write(connections[connection.topic], timestamp, typestore.serialize_cdr(msg, connection.msgtype))

            except Exception as e:
                print(connection.msgtype)
                print(timestamp)
                print(e)

                continue

