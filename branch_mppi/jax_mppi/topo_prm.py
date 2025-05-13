import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from branch_mppi.jax_mppi.grid import OccupGrid
from enum import Enum
import math
import pydecomp as pdc

class NODE_TYPE(Enum):
    GUARD=1
    CONNECTOR=2

class NODE_STATE(Enum):
    NEW=1
    CLOSE=2
    OPEN=3
class Node:
    def __init__(self, pos, type, id, sz=False):
        self.pos = pos # np.ndarray
        self.type = type # NODE_TYPE
        self.state = NODE_STATE.NEW # NODE_STATE
        self.id = id # int
        self.sz = sz
        self.neighbors = [] # 
    def __repr__(self):
        return repr(f'ID: {self.id} Type: {self.type} State: {self.state} Pos: {self.pos}')

class TopoPRM:
    def __init__(self, occup_grid, 
                # sample_inflate=[10.0, 30.0, 0.0],
                 sample_inflate=[5.0, 5.0, 0.0],
                 origin=[0.0,0.0], 
                 resolution=0.1, 
                 wh = [10,10], 
                 max_raw_path=10, 
                 max_raw_path2=5, 
                 max_sample_num=200, 
                 reserve_num=3, 
                 ratio_to_short=2,
                 sample_sz_p=1.0,
                 footprint = [[0.0, 0.0]],
                 occup_value=[100],
                 max_time=0.03
                #  max_time=0.01
                ):
        self.sample_inflate = sample_inflate
        self.sample_r = [0.0,0.0,0.0]
        self.occup_grid = occup_grid
        self.resolution = resolution
        self.wh = wh
        self.origin = origin
        self.max_sample_num = max_sample_num
        # self.clearance = 0.5
        self.ray_resolution = 0.2
        self.topo_resolution = 0.2
        self.translation = -1
        self.R = np.zeros((3,3))
        self.graph = []
        self.reserve_num = reserve_num
        self.ratio_to_short = ratio_to_short
        self.start_pts = []
        self.end_pts = []
        self.max_raw_path = max_raw_path
        self.max_raw_path2 = max_raw_path2
        self.max_time = max_time
        self.safe_zones = None
        self.sample_sz_p=sample_sz_p
        self.footprint = np.array(footprint)
        self.occup_value = occup_value
        self.virtual_occup_value = 10

    def find_occupied(self, occup, pt, mark_out_of_bounds=False):
        # footprint = self.footprint*0.5+pt[:2]
        footprint = np.array([pt[:2]])
        # footprint = 
        # footprint = pt[:2]
        # x_ind, y_ind= np.floor((pt[:2]-self.origin)/self.resolution).astype(np.int32)
        # print(np.floor((footprint-self.origin)/self.resolution).astype(np.int32))
        inds = np.floor((footprint-self.origin)/self.resolution).astype(np.int32)
        if (np.any(inds<0) or np.any(inds[:, 0] >= self.wh[0]) or np.any(inds[:, 1] >= self.wh[1])):
            if mark_out_of_bounds:
                return True
            else:
                return False
            # return True
        # if x_ind < 0 or x_ind >= self.wh[0] or y_ind < 0 or y_ind >= self.wh[1]:
        #     return False
        try:
            # if occup[x_ind, y_ind]== self.occup_value:
            # if np.any(occup[x_ind, y_ind] >= self.occup_value):
            if np.any(occup[inds[:,0], inds[:,1]] >= self.occup_value):
                return True

        except:
            return True

        return False

    def findTopoPaths(self, start, end, start_pts=[], end_pts=[], reset=True):
        
        start = np.array(start).reshape((3,1)) 
        end = np.array(end).reshape((3,1))
        start[2] = 0.0
        end[2] = 0.0
        self.start_pts = start_pts
        self.end_pts = end_pts
        if not reset:
            self.graph = self.changeGraphStart(start, end, self.graph)
            # breakpoint()
            self.graph, samples = self.createGraph(start, end, self.graph)
        else:
            self.graph = []
            self.graph, samples = self.createGraph(start, end, self.graph)
        raw_paths = self.searchPaths(self.graph)
        short_paths = self.shortcutPaths(raw_paths)
        filtered_paths = self.pruneEquivalent(short_paths)
        select_paths = self.selectShortPaths(filtered_paths, 1)
        # select_paths = self.selectShortPaths(short_paths, 1)
        # select_paths = self.selectShortPaths(raw_paths, 1)
        return select_paths, samples
        # print(f"raw_paths: {raw_paths}")
        # print(f"Select_paths: {select_paths}")
        # return raw_paths, samples

    def sampleSafeZones(self, samples, graph, node_id):
        if self.safe_zones is not None:
            for pt in self.safe_zones:
                if self.find_occupied(self.occup_grid, pt):
                    continue
                visible_guards = self.findVisibGuard(pt, graph)
                if len(visible_guards) == 0:
                    guard = Node(pt, NODE_TYPE.GUARD, node_id, sz=True)
                    node_id += 1
                    graph.append(guard)
                    samples.append(pt)
                elif len(visible_guards) == 2:
                    need_connect = self.needConnection(visible_guards[0], visible_guards[1], pt)
                    samples.append(pt)
                    if not need_connect:
                        continue
                    connector = Node(pt, NODE_TYPE.CONNECTOR, node_id, sz=True)
                    node_id += 1
                    visible_guards[0].neighbors.append(connector)
                    visible_guards[1].neighbors.append(connector)
                    connector.neighbors.append(visible_guards[0])
                    connector.neighbors.append(visible_guards[1])
                    graph.append(connector)
        return samples, graph, node_id

    def get_random_sz(self):
        r_index = int(np.random.uniform(high=len(self.safe_zones)))
        return self.safe_zones[r_index]

    def createGraph(self, start, end, graph):
        if len(graph) == 0:
            start = np.array(start).reshape((3,1))
            end = np.array(end).reshape((3,1))
            graph.append(Node(start.ravel(), NODE_TYPE.GUARD, 0))
            graph.append(Node(end.ravel(), NODE_TYPE.GUARD, 1))

        self.sample_r = [0.5*np.linalg.norm(end-start) + self.sample_inflate[0],
                          self.sample_inflate[1],
                          self.sample_inflate[2]]
        self.translation  = np.array(0.5*(start+end)).reshape((3,1))
        xtf = (end-self.translation).T
        xtf = xtf / np.linalg.norm(xtf)
        ytf = np.cross(xtf, [[0,0,-1]])
        ytf = ytf / np.linalg.norm(ytf)
        ztf = np.cross(xtf, ytf)
        self.R = np.vstack((xtf, ytf, ztf)).T

        node_id = 2
        sample_num = 0
        samples = []
        st = time.time()

        # samples, graph, node_id = self.sampleSafeZones(samples, graph, node_id)
        while (sample_num < self.max_sample_num) and (time.time() - st < self.max_time):
            pr = np.random.uniform()
            if pr<self.sample_sz_p:
                pt = self.get_random_sz()
            else:
                pt = self.getSample()
            sample_num += 1
            if self.find_occupied(self.occup_grid, pt):
                continue

            visible_guards = self.findVisibGuard(pt, graph)
            if len(visible_guards) == 0:
                guard = Node(pt, NODE_TYPE.GUARD, node_id)
                node_id += 1
                graph.append(guard)
                samples.append(pt)
            elif len(visible_guards) >= 2:
                need_connect = self.needConnection(visible_guards[0], visible_guards[1], pt)
                samples.append(pt)
                if not need_connect:
                    continue
                connector = Node(pt, NODE_TYPE.CONNECTOR, node_id)
                node_id += 1
            
                visible_guards[0].neighbors.append(connector)
                visible_guards[1].neighbors.append(connector)

                connector.neighbors.append(visible_guards[0])
                connector.neighbors.append(visible_guards[1])

                graph.append(connector)
        # for i in range(3):
        #     samples, graph, node_id = self.sampleSafeZones(samples, graph, node_id)
        graph = self.pruneGraph(graph)
        
        return graph, samples

    def changeGraphStart(self, start, end, graph):
        st_time = time.time()
        if len(graph) == 0:
            return []
        for neighbor in graph[0].neighbors:
            # breakpoint()
            neighbor.neighbors.remove(graph[0])
        for neighbor in graph[1].neighbors:
            neighbor.neighbors.remove(graph[1])
        graph.pop(0)
        graph.pop(1) 
        for node in graph:
            node.id+=1
        graph.insert(0, Node(start.ravel(), NODE_TYPE.GUARD, 0))
        graph.insert(1, Node(end.ravel(), NODE_TYPE.GUARD, 1))

        for node in graph[2:]:
            if (node.type == NODE_TYPE.CONNECTOR):
                if (len(node.neighbors) < 2):
                    if self.lineVisib(node.pos, graph[0].pos, self.ray_resolution):
                        node.neighbors.append(graph[0])
                        graph[0].neighbors.append(node)
                    # if self.lineVisib(node.pos, graph[1].pos, self.ray_resolution):
                    #     node.neighbors.append(graph[1])
                    #     graph[1].neighbors.append(node)
                else:
                    graph.remove(node)
                
                # visible_guards = self.findVisibleGuard(node.pos, graph)
                # if len(visible_guards) >= 2:
                #     need_connect = self.needConnection(visible_guards[0], visible_guards[1], node.pos)
                #     if not need_connect:
                #         continue
                # visible_guards[0].neighbors.append(node)
                # visible_guards[1].neighbors.append(node)
        # print(f"Time taken to change graph start: {time.time()-st_time}")
        # print(f"Graph size: {len(graph)}")
        return graph

    # def changeGraphStart(self, start, end, graph):
    #     st_time = time.time()
    #     if len(graph) == 0:
    #         return []
    #     graph[0].pos = start.ravel()
    #     graph[1].pos = end.ravel()
    #     for neighbor in graph[0].neighbors:
    #         if not self.lineVisib(neighbor.pos, graph[0].pos, self.ray_resolution):
    #             graph[0].neighbors.remove(neighbor)
    #             neighbor.neighbors.remove(graph[0])
    #             # graph.remove(neighbor)
                
    #     for neighbor in graph[1].neighbors:
    #         if not self.lineVisib(neighbor.pos, graph[1].pos, self.ray_resolution):
    #             graph[1].neighbors.remove(neighbor)
    #             neighbor.neighbors.remove(graph[1])
    #             # graph.remove(neighbor)

    #     print(f"Time taken to change graph start: {time.time()-st_time}")
    #     print(f"Graph size: {len(graph)}")
    #     return graph

    def findVisibGuard(self, pt, graph):
        visible_guards = []
        visib_num = 0

        for node in graph:
            if node.type == NODE_TYPE.CONNECTOR:
                continue
            if np.all(node.pos == pt):
                return []

            if self.lineVisib(pt, node.pos, self.ray_resolution):
                visible_guards.append(node)
                visib_num +=1
                if visib_num > 2:
                    break
        return visible_guards
    
    def needConnection(self, node1, node2, pt):
        path1 = [node1.pos,
                pt,
                node2.pos]
        path2 = [node1.pos,
                 0,
                 node2.pos]
        
        for n1_neigh in node1.neighbors:
            for n2_neigh in node2.neighbors:
                if n1_neigh.id == n2_neigh.id:
                    path2[1] = n1_neigh.pos
                    same_topo = self.sameTopoPath(path1, path2, self.topo_resolution)
                    if same_topo:
                        if self.pathLength(path1) < self.pathLength(path2):
                            n1_neigh.pos = pt
                        return False
        return True

    def getSample(self):
        pt = np.array([[np.random.uniform(-1.0,1.0) * self.sample_r[0]],
                [np.random.uniform(-1.0,1.0) * self.sample_r[1]],
                [np.random.uniform(-1.0,1.0) * self.sample_r[2]]
                ])
        pt = self.R @ pt + self.translation
        return pt.ravel()

    def lineVisib(self, pos1, pos2, thresh, check_endpoints=True):
        dist = np.linalg.norm(pos2 -pos1)
        try:
            steps = int(np.floor(dist/thresh))
            ray = np.linspace(pos1, pos2, steps)
        except:
            breakpoint()
        if not check_endpoints:
            ray = ray[1:-1]

        for pos in ray:
            # if np.any(np.linalg.norm(pos[:2]-self.occup, axis=1) < thresh):
            #     return False
            if self.find_occupied(self.occup_grid, pos):
            # if self.occup.find_occupied(pos):
                return False
        return True 

    def pruneGraph(self, graph):
        if len(graph) > 2:
            for node1 in graph:
                if node1.id <= 1:
                    continue
                if len(node1.neighbors) <= 1:
                    for node2 in graph:
                        if node1 in node2.neighbors:
                            node2.neighbors.remove(node1)
                            break
                    graph.remove(node1)
        return graph

    def pruneEquivalent(self, paths):
        if paths is None:
            return None
        prune_paths = []
        # breakpoint()
        if len(paths) < 1:
            return prune_paths
        exist_paths_id = []
        exist_paths_id.append(0)
        for i, path1 in enumerate(paths):
            new_path = True
            for j in exist_paths_id:
                same_topo = self.sameTopoPath(path1, paths[j], self.topo_resolution)
                if same_topo:
                    new_path = False
                    break

            if new_path:
                exist_paths_id.append(i)
        for i in exist_paths_id:
            prune_paths.append(paths[i])
        return prune_paths
    def cutToMax(self, path, max_len):
        # len = 0
        # temp_path = self.discretizePath(path, 10)
        # positions = np.array([node.pos[0:2] for node in temp_path]) 
        st = time.time()
        np_path = np.array(path)
        lengths = np.cumsum(np.linalg.norm(np_path[:-1]-np_path[1:], axis=1))
        if lengths[-1] < max_len:
            return path
        else:
            last_exist = np.argmax(lengths>max_len)
            backtrack = lengths[last_exist] - max_len
            new_end = np_path[last_exist+1]-backtrack*(np_path[last_exist+1]-np_path[last_exist])/ \
                                        np.linalg.norm(np_path[last_exist+1]-np_path[last_exist])
            new_path = path[:last_exist+1]
            new_path.append(new_end)
            # breakpoint()
            return new_path

    def cutToSafe(self, path):
        temp_path, idx = self.discretizePath(path, 10)
        for i, pt in enumerate(temp_path):
            if self.find_occupied(self.occup_grid, pt, mark_out_of_bounds=True):
                break
        new_path = path[:idx[i]+1]
        new_path.append(temp_path[i])
        return new_path

    def selectSafeZonePaths(self, paths, step):
        if paths is None:
            return None
        min_cost = np.inf
        short_paths = []
        for i in range(self.reserve_num):
            path_id = self.safeZonePath(paths)
            if len(paths) <= 0:
                break
            if (i==0):
                short_paths.append(paths[path_id])
                min_cost = self.pathLength(paths[path_id])
                paths.pop(path_id)
            else:
                rat = self.pathLength(paths[path_id]) / min_cost
                if (rat < self.ratio_to_short):
                    short_paths.append(paths[path_id])
                    paths.pop(path_id)
                else:
                    break
        # for sp in short_paths:
        #     sp.insert(0, self.start_pts)
        #     sp.append(self.end_pts)
        for i, sp in enumerate(short_paths):
            # sp = self.shortcutPath(sp, i, 5)
            sp = self.lazyShortcut(sp, 10)
            # sp = sp
        short_paths = self.pruneEquivalent(short_paths)
        return short_paths

    def selectShortPaths(self, paths, step): 
        if paths is None:
            return None
        min_len = np.inf
        short_paths = []
        # breakpoint()
        for i in range(self.reserve_num):
            path_id = self.shortestPath(paths)
            if len(paths) <= 0:
                break
            if (i==0):
                short_paths.append(paths[path_id])
                min_len = self.pathLength(paths[path_id])
                paths.pop(path_id)
            else:
                rat = self.pathLength(paths[path_id]) / min_len
                if (rat < self.ratio_to_short):
                    short_paths.append(paths[path_id])
                    paths.pop(path_id)
                else:
                    break
        # for sp in short_paths:
        #     sp.insert(0, self.start_pts)
        #     sp.append(self.end_pts)
        for i, sp in enumerate(short_paths):
            # sp = self.shortcutPath(sp, i, 5)
            dis_path, _ = self.discretizePath(sp, 20)
            sp = self.lazyShortcut(dis_path, 10)
            # sp = self.lazyShortcut(sp, 10)
            # sp = sp
        # short_paths = self.pruneEquivalent(short_paths)
        return short_paths

    def sameTopoPath(self, path1, path2, thresh):
        len1 = self.pathLength(path1)
        len2 = self.pathLength(path2)
        max_len = np.maximum(np.maximum(len1, len2),2)

        pt_num = int(np.ceil(max_len/ thresh))
        pts1, _ = self.discretizePath(path1, pt_num)
        pts2, _ = self.discretizePath(path2, pt_num)
        if(np.any(np.isnan(pts1))) or (np.any(np.isnan(pts2))):
            breakpoint()
        for i in range(pt_num):
            if not self.lineVisib(pts1[i], pts2[i], self.ray_resolution, check_endpoints=False):
                return False
        return True

    def shortestPath(self, paths):
        short_id = -1
        min_len = np.inf
        for i, path in enumerate(paths):
            len = self.pathLength(path)
            if len<min_len:
                short_id = i
                min_len = len
        return short_id

    def pathLength(self, path):
        length = 0.0
        if len(path) < 2:
            return length
        
        for i in range(len(path)-1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length


    def safeZonePath(self, paths):
        short_id = -1
        min_cost = np.inf
        for i, path in enumerate(paths):
            cost = self.safeZoneCost(path)
            if cost<min_cost:
                short_id = i
                min_cost = cost
        return short_id

    def safeZoneCost(self, path):
        cost = 0
        for pt in path:
            cost += np.sum(pt-self.safe_zones)
        # szs = np.tile(self.safe_zones, (len(path), 1))
        # path_tiled =  np.tile(path, (len(szs), 1))
        return cost


    def discretizePath(self, path, pt_num):
        # breakpoint()
        len_list = [0.0]
        for i in range(len(path)-1):
            length = np.linalg.norm(path[i+1] - path[i])
            len_list.append(length+len_list[i])
        len_total = len_list[-1]
        dl = len_total / (pt_num-1.0)
        cur_l = 0.0
        dis_path = []
        path_idx = []
        for i in range(pt_num) :
            cur_l = i * dl
            idx = -1
            for j in range(len(len_list)-1):
                if (cur_l >= len_list[j] - 1e-4) and cur_l <= len_list[j+1] + 1e-4:
                    idx = j
                    break
            l = (cur_l - len_list[idx]) / (len_list[idx+1] - len_list[idx])
            # breakpoint()
            inter_pt = (1-l) * path[idx] + l * path[idx+1]
            dis_path.append(inter_pt)
            path_idx.append(idx)

        # path_idx.append(len(path))
        return dis_path, path_idx

    # def shortcutPath(self, path, path_id, iter_num):
    #     short_path = path

    #     for k in range(iter_num):
    #         dis_path = self.discretizePath(short_path, 10)
    #         if len(dis_path) < 2
    #             short_ 

    #     return sho

    def shortcutPaths(self, paths):
        if paths is None:
            return None
        cut_paths = []
        for path in paths:
            dis_path,_ = self.discretizePath(path, 20)
            # path = self.lazyShortcut(dis_path, 50)
            path = self.expShortcut(dis_path)
            # after = len(path)
            cut_paths.append(path)
        # return paths
        return cut_paths

    def lazyShortcut(self, dis_path, iter_num):
        start_time = time.time()
        # dis_path, _ = self.discretizePath(path, discretize)
        # if len(path) <=3:
        #     return path
        for i in range(iter_num):
            path_len = len(dis_path)
            ind1 =  round(np.random.rand()*(path_len-1))
            ind2 =  round(np.random.rand()*(path_len-1))
            if ind2 < ind1:
                ind1, ind2 = ind2, ind1
            while ind2 - ind1 < 1:
                ind1 =  round(np.random.rand()*(path_len-1))
                ind2 =  round(np.random.rand()*(path_len-1))
                if ind2 < ind1:
                    ind1, ind2 = ind2, ind1
            if self.lineVisib(dis_path[ind1], dis_path[ind2], self.ray_resolution):
                del dis_path[ind1+1:ind2]

        elapsed = time.time() - start_time
        return dis_path
    
    def expShortcut(self, path):
        start_time = time.time()
        dis_path, _ = self.discretizePath(path, 20)
        # if len(path) <=3:
        #     return path
        new_path = [dis_path[0]]
        last_index = 0
        for i, pt in enumerate(dis_path[2:]):
            if not self.lineVisib(pt, new_path[-1], self.ray_resolution):
                new_path.append(dis_path[i+1])
                # temp_path = dis_path[last_index:(i+1)]
                # new_path.extend(self.lazyShortcut(temp_path, 50)[1:])
                # last_index = i
        new_path.append(dis_path[-1])

        elapsed = time.time() - start_time
        return new_path
    def discretizeLine():
        pass
    def discretizePaths():
        pass
    def getOrthoPoint():
        pass
    def searchPaths(self,graph):
        raw_paths = []
        filter_raw_paths = []
        visited = [graph[0]]
        raw_paths = self.depthFirstSearch(visited, raw_paths) 
        min_node_num = np.inf
        max_node_num = 1
        path_list = [[] for i in range(100)]

        for i, path in enumerate(raw_paths):
            if len(path) > max_node_num:
                max_node_num = len(path)
            if len(path) < min_node_num:
                min_node_num = len(path)

            path_list[len(path)].append(i)
        if max_node_num==1:
            # breakpoint()
            return None
        for i in range(min_node_num, max_node_num+1):
            reach_max = False
            for path in path_list[i]:
                filter_raw_paths.append(raw_paths[path])
                if len(filter_raw_paths) >= self.max_raw_path2:
                    reach_max = True
                    break
            if reach_max:
                break
        raw_paths = filter_raw_paths
        return raw_paths

    def depthFirstSearch(self, visited, raw_paths):
        cur = visited[-1]
        # breakpoint()
        for neighbor in cur.neighbors:
            if neighbor.id == 1:
                path = []
                for node in visited:
                    path.append(node.pos)
                path.append(neighbor.pos)
                raw_paths.append(path)
                if len(raw_paths) > self.max_raw_path:
                    return raw_paths
                break
        # breakpoint() 
        for neighbor in cur.neighbors:
            if neighbor.id == 1:
                continue
            revisit = False
            for node in visited:
                if neighbor.id == node.id:
                    revisit = True
                    break
            if revisit:
                continue

            visited.append(neighbor)
            self.depthFirstSearch(visited, raw_paths)
            if len(raw_paths) >= self.max_raw_path:
                return raw_paths
            visited.pop()
        return raw_paths

    def triangelVisib():
        pass
        
    @staticmethod
    def SampleUnitNBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    @staticmethod
    def calc_dist(start, end):
        return math.hypot(start.x - end.x, start.y - end.y)

    @staticmethod
    def calc_dist_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

def main():
    # np.random.seed(0)
    # obs = np.array([[-12.5, -5,2.5],
    #                 [-15,0,1],
    #                 [-5, 2, 2],
    #                 [-20,0, 4],
    #                 [-22,-4.0, 4],
    #                 # [-25,-7, 4],
    #                 # [-30,-6, 4]
    #                 ])
    np.random.seed(0)
    # obs = np.array([[-28.068606760949095, 3.010129648108098, 1.0080897195797134], 
    #                 [-16.25489196522279, -2.4713306518253466, 0.5500006870450237], 
    #                 [-21.33570793603018, -1.8354193880368306, 1.2459807107486034], 
    #                 [-15.258411457358244, 2.7329369067833458, 0.6747815346142402], [-10.043845589703295, -1.8771411475716824, 1.2697933026154733]])
    obs = np.random.uniform(size=(20,3), low = np.array([-30.0,-5.0,0.5]), high = np.array([3.5,3.5,1.5]))
    safe_zones = np.array([[-30.0, -5.0, 0.0], [3.5, 3.5, 0.0], [-2.778297862194872, -3.176028257697845, 0.0], [-11.471293465494202, -2.072494027745604, 0.0], [-29.216110703218426, 1.1292642436732496, 0.0], [-25.048643011614416, 2.0148486507713397, 0.0], [-26.219578771539478, -4.160093187259887, 0.0], [-14.707025904645588, 1.3192407802563944, 0.0], [-13.107551539668314, 1.1585229700835233, 0.0], [-26.336070365683952, 2.642097591864471, 0.0], [-27.315071583724, -3.1874147614653237, 0.0], [-8.937436667042604, -3.2496254705763787, 0.0], [-24.87510145052023, -1.7737466928215029, 0.0], [2.9419790149662006, -4.762460873647652, 0.0], [-16.8960951743667, -3.9431216258844275, 0.0], [-19.507187155217533, 2.990607547156066, 0.0], [1.0, 1.20141908822204, 0.0], [-19.375500631202613, 1.9868821917652983, 0.0], [1.0, 1.282696947945408, 0.0], [0.5545096817442925, -1.59167100061837, 0.0]])
    # safe_zones = np.array([[-30.0, 0.0, 0.0],
    #                        [-10.0, 0.0, 0.0],
    #                        [-6.0, -8.0, 0.0],
    #                        [-10.0, -8.0, 0.0],
    #                        [-15.0, -8.0, 0.0],
    #                        [-20.0, -8.0, 0.0],
    #                        [-24.0, -8.0, 0.0],
    #                        [-28.0, -8.0, 0.0],
    #                        [-32.0, -8.0, 0.0],
    #                        [-34.0, -6.0, 0.0],
    #                        [-6.0, -3.0, 0.0 ],
    #                        [-4.0, -4.0, 0.0 ], 
    #                        [3.5, 1.0, 0.0],
    #                        [-2.0, -6.0, 0.0],
    #                        [1.0,-3.0,0.0]])  # Safe States
    resolution = 0.4
    origin = np.array([-40,-10])
    wh = np.array([50/resolution, 20/resolution]) # width, height
    boundary = [[origin[0], origin[0]+wh[0]*resolution], [origin[1], origin[1]+wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)
    occupied = grid.find_all_occupied(obs)
    planner = TopoPRM(None, resolution=resolution, max_raw_path=20, max_raw_path2=20,reserve_num=6, ratio_to_short=20, sample_sz_p=0.05, max_time=1.0)
    planner.occup_grid = grid.occup_grid
    planner.origin = origin
    planner.resolution = resolution
    planner.wh = wh
    planner.safe_zones = safe_zones
    
    # start=np.array([-5.0,4.0, 0.0])
    # end=np.array([-20.0,-2.0, 0.0]    
    start=np.array([3.5, 3.5, -np.pi*0.75]) 
    # start=np.array([-10.0, 0.0, -np.pi*0.75]) 
    # start=np.array([-24.0, -8.0, -np.pi*0.75]) 
    # start=np.array([-5.0, -5.0, -np.pi*0.75]) 
    end = np.array([-30.0, -5.0, 0.0])  # Reference state)
    st = time.time()
    paths, samples = planner.findTopoPaths(start, end) 
    print(time.time()-st)
    print(len(paths))
    box = np.array([[3,3]])

    # for path in paths:
    #     p = np.array(path)[:,0:2]
    #     A, b = pdc.convex_decomposition_2D(occupied, p, box)
    # print(f"Total time: {time.time()-st}")
    
    for path in paths:
        p = np.array(path)[:,0:2]
        A, b = pdc.convex_decomposition_2D(occupied, p, box)
        path_cut = planner.cutToMax(path, 20)
        path_cut = path
        p_cut =  np.array(path_cut)[:,0:2]
        A_cut, b_cut = pdc.convex_decomposition_2D(occupied, p_cut, box)
        # ax = pdc.visualize_environment(Al=A, bl=b, p =p, planar=True)
        # pdc.visualize_environment(Al=A_cut, bl=b_cut, p =p_cut, planar=True)
        pdc.visualize_environment(Al=A, bl=b, p =p, planar=True)
        for circ in obs:
            circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
            plt.gca().add_artist(circle)
        # plt.show(block=False)
        # plt.waitforbuttonpress()
        # print(path)
        plt.show()
        # breakpoint()

    rect = plt.Rectangle((start[0]-0.25, start[1]-0.25), 0.5, 0.5)
    plt.gca().add_artist(rect)
    rect = plt.Rectangle((end[0]-0.25, end[1]-0.25), 0.5, 0.5)
    plt.gca().add_artist(rect)


    samples = np.array(samples)[:,:2] 

    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    # for node in planner.graph:
    #     pos = node.pos
    #     if node.type == NODE_TYPE.CONNECTOR:
    #         ls = '--'
    #     else:
    #         ls = '-'
    #     for neighbor in node.neighbors:
    #         plt.plot([pos[0], neighbor.pos[0]], 
    #                  [pos[1], neighbor.pos[1]], ls)    
    #     plt.scatter(pos[0],pos[1])
    #     # plt.waitforbuttonpress()

    #     # ax = plt.gca()
    #     # for line in ax.lines:
    #     #     line.remove()
    # plt.grid(True)
    # plt.xlim(boundary[0])
    # plt.ylim(boundary[1])
    # plt.scatter(samples[:,0], samples[:,1])
    # plt.show()
    for path in paths:
        plt.plot(np.array(path)[:,0], np.array(path)[:,1])

    plt.plot(safe_zones[:,0], safe_zones[:,1], 'bo', label="safe zones")
    plt.legend()
    plt.xlim(boundary[0])
    plt.ylim(boundary[1])
    plt.show()
    


if __name__ == '__main__':
    main()