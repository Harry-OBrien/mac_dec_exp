import numpy as np
from . import weight
from .priority_queue import DPQ_t
from .path import path_t
from . import general

class Path_Planner:
    def __init__(self, map_dim):
        self._map_height, self._map_width = map_dim
        self._num_nodes = self._map_height * self._map_width

    def compute_route(self, start_pos, end_pos, obstacle_map):
        """
        Computes a route between two positions using a given map

        # Arguments
            start_pos - start point of the route: tuple (int, int)
            end_pos - end point of the route: tuple (int, int)
            maps - dictionary of maps that represent an agent's known view of the world: Dict { String:Array<Array<Int>> }

        # Returns
            Distance between src and dst as a float
        """
        assert start_pos is not None or end_pos is not None
        assert obstacle_map is not None

        # perform an a* path finding search
        result = self.a_star_search(obstacle_map, start_pos, end_pos)

        return result

    def get_distance(self, src, dst):
        """
        Calculates the distance between two locations on the gridworld

        # Arguments
            src: tuple (int, int)
            dst: tuple (int, int)

        # Returns
            Distance between src and dst as a float
        """
        a = src[0] - dst[0]
        b = src[1] - dst[1]

        return np.sqrt(a**2 + b**2)

    def _calculate_heuristic(self, point, goal, occupation_map):
        """
        Calculates the heuristic value for all points in the map and the end position

        # Arguments
            point: tuple (int, int) - start point
            goal: tuple (int, int)
            occupation_map: Array<Array<Int>>

        # Returns
            heuristic map: Array<Array<Int>>
        """
        assert(goal != None)
        assert(occupation_map is not None)

        if occupation_map[point] == 1:
            return weight.weight_inf()
        else:
            return weight.weight_t(self.get_distance(point, goal))

    def _get_neighbouring_nodes(self, u, occupancy_map):
        """
        Gets all the neighbouring nodes to a given point in the map that is not occupied by something

        # Arguments
            u - starting node: (Int, Int)
            occupancy_map - a map of occupied positions in the world: Array<Array<Int>>

        # Returns
            list of all unoccupied neighbouring nodes: [(Int, Int)]
        """
        neighbours = []
        neighbour_pos = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for pos in neighbour_pos:
            # generate the neighbour position
            v = (u[0] + pos[0], u[1] + pos[1])

            # If in range and not obstructed
            if v[0] >= 0 and v[0] < self._map_height\
                and v[1] >= 0 and v[1] < self._map_width\
                and occupancy_map[v] != 1:
                    neighbours.append(v)

        return neighbours

    def _idx_to_flat_idx(self, idx_2d):
        return np.int(idx_2d[0] * self._map_width + idx_2d[1])

    def _flat_idx_to_idx(self, flat_idx):
        row = np.floor(flat_idx / self._map_width)
        col = flat_idx % self._map_width

        return (np.int(row), np.int(col))

    def a_star_search(self, occupancy_map, src, dst):
        """
        Finds a path between src and dst using a* search.

        # Arguments
            occupancy_map - a map of occupied positions in the world: Array<Array<Int>>
            src - starting location: (Int, Int)
            dst - starting location: (Int, Int)
            
        # Returns
            either a path from src to dst or None if no path can be found.
        """
        # 2D array listing the heuristic values for every node to the dst node: Array<Array<Int>>
        h = np.full(occupancy_map.shape, None)

        # Create a priority queue
        pq = DPQ_t(self._num_nodes)

        # f <- 0
        finished = np.array([[False] * self._map_width for _ in range(self._map_height)])
        finished[src] = True

        # Create a vector for predecessors and distances and init all as inf or invalid
        pred = np.array([[(None, None)] * self._map_width for _ in range(self._map_height)])
        pred[src] = src

        dist = np.array([[weight.weight_inf()] * self._map_width for _ in range(self._map_height)])
        dist[src] = weight.weight_zero()

        # insert the starting node into the queue
        pq.DPQ_insert(self._idx_to_flat_idx(src), weight.weight_zero())

        # Looping till priority queue becomes empty
        while not pq.DPQ_is_empty():
            u = pq.DPQ_pop_min()
            u = self._flat_idx_to_idx(u)

            if u == dst:
                break

            successors = self._get_neighbouring_nodes(u, occupancy_map)
            finished[u] = True

            for v in successors:
                new_weight = weight.weight_add(dist[u], weight.weight_one())
                if (weight.weight_less(new_weight, dist[v])):
                    dist[v] = new_weight

                    # If we haven't previously calculated the heuristic value for this cell, do so
                    if h[v] is None:
                        h[v] = self._calculate_heuristic(v, dst, occupancy_map)

                    assert(h[v].weight_is_finite())
                    priority = weight.weight_add(new_weight, h[v])
                    pred[v] = u

                    v_flat = self._idx_to_flat_idx(v)
                    if pq.DPQ_contains(v_flat):
                        pq.DPQ_decrease_key(v_flat, priority)
                    else:
                        pq.DPQ_insert(v_flat, priority)

        return path_t(pred, dst) if dist[dst].weight_is_finite() else None