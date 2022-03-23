from .path_planner import Path_Planner

class Path_Not_Found_Error(Exception):
    """Raised when no path can be found to the given target"""
    pass

class Navigation_Controller:
    def __init__(self, mapping, localiser):
        self._maps = mapping
        self._location = localiser

        self._path_planner = Path_Planner(self._maps.get_shape())
        self._current_goal = None
        self._next_step = 1
      
    def reached_goal(self):
        pos = self._location.get_pos()

        return self._current_goal is None or \
            (self._current_goal[0] == pos[0] and self._current_goal[1] == pos[1])

    def set_goal(self, new_goal):
        """
        Sets the new goal location in the world.

        # Arguments:
            goal_pos: Tuple (Int, Int)
        """
        if self._current_goal != new_goal:
            self._current_goal = new_goal
            try:
                self._calculate_path()
                self._next_step = 1
            except Path_Not_Found_Error:
                raise Path_Not_Found_Error()

    def next_move(self):
        """
        Gets the next action to do based on our current position and the result of the a_star
        search from the path planner.

        # Returns
            Action to do next: Str ("left", "right" "forward")
            or None if no action can be made
        """
        
        pos = self._location.get_pos()
        assert pos != None

        if not self._path_is_legal():
            self._calculate_path()

        # If we don't have a path, there is no next action to take
        if self._path == None:
            return None

        # If we're not at our current goal
        if not self.reached_goal():
            (next_y, next_x) = self._path.path_get(self._next_step)
            # TODO: Check if the next square is next to us (we might have gone the wrong way :/ )
            
            (y, x) = pos

            delta_x = next_x - x
            delta_y = next_y - y

            assert delta_x != delta_y   # no diagonals and no '0' moves

            # change in y
            UP = 0
            RIGHT = 1
            DOWN = 2
            LEFT = 3
            # NO_ACTION = 4

            move = 4

            if delta_x < 0:
                move = LEFT
            elif delta_x > 0:
                move = RIGHT
            elif delta_y < 0:
                move = UP
            elif delta_y > 0:
                move = DOWN
            else:
                assert False # we shouldn't be here due our previous assertion

            self._next_step += 1
            return move

    def _path_is_legal(self):
        # if any node on our path is occupied, the path is not legal
        if self._path == None:
            return False

        obstacle_map = self._maps.get_maps()["obstacles"]
        for i in range(self._path.path_len()):
            node = self._path.path_get(i)
            if obstacle_map[node]:
                return False

        return True

    def _calculate_path(self):
        self._path = self._path_planner.compute_route(self._location.get_pos(), self._current_goal, self._maps.get_maps())
        if self._path == None:
            raise Path_Not_Found_Error()

        