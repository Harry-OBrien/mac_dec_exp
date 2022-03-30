from .path_planner import Path_Planner
# from ...env.actions import Action

class PathNotFoundError(Exception):
    """Raised when no path can be found to the given target"""
    pass

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NO_ACTION = 4

class Navigation_Controller:
    def __init__(self, mapping, localiser, teammate_detector):
        self._maps = mapping
        self._location = localiser
        self._teammate_detector = teammate_detector

        self._path_planner = Path_Planner(self._maps.get_shape())
        self._path = None
        self._current_goal = None

    def episode_reset(self):
        self._current_goal = None

    def reached_goal(self):
        pos = self._location.get_pos()
        goal_reached = self._current_goal is None or self._same_location(pos, self._current_goal)

        # if not at goal, check if the path is still legal
        if not goal_reached and not self._path_is_legal():
            try:
                # try and recalculate path if we're stuck
                self._calculate_path()
            except PathNotFoundError:
                # Not able to reach goal
                return True

        return goal_reached

    def set_goal(self, new_goal):
        """
        Sets the new goal location in the world.

        # Arguments:
            goal_pos: Tuple (Int, Int)
        """
        # if not set or is different
        if self._current_goal is None or not self._same_location(self._current_goal, new_goal):
            self._current_goal = new_goal
        try:
            self._calculate_path()
        except PathNotFoundError:
            self._current_goal = None
            raise PathNotFoundError()
        
    def _same_location(self, point_1, point_2):
        return point_1[0] == point_2[0] and point_1[1] == point_2[1]

    def next_move(self):
        """
        Gets the next action to do based on our current position and the result of the a_star
        search from the path planner.

        # Returns
            Action to do next in range n_actions or None if no action can be made
        """
        pos = self._location.get_pos()
        assert pos != None

        # If we don't have a path, there is no next action to take
        if self._path == None:
            print("WARN: Asked for the next move in a path before setting a goal location")
            return NO_ACTION

        if self._path.path_len() == 1:
            return NO_ACTION

        # check that we've moved since last time
        target_path_pos = self._path.path_get(self._next_step)
        if self._same_location(pos, target_path_pos) and self._path_is_legal():
            self._current_step = self._next_step
            self._next_step += 1
        else:
            try:
                self._calculate_path()
            except PathNotFoundError:
                self._current_goal = None
                raise PathNotFoundError()

        # If we're not at our current goal
        if not self.reached_goal():
            (next_y, next_x) = self._path.path_get(self._next_step)

            (y, x) = pos

            delta_x = next_x - x
            delta_y = next_y - y

            # We might have (randomly) gone the wrong way because of the 'movement_failure_prob' param
            # in the environment, so just recalculate the path to our goal
            if (delta_x >= 1 and delta_y >= 1) or (delta_x <= -1 and delta_y <= -1):
                self._calculate_path()
                (next_y, next_x) = self._path.path_get(self._next_step)
                delta_x = next_x - x
                delta_y = next_y - y

            assert delta_x != delta_y   # no diagonals and no '0' moves

            move = NO_ACTION

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

            return move
        else:
            print("WARN: somehow we've reached the goal but you're still asking for the next move???")
            return NO_ACTION

    def _path_is_legal(self):
        # if any node on our path is occupied, the path is not legal
        if self._path is None:
            return False

        obstacle_map = self._maps.get_maps()["obstacles"]

        for i in range(self._path.path_len()):
            node = self._path.path_get(i)
            if obstacle_map[node]:
                return False

        return True

    def _calculate_path(self):
        local_maps = self._maps.get_maps()
        obstacle_map = local_maps["obstacles"] | local_maps["robot_positions"]

        self._path = self._path_planner.compute_route(self._location.get_pos(), self._current_goal, obstacle_map)
        if self._path == None:
            raise PathNotFoundError()
        
        self._current_step = 0
        self._next_step = 1

        