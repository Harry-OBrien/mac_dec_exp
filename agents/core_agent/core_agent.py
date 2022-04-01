import sys
sys.path.append("../")
sys.path.append("../../env/")
from ..mac_action import low_level_controller
from ..mac_obs import goal_extraction, localisation, mapping, teammate_detection
from env.actions import Action


class CoreAgent():
    def __init__(self, id, env, map_dim, observation_dim, frontier_width):
        self.id = id
        self._numerical_id = int(id[-1])

        self._localiser = localisation.Localisation()
        self._teammate_detector = teammate_detection.Teammate_Detector(env, id)
        self._mapping = mapping.Local_Map(map_dim=map_dim, view_dim=observation_dim, our_numerical_id=self._numerical_id)
        self._goal_extraction = goal_extraction.Goal_Extractor(local_mapper=self._mapping, frontier_width=frontier_width)
        self._navigator = low_level_controller.Navigation_Controller(self._mapping, self._localiser, self._teammate_detector)

        self.reset_observations()

    def reset_observations(self):
        # print("resetting agent!")
        self._mapping.reset_maps()
        self._localiser.update_location(None)
        self._navigator.episode_reset()

        self._current_goal = None
        self._last_goal = None

        self.goal_candidates = None

    def update_local_observation(self, latest_observation):
        # Update location
        self._localiser.update_location(latest_observation["pos"])

        # update the observation for the teammate detector
        self._teammate_detector.update_observation(latest_observation["robot_positions"])

        # Append observation to map
        self._mapping.append_observation(
            obstacle_observation=latest_observation["obstacles"], 
            explored_observation=latest_observation["explored_space"], 
            robot_positions=latest_observation["robot_positions"],
            teammates_in_range=self._teammate_detector.teammate_in_range(),
            observation_bounds=latest_observation["bounds"])

        # find teammates and share data
        if self._teammate_detector.teammate_in_range():
            self._teammate_detector.communicate_with_team()

        # re-calculate potential goals
        self.goal_candidates = self._goal_extraction.generate_goals()
        self._mapping.set_goal_candidates(self.goal_candidates)

    @property
    def pos(self):
        return self._localiser.get_pos()

    @property
    def maps(self):
        return self._mapping.get_maps()

    @property
    def explored_ratio(self):
        return self._mapping.explored_ratio

    @property
    def reached_goal(self):
        return self._navigator.reached_goal()

    @property
    def can_move_to_goal(self):
        if self._current_goal is not None and not self._navigator._path_is_legal():
            try:
                self._navigator._calculate_path()
            except low_level_controller.PathNotFoundError:
                return False

        return self._current_goal is not None

    @property
    def current_goal(self):
        return self._current_goal

    @current_goal.setter
    def current_goal(self, new_goal):

        # check if we can reach the goal
        if new_goal is not None and self._navigator.can_reach_location(new_goal):
            self._current_goal = new_goal
            self._navigator.set_goal(new_goal)
        else:
            self._current_goal = None

    @property
    def last_goal(self):
        return self._last_goal if self._last_goal is not None else self.current_goal
        
    @property
    def next_action(self):
        if self._current_goal is None:
            return Action.NO_MOVEMENT

        # if not self.can_move_to_goal:
        #     return Action.NO_MOVEMENT
        
        return self._navigator.next_move()

    @property
    def can_see_teammate(self):
        return self._teammate_detector.teammate_in_range()

    @property
    def callbacks(self):
        return (self.data_tx_callback, self.data_rx_callback)

    def data_tx_callback(self):
        output_data = {
            "agent_id":self._numerical_id,
            "last_goal":self._last_goal if self._last_goal is not None else self._current_goal,
            "current_goal":self._current_goal,
            "maps":self._mapping.get_maps()
        }

        return output_data

    def data_rx_callback(self, data):
        # import the data into our maps
        self._mapping.import_shared_map(
            data["maps"], 
            data["agent_id"], 
            data["last_goal"], 
            data["current_goal"]
        )