import json
import numpy as np

class EpisodeLogger():
    def __init__(self, world, output_dir):
        self._world = world
        self._log_file = open(output_dir, 'w')
        self._log_file.write("{\"episodes\":[")

    def episode_reset(self):
        self._step_count = 0
        self._reward_sum = 0
        self._objective_function_sum = 0
        self._distance_travelled_sum = 0
        self._local_interactions = 0

    def capture_step(self, reward, explored_count, map_size, distance_travelled, local_interactions):
        self._step_count += 1
        self._reward_sum += reward
        self._distance_travelled_sum += distance_travelled
        self._local_interactions += local_interactions

        if self._distance_travelled_sum > 0:
            self._objective_function_sum += (100 * explored_count / map_size) / self._distance_travelled_sum

        # with open("step.csv", 'a') as step_file:
        #     output_tuple = (explored_count, map_size, distance_travelled, (100*explored_count/map_size)/self._distance_travelled_sum)
        #     step_file.write(json.dumps(output_tuple) + "\n")

    def capture_episode(self, episode_idx, final_episode):
        output_log_text = {
            "episode":episode_idx,
            "reward":self._reward_sum,
            "total_steps":self._step_count,
            "dist_travelled":self._distance_travelled_sum,
            "local_interactions":self._local_interactions,
            "obj_function":self._objective_function_sum
        }

        self._log_file.write(json.dumps(output_log_text))
        if not final_episode:
            self._log_file.write(",\n")
        self.episode_reset()

    def close(self):
        self._log_file.write("]}")
        self._log_file.close()