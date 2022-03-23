class Invalid_Move_Exception(Exception):
    """Robot cannot move to this tile or in instructed manner"""
    pass

class Localisation:
    """
    The localisation sub-module determines the robot's global position 𝑥𝑖 ∈ 𝑋, with respect to a world coordinate frame.
    """
    def get_state(self):
        return (self._pos, self._orientation)

    def get_pos(self):
        return self._pos
        
    def get_dir(self):
        return self._orientation

    def update_location(self, new_pos, new_dir):
        self._pos = new_pos
        self._orientation = new_dir