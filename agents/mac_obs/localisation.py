class Invalid_Move_Exception(Exception):
    """Robot cannot move to this tile or in instructed manner"""
    pass

class Localisation:
    """
    The localisation sub-module determines the robot's global position 𝑥𝑖 ∈ 𝑋, with respect to a world coordinate frame.
    """
    def __init__(self):
        self._pos = None

    def get_pos(self):
        return self._pos
        
    def update_location(self, new_pos):
        self._pos = new_pos
