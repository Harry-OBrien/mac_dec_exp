import numpy as np
from gym.envs.classic_control import rendering

class Renderer():
    def __init__(self, world, screen_size):
        self._world = world
        self._screen_size = screen_size
        self._height, self._width = self._world.shape
        self._viewer = None
        self._rendering_grid = np.full(self._world.shape, None)

        self._colours = [
            (255, 0, 0),        # Red
            (0, 255, 0),        # Green
            (0, 0, 255),        # Blue
            (255, 255, 0),      # Yellow
            (255, 0, 255),      # Purple
            (0, 255, 255)]      # Turqoise

    def _initialise_visual_grid(self):
        square_dimension = self._screen_size / self._width
        self._viewer = rendering.Viewer(self._screen_size, self._screen_size)

        for i in range(self._height):
            for j in range(self._width):
                l, r, t, b = (
                    j * square_dimension,
                    (j + 1) * square_dimension,
                    i * square_dimension,
                    (i + 1) * square_dimension,
                )
                square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                border = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], True)

                self._rendering_grid[i][j] = square
                self._viewer.add_geom(square)
                self._viewer.add_geom(border)
    
    def render(self):
        if self._viewer is None:
            self._initialise_visual_grid()

        maps = self._world.maps

        for i in range(self._height):
            for j in range(self._width):
                # For some weird reason pyglet puts (0,0) in the bottom left, so we need to invert 'y' here
                square = self._rendering_grid[self._height - 1 - i][j]

                # If a robot exists in this square
                if (maps["robot_positions"][i, j] != 0):
                    #robot's pos
                    agent = maps["robot_positions"][i, j] - 1
                    square.set_color(*self._colours[agent])

                # if unexplored
                elif not maps["explored_space"][i, j]:
                    if (not maps["obstacles"][i, j]):
                        square.set_color(0.8, 0.8, 0.8)
                    else:
                        square.set_color(0.3, 0.3, 0.3)

                # Square is explored and blocked
                elif (maps["obstacles"][i, j]):
                    square.set_color(0, 0, 0)

                # square is explored and empty
                else:
                    square.set_color(1, 1, 1)

        return self._viewer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()