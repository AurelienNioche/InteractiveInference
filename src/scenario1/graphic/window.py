import numpy as np
import pygame


class Window:

    def __init__(self, size=(800, 600), fps=30, background="white", hide_cursor=True, caption=""):

        pygame.init()
        self.surface = pygame.display.set_mode(size, 0, 32)
        pygame.display.set_caption(caption)
        self.fps = fps  # frames per second setting
        self.fps_clock = pygame.time.Clock()

        self.background = pygame.Color(background)

        if hide_cursor:
            pygame.mouse.set_visible(False)

    def clear(self):
        self.surface.fill(self.background)

    def update(self):

        pygame.display.update()
        self.fps_clock.tick(self.fps)

    def cursor_touch_border(self):

        x, y = pygame.mouse.get_pos()
        x_max, y_max = self.surface.get_size()

        return np.isclose(x, 0, atol=0.01) \
            or np.isclose(x, x_max, atol=0.01) \
            or np.isclose(y, 0, atol=0.01) \
            or np.isclose(y, y_max, atol=0.01)

    def move_back_cursor_to_the_middle(self):

        pygame.event.set_grab(True)
        x_max, y_max = self.surface.get_size()
        x = 0.5*x_max
        y = 0.5*y_max

        pygame.mouse.set_pos(x, y)

    @property
    def mouse_position(self):

        x, y = pygame.mouse.get_pos()
        return np.array([x, y])

    @staticmethod
    def move_mouse(movement):

        x, y = pygame.mouse.get_pos()

        mv_x, mv_y = movement

        x += mv_x
        y += mv_y

        pygame.mouse.set_pos(x, y)

    def size(self):
        return np.array(self.surface.get_size())

    def center(self):
        return self.size() / 2

    def quadrant_center(self, quadrant):

        if quadrant == "first":
            return (self.size()[0] - self.center()[0]) / 2, self.center()[1] / 2
        elif quadrant == "second":
            return self.center()[0] / 2, self.center()[1] / 2
        elif quadrant == "third":
            return self.center()[0] / 2, (self.size()[0] - self.center()[0]) / 2
        elif quadrant == "fourth":
            return (self.size()[0] - self.center()[0]) / 2, (self.size()[0] - self.center()[0]) / 2
        else:
            raise ValueError
