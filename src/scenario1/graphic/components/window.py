import numpy as np
import pygame


class Window:

    def __init__(self, size=(800, 600), fps=30, background="white", hide_cursor=True, caption=""):

        pygame.init()
        self.surface = pygame.display.set_mode(size, 0, 32)
        pygame.display.set_caption(caption)
        self.fps = fps
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
            or np.isclose(x, x_max, atol=20.0) \
            or np.isclose(y, 0, atol=0.01) \
            or np.isclose(y, y_max, atol=20.0)

    def move_back_cursor_to_the_middle(self):

        pygame.event.set_grab(True)
        pygame.mouse.set_pos(*self.center)

    @property
    def center(self):
        return 0.5*self.size()

    @property
    def mouse_position(self):
        return np.asarray(pygame.mouse.get_pos())

    def move_mouse(self, movement):

        mouse_pos = self.mouse_position.astype(np.float64)
        mouse_pos += movement
        pygame.mouse.set_pos(*np.round(mouse_pos))

    def size(self):
        return np.asarray(self.surface.get_size())

    def quadrant_center(self, quadrant):

        x_span, y_span = self.size()

        if quadrant == "first":
            return 0.75*x_span, 0.25*y_span
        elif quadrant == "second":
            return 0.25*x_span, 0.25*y_span
        elif quadrant == "third":
            return 0.25*x_span, 0.75*y_span
        elif quadrant == "fourth":
            return 0.75*x_span, 0.75*y_span
        else:
            raise ValueError
