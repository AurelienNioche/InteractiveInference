import sys
import numpy as np
import pygame
from graphic.shape import Line


class Window:

    def __init__(self, size=(800, 600), fps=30, background="white", hide_cursor=True, caption=""):

        pygame.init()
        self.size = size
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        pygame.display.update()
        self.fps_clock.tick(self.fps)

    def cursor_touch_border(self):

        x_coord, y_coord = pygame.mouse.get_pos()
        x_max, y_max = self.surface.get_size()
        x, y = x_coord / x_max, y_coord / y_max

        return np.isclose(x, 0, atol=0.01) \
               or np.isclose(x, 1, atol=0.01) \
               or np.isclose(y, 0, atol=0.01) \
               or np.isclose(y, 1, atol=0.01)

    def move_back_cursor_to_the_middle(self):

        pygame.event.set_grab(True)
        x, y = 0.5, 0.5
        x_max, y_max = self.surface.get_size()
        x_scaled = x * x_max
        y_scaled = y * y_max

        pygame.mouse.set_pos(x_scaled, y_scaled)

    @property
    def mouse_position(self):

        x_scaled, y_scaled = pygame.mouse.get_pos()
        x_max, y_max = self.surface.get_size()
        x = x_scaled / x_max
        y = y_scaled / y_max

        return np.array([x, y])

    def show_margins(self, margin):
        # Might be useful for debug
        for start_position, stop_position in (
                (margin, margin), (1 - margin, margin),
                (margin, margin), (margin, 1 - margin),
                (1 - margin, margin), (1 - margin, 1 - margin),
                (margin, 1 - margin), (1 - margin, 1 - margin),
        ):
            Line(window=self,
                 start_position=start_position,
                 stop_position=stop_position,
                 color='black', width=2)