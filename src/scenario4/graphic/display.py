import sys
import numpy as np
import pygame

from graphic.components.window import Window
from graphic.components.shape import Circle


class Display:

    def __init__(self,
                 colors,
                 fps=30,
                 hide_cursor=False,
                 control_per_artificial_user=False,
                 control_using_mouse=False,
                 mouse_move_targets=False):

        """
        :param colors: Iterable. Define the number of targets.
        :param fps: Int. Frame per second.
        :param hide_cursor: Bool.
        :param control_per_artificial_user: Bool.
        :param control_using_mouse: Bool.
        :param mouse_move_targets: Bool.
        """

        self.fps = fps
        self.n_frame_selected = 20
        self.seed = 123
        self.mouse_scale = 0.5

        # --- Graphic parameters ---

        self.hide_cursor = hide_cursor

        self.line_scale = 4.0
        self.base_radius = 20
        self.var_radius = 50
        self.width_circle_line = 2
        self.margin = 0.05

        # ---------------------------------------- #

        self.window = Window(fps=self.fps, hide_cursor=self.hide_cursor)

        # --------------------------------------- #

        self.n_target = len(colors)
        self.colors = np.asarray(colors, dtype=object)
        self.color_selected = "red"

        # ---------------------------------------- #

        self.rng = np.random.default_rng(seed=self.seed)

        self.p_val = np.zeros(self.n_target)
        self.selected = np.zeros(self.n_target, dtype=bool)

        self.n_frame_since_selected = 0
        self.mouse_pos_prev_frame = np.zeros(2)
        self.control = np.zeros(2)
        self.pos = np.zeros((self.n_target, 2))

        # ------------------------------------------ #

        self.constant_amplitude = 3
        self.angle = np.zeros(self.n_target)

        self.movement = np.zeros((self.n_target, 2))

        # ------------------------------------------ #

        self.control_per_artificial_user = control_per_artificial_user
        self.control_using_mouse = control_using_mouse
        if self.control_using_mouse:
            assert self.control_using_mouse, "Mouse control should be on for the artificial user control to be on too"
        self.mouse_move_targets = mouse_move_targets
        # ------------------------------------------ #

        self.initialize()

    def initialize(self):

        while not np.isclose(self.window.mouse_position, self.window.center).all():
            self.window.move_back_cursor_to_the_middle()
            self.check_keys()
            self.window.update()

    def reset(self):

        self.p_val[:] = 0
        self.selected[:] = 0
        self.n_frame_since_selected = 0

        self.movement[:, 0] = self.constant_amplitude
        self.movement[:, 1] = 0

        self.window.move_back_cursor_to_the_middle()

    def implement_assistant_action(self, assistant_action):

        self.pos[:] = assistant_action

        self.p_val[:] = 0
        self.selected[:] = 0

    def update_control(self, user_action):

        if self.control_per_artificial_user:
            self.window.move_mouse(movement=user_action)

        ctb = self.window.cursor_touch_border()

        if self.mouse_move_targets:
            if not ctb:
                mouse_pos = self.window.mouse_position
                delta_mouse = mouse_pos - self.mouse_pos_prev_frame
                self.mouse_pos_prev_frame[:] = mouse_pos
                add_to_ctl = delta_mouse * self.mouse_scale

            else:
                self.window.move_back_cursor_to_the_middle()
                self.mouse_pos_prev_frame[:] = self.window.mouse_position
                add_to_ctl = np.zeros(2)

            for coord in range(2):
                self.control[coord] += add_to_ctl[coord]
                self.pos[:, coord] += self.control[coord]

        else:
            if ctb:
                self.window.move_back_cursor_to_the_middle()

    def draw(self):

        self.check_keys()

        colors = np.copy(self.colors)
        colors[self.selected] = self.color_selected

        radius = self.base_radius + self.var_radius*self.p_val

        self.window.clear()

        for i in range(self.n_target):
            Circle(window=self.window,
                   position=self.pos[i],
                   color=colors[i],
                   radius=radius[i]).draw()

        self.window.update()

    @staticmethod
    def check_keys():

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

    def update(self, user_action, assistant_action):

        self.update_control(user_action)

        if self.selected.any():
            self.draw()
            self.n_frame_since_selected += 1
            if self.n_frame_since_selected == self.n_frame_selected:
                self.reset()

        self.implement_assistant_action(assistant_action)
        self.draw()

    @property
    def target_positions(self):
        return self.pos
