import numpy as np
import sys
import pygame

from graphic.window import Window
from graphic.shape import Circle

from users.users import User

np.seterr(all='raise')


def distance(pos1, pos2):
    return np.sqrt(((pos1 - pos2) ** 2).sum())


class Assistant:

    def __init__(self, colors, hide_cursor=False):

        self.fps = 30

        self.n_frame_selected = 20

        self.seed = 123

        self.mouse_scale = 0.5

        self.init_frames = 5

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
        self.init_position = np.zeros((self.n_target, 2))
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

        self.initialize()
        self.reset()

    def initialize(self):

        for i in range(self.n_target):
            self.pos[i] = self.rng.random(2) * self.window.size()

        # To be sure that the mouse can be captured correctly...
        for i in range(self.init_frames):
            self.window.clear()
            self.window.update()

    def reset(self):

        self.p_val[:] = 0
        self.selected[:] = 0
        self.n_frame_since_selected = 0

        self.movement[:, 0] = self.constant_amplitude
        self.movement[:, 1] = 0

        self.window.move_back_cursor_to_the_middle()

    def update(self):

        self.angle[:] = self.rng.choice([0, 90], size=self.n_target)  # 90 #self.rng.uniform(0, 360, size=self.n_target)

        for i in range(self.n_target):
            angle = self.angle[i]
            x, _ = self.movement[i]

            x_prime = np.abs(x)
            if 90 > angle < 270:
                x_prime *= -1

            y_prime = np.tan(np.radians(angle)) * x_prime

            norm = self.constant_amplitude / np.sqrt(y_prime**2 + x_prime**2)
            self.movement[i, :] = np.asarray([x_prime, y_prime]) * norm

        self.p_val[:] = 0
        self.selected[:] = 0
        self.pos[:] += self.movement

    def update_control(self, user_action):

        print("DEBUG NO UPDATE CONTROL")

        self.window.move_mouse(movement=user_action)

        ctb = self.window.cursor_touch_border()

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

    def draw(self):

        visual_pos = np.zeros_like(self.pos)
        max_coord = self.window.size()
        for coord in range(2):
            vp = self.pos[:, coord] + self.control[coord]
            vp = np.clip(vp, 0, max_coord[coord])
            visual_pos[:, coord] = vp

        colors = np.copy(self.colors)
        colors[self.selected] = self.color_selected

        radius = self.base_radius + self.var_radius*self.p_val

        self.window.clear()

        for i in range(self.n_target):
            Circle(window=self.window,
                   position=visual_pos[i],
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

    def step(self, user_action):

        self.check_keys()

        self.update_control(user_action)

        if self.selected.any():
            self.n_frame_since_selected += 1
            if self.n_frame_since_selected == self.n_frame_selected:
                self.reset()
        else:
            self.update()

        self.draw()

        return self.pos


def main():

    hide_cursor = True

    user_control = False

    user_goal = 0
    user_sigma = 0.3
    user_alpha = 0.5

    colors = "orange", "blue"

    n_target = len(colors)

    model = Assistant(colors=colors, hide_cursor=hide_cursor)
    model.reset()
    user = User(n_target=n_target, sigma=user_sigma, goal=user_goal, alpha=user_alpha)
    user.reset()

    user_action = np.zeros(2)

    if not user_control:
        print("DEBUG MODE: NO USER CONTROL")

    while True:
        model_action = model.step(user_action=user_action)
        if user_control:
            user_action, _, _, _ = user.step(model_action)
            user_action *= 2
        else:
            user_action = np.zeros(2)


if __name__ == "__main__":
    main()
