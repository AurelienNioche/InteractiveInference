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
        self.base_radius = 5
        self.var_radius = 50
        self.width_circle_line = 2
        self.margin = 0.05

        # ---------------------------------------- #

        self.window = Window(fps=self.fps, hide_cursor=self.hide_cursor)

        # --------------------------------------- #

        self.n_target = len(colors)
        self.init_position = np.zeros((self.n_target, 2))
        self.color_still = np.zeros(self.n_target, dtype=object)
        for i in range(self.n_target):
            self.init_position[i] = np.random.random(2) * self.window.size()
            self.color_still[i] = colors[i]

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

        self.angle[:] = self.rng.choice([0, 90], size=self.n_target)#90 #self.rng.uniform(0, 360, size=self.n_target)

        for i in range(self.n_target):
            angle = self.angle[i]
            x, y = self.movement[i]

            if angle <= 90 or angle >= 270:
                x_prime = np.abs(x)
            else:
                x_prime = - np.abs(x)

            y_prime = np.tan(np.radians(angle)) * x_prime

            norm = self.constant_amplitude / np.sqrt(y_prime**2 + x**2)
            y_norm = y_prime * norm
            x_norm = x_prime * norm
            print(f"target {i}: old x={x:.3f}, y={y:.3f} || y_prime={y_prime:.3f} norm={norm:.3f} || new x={x_norm:.3f}, y= {y_norm:.3f}")
            self.movement[i, :] = x_norm, y_norm


            if angle <= 90 or angle >= 270:
                x_prime = np.abs(x)
            else:
                x_prime = - np.abs(x)

            y_prime = np.tan(np.radians(angle)) * x_prime

            norm = self.constant_amplitude / np.sqrt(y_prime**2 + x**2)
            y_norm = y_prime * norm
            x_norm = x_prime * norm
            print(f"target {i}: {x_norm:.3f}, {y_norm:.3f}")
            self.movement[i, :] = x_norm, y_norm

        self.p_val[:] = 0
        self.selected[:] = 0
        self.pos[:] += self.movement

    def update_control(self, user_action):

        print("DEBUG NO UPDATE CONTROL")

        # self.window.move_mouse(movement=user_action)
        #
        # ctb = self.window.cursor_touch_border()
        #
        # if not ctb:
        #     mouse_pos = self.window.mouse_position
        #     delta_mouse = mouse_pos - self.mouse_pos_prev_frame
        #     self.mouse_pos_prev_frame[:] = mouse_pos
        #     add_to_ctl = delta_mouse * self.mouse_scale
        #
        # else:
        #     self.window.move_back_cursor_to_the_middle()
        #     self.mouse_pos_prev_frame[:] = self.window.mouse_position
        #     add_to_ctl = np.zeros(2)
        #
        # for coord in range(2):
        #     self.control[coord] += add_to_ctl[coord]

    def draw(self):

        visual_pos = np.zeros_like(self.pos)
        for coord in range(2):
            visual_pos[:, coord] = self.pos[:, coord] + self.control[coord]

        # visual_pos[:] = np.clip(visual_pos, -100, 100)

        # TODO: maintain pos withing window

        for i in range(self.n_target):
            visual_pos[i] += self.init_position[i]

        color = np.zeros(self.n_target, dtype=object)
        color[:] = self.color_still
        color[self.selected] = self.color_selected

        radius = self.base_radius + self.var_radius*self.p_val

        self.window.clear()

        for i in range(self.n_target):
            Circle(window=self.window,
                   position=visual_pos[i],
                   color=color[i],
                   radius=radius[i]).draw()

        self.window.update()

    def check_keys(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_a:
            #         self.phase_inc += 0.0005
            #         print(f"Increasing `phase_inc` (={self.phase_inc})")
            #     elif event.key == pygame.K_q:
            #         self.phase_inc -= 0.0005
            #         print(f"Decreasing `phase_inc` (={self.phase_inc})")
            #     elif event.key == pygame.K_z:
            #         self.disturbance_scale += 50
            #         print(f"Increasing `disturbance_scale` (={self.disturbance_scale})")
            #     elif event.key == pygame.K_s:
            #         self.disturbance_scale -= 50
            #         print(f"Decreasing `disturbance_scale` (={self.disturbance_scale})")

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

    # print("user_sigma", user_sigma)

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
            # print('user action', user_action)
        else:
            user_action = np.zeros(2)
            # print("NO USER CONTROL")
            # user_action = np.random.random(size=2)


if __name__ == "__main__":
    main()
