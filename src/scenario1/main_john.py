import numpy as np
import sys
import pygame

from graphic.components.window import Window
from graphic.components.shape import Circle

from users.moving_average_user import User

np.seterr(all='raise')


def distance(pos1, pos2):
    return np.sqrt(((pos1 - pos2) ** 2).sum())


class JohnModel:

    def __init__(self, colors, hide_cursor=False):

        self.fps = 30

        # --- Model parameters ---
        self.add_coeff = 5.0  # `ADD_COEFF`
        self.decay_coeff = 1.15  # `DECAY_COEFF`
        self.decay_threshold = -0.05  # `DECAY_THRESH` / originally: -0.9
        self.add_threshold = -2.0  # `ADD_THRESH`  / originally: -2.0

        self.n_sin = 4  # `N_SIN`
        self.freq_min = 1
        self.freq_max = 51
        self.phase_min = 0
        self.phase_max = 2*np.pi

        self.phase_inc = 0.003    # `phase_inc`; maybe 0.0005
        self.disturbance_scale = 655.5  # Maybe 255.5
        self.lag_tau = 0.08

        self.selection_threshold = 0.955

        # self.time_window_sec = XXX
        # self.time_window = int(self.time_window_sec * self.window.fps)
        self.n_frame_var = 20    # `VAR_WIN`
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

        self.hist_pos = np.zeros((self.n_target, 2, self.n_frame_var))
        self.hist_control = np.zeros((2, self.n_frame_var))
        self.hist_model = np.zeros((self.n_target, 2, self.n_frame_var))

        rng = np.random.default_rng(seed=self.seed)
        self.freq = rng.uniform(size=(self.n_target, self.n_sin), low=self.freq_min, high=self.freq_max)
        self.phase = rng.uniform(size=(self.n_target, self.n_sin), low=self.phase_min, high=self.phase_max)

        self.var_ratio = np.zeros(self.n_target)
        self.selected = np.zeros(self.n_target, dtype=bool)
        self.n_frame_since_selected = 0
        self.disturbance_phase = 0  # Will be incremented
        self.control = np.zeros(2)
        self.pos = np.zeros((self.n_target, 2))
        self.mouse_pos_prev_frame = np.zeros(2)

        self.initialize()
        self.reset()

    def initialize(self):

        # To be sure that the mouse can be captured correctly...
        for i in range(self.init_frames):
            self.window.clear()
            self.window.update()

        self.pos[:] = 0.5

    def reset(self):

        self.var_ratio[:] = 0
        self.selected[:] = 0

        self.window.move_back_cursor_to_the_middle()

        self.control[:] = 0
        self.hist_control[:] = 0
        self.mouse_pos_prev_frame[:] = self.window.mouse_position

        for i in range(self.n_target):
            for coord in range(2):
                self.hist_pos[i, coord, :] = self.pos[i, coord]
                self.hist_model[i, coord, :] = self.pos[i, coord]

        self.n_frame_since_selected = 0
        # self.disturbance_phase = 0  # Will be incremented

    def update_control(self, user_action):

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

        self.hist_control = np.roll(self.hist_control, -1)
        self.hist_control[:, -1] = self.control

    def draw(self):

        p_val = self.var_ratio / 1e5

        # print(f"p_val {p_val}")

        self.selected[:] = p_val >= self.selection_threshold

        visual_pos = np.zeros_like(self.pos)
        for coord in range(2):
            visual_pos[:, coord] = self.pos[:, coord] + self.control[coord]

        visual_pos[:] = np.clip(visual_pos, -100, 100)

        for i in range(self.n_target):
            visual_pos[i] += self.init_position[i]

        color = np.zeros(self.n_target, dtype=object)
        color[:] = self.color_still
        color[self.selected] = self.color_selected

        radius = self.base_radius + self.var_radius*p_val

        self.window.clear()

        for i in range(self.n_target):
            Circle(window=self.window,
                   position=visual_pos[i],
                   color=color[i],
                   radius=radius[i]).draw()

        self.window.update()

    def update_deriv(self):

        # Update `hist_pos`
        self.hist_pos = np.roll(self.hist_pos, -1)
        self.hist_pos[:, :, -1] = self.pos

        # Updating `hist_model`
        new_val = np.zeros((self.n_target, 2))
        for i in range(self.n_target):
            for coord in range(2):
                pos = self.pos[i, coord]
                hm = self.hist_model[i, coord, -2]
                new_val[i, coord] = (1 - self.lag_tau) * hm + self.lag_tau * pos
        self.hist_model = np.roll(self.hist_model, -1)
        self.hist_model[:, :, -1] = new_val

        # Compute `c_var` (accumulation of control change)
        var = np.zeros(2)
        for frame in range(1, self.n_frame_var):
            for coord in range(2):
                c_now = self.hist_control[coord, frame]
                c_prev_frame = self.hist_control[coord, frame - 1]
                diff = c_now - c_prev_frame
                var[coord] += diff**2
        c_var = distance(np.zeros(2), var)
        if c_var > 2000:
            c_var = 2000

        # print(f"c_var {c_var}")

        # Compute `e_var` (vector norm of 2D variances of disturbance)
        e_var = np.zeros(self.n_target)
        for i in range(self.n_target):
            var = np.zeros(2)
            for coord in range(2):
                h = self.hist_pos[i, coord, :]
                diff = h - h.mean()
                var[coord] = (diff**2).sum()
            e_var[i] = distance(np.zeros(2), var)

        # print(f"e_var {e_var}")

        # Compute `a_var` (accumulation of object movement under control)
        a_var = np.zeros(self.n_target)
        for i in range(self.n_target):
            var = np.zeros(2)
            for coord in range(2):
                hc = self.hist_control[coord, :]
                hm = self.hist_model[i, coord, :]
                add = hc + hm
                var[coord] = (add**2).sum()
            a_var[i] = distance(np.zeros(2), var)

        # print(f"a_var {a_var}")

        # Compute `var_rat`
        var_rat = np.zeros(self.n_target)
        for i in range(self.n_target):
            ratio = (e_var[i]+22000) / (a_var[i]+1e-5)
            # print("ratio",i, ratio)
            ratio = 1.0 - np.sqrt(ratio)
            # print("ratio",i, ratio)
            var_rat[i] = ratio
        var_rat *= c_var / 1000.0
        # print(f"var_rat {var_rat}")

        # Update var_ratio depending on the threshold
        need_add = var_rat < self.add_threshold
        for i in np.nonzero(need_add)[0]:
            # print("NEED ADD", i)
            self.var_ratio[i] -= var_rat[i] * self.add_coeff
        need_decay = var_rat > self.decay_threshold
        for i in np.nonzero(need_decay)[0]:
            # print("NEED DECAY", i)
            self.var_ratio[i] /= self.decay_coeff

        # print("var_ratio", self.var_ratio)

        # Apply normalization
        norm_ratio = np.sum(self.var_ratio + 0.4)
        self.var_ratio += 0.4
        self.var_ratio /= norm_ratio
        # print(f"var_ratio {self.var_ratio}")
        self.var_ratio *= 10e4
        # print(f"var_ratio {self.var_ratio}")

    def update_wave(self):

        self.disturbance_phase += self.phase_inc

        disturbance = \
            (np.sin(self.disturbance_phase * self.freq
                    + self.phase) / 1.7) ** 9 \
            * self.disturbance_scale

        for i in range(self.n_target):
            total_d = np.zeros(2)
            for j in range(0, self.n_sin, 2):
                for k in range(2):
                    total_d[k] += disturbance[i, j + k]

            # total_d = (1 / (1 + np.exp(-total_d * 4)) - 0.5) * 2
            self.pos[i] += total_d

        self.pos = np.clip(self.pos, -100, 100)

    def check_keys(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.phase_inc += 0.0005
                    print(f"Increasing `phase_inc` (={self.phase_inc})")
                elif event.key == pygame.K_q:
                    self.phase_inc -= 0.0005
                    print(f"Decreasing `phase_inc` (={self.phase_inc})")
                elif event.key == pygame.K_z:
                    self.disturbance_scale += 50
                    print(f"Increasing `disturbance_scale` (={self.disturbance_scale})")
                elif event.key == pygame.K_s:
                    self.disturbance_scale -= 50
                    print(f"Decreasing `disturbance_scale` (={self.disturbance_scale})")

    def act(self, user_action):

        self.check_keys()

        self.update_control(user_action)

        self.update_wave()

        if self.selected.any():
            self.n_frame_since_selected += 1
            if self.n_frame_since_selected == self.n_frame_selected:
                self.reset()
        else:
            self.update_deriv()
        self.draw()

        return self.pos


def main():

    hide_cursor = True

    user_control = True

    user_goal = 0
    user_sigma = 0.3
    user_alpha = 0.5

    colors = ("orange", ) + ("blue", ) * 20

    n_target = len(colors)

    # print("user_sigma", user_sigma)

    model = JohnModel(colors=colors, hide_cursor=hide_cursor)
    model.reset()
    user = User(n_target=n_target, sigma=user_sigma, goal=user_goal, alpha=user_alpha)
    user.reset()

    user_action = np.zeros(2)

    if not user_control:
        print("DEBUG MODE: NO USER CONTROL")

    while True:

        model_action = model.act(user_action=user_action)
        if user_control:
            user_action = user.act(model_action)
            user_action *= 2
            # print('user action', user_action)
        else:
            user_action = np.zeros(2)
            # print("NO USER CONTROL")
            # user_action = np.random.random(size=2)


if __name__ == "__main__":
    main()
