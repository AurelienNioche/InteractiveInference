import numpy as np
import pygame
from scipy.special import softmax

from graphic.window import Window
from graphic.shape import Line, Circle

np.seterr(all='raise')


def distance(pos1, pos2):
    return np.sqrt(((pos1 - pos2) ** 2).sum())


class Model:

    def __init__(self):

        self.n_target = 2

        self.fps = 30

        # --- Model parameters ---
        self.add_coeff = 5.0  # `ADD_COEFF`
        self.decay_coeff = 1.15  # `DECAY_COEFF`
        self.decay_threshold = -0.9  # `DECAY_THRESH`
        self.add_threshold = -2.0  # `ADD_THRESH`

        self.selection_threshold = 0.95

        self.n_sin = 4  # `N_SIN`
        self.freq_min = 1
        self.freq_max = 51
        self.phase_min = 0
        self.phase_max = 2*np.pi

        self.phase_inc = 0.003    # `phase_inc`
        self.disturbance_scale = 655.5
        self.disturbance_phase = 0.0  # Will be incremented
        self.lag_tau = 0.08

        # self.time_window_sec = XXX
        # self.time_window = int(self.time_window_sec * self.window.fps)
        self.n_frame_var = 20    # `VAR_WIN`

        self.n_frame_selected = 20

        # self.var_ratio_min = 0.0
        # self.var_ratio_max = 6.0

        self.seed = 123

        self.scale_noise = 0.03

        self.mouse_scale = 50

        self.init_frames = 5

        # --- Graphic parameters ---

        self.window_size = (1400, 1000)

        self.margin = 0.05

        self.hide_cursor = False

        self.line_scale = 4.0

        self.base_radius = 5
        self.var_radius = 50

        self.width_circle_line = 2

        self.max_radius = self.base_radius + self.var_radius

        self.color_still = "chartreuse1"
        self.color_selected = "red"

        # ---------------------------------------- #

        self.var_ratio = np.zeros(self.n_target)

        self.window = Window(size=self.window_size, fps=self.fps, hide_cursor=self.hide_cursor)

        self.rng = np.random.default_rng(seed=self.seed)

        self.hist_pos = np.zeros((self.n_target, 2, self.n_frame_var))
        self.hist_control = np.zeros((2, self.n_frame_var))
        self.hist_model = np.zeros((self.n_target, 2, self.n_frame_var))

        self.freq = self.rng.uniform(size=(self.n_target, self.n_sin), low=self.freq_min, high=self.freq_max)
        self.phase = self.rng.uniform(size=(self.n_target, self.n_sin), low=self.phase_min, high=self.phase_max)

        self.color = np.zeros(self.n_target, dtype=str)
        self.radius = np.zeros(self.n_target)

        self.old_mouse_pos = np.zeros(2)
        self.old_noise = np.zeros((self.n_target, 2))

        self.n_frame_since_selected = 0
        self.control = np.zeros(2)
        self.pos = np.zeros((self.n_target, 2))
        self.selected = np.zeros(self.n_target, dtype=bool)

        self.ign = 0

        self.mouse_pos_prev_frame = np.zeros(2)

        self.initialize()
        self.reset()
        while True:
            self.loop()

    def initialize(self):

        # To be sure that the mouse can be captured correctly...
        for i in range(self.init_frames):
            self.window.clear()
            self.window.update()

        self.window.move_back_cursor_to_the_middle()

        self.pos[:] = 0.5, 0.5

    def reset(self):

        self.control[:] = 0
        self.mouse_pos_prev_frame[:] = self.window.mouse_position

        for i in range(self.n_target):
            for coord in range(2):
                self.hist_pos[i, coord, :] = self.pos[i, coord]
                self.hist_model[i, coord, :] = self.pos[i, coord]

        self.n_frame_since_selected = 0
        # self.disturbance_phase = 0  # Will be incremented

        self.var_ratio[:] = 0
        self.selected[:] = 0

        self.ign = 0

    def loop(self):

        self.update_objs()
        self.draw_trans_obj()

    def update_objs(self):

        mouse_pos = self.window.mouse_position
        delta_mouse = mouse_pos - self.mouse_pos_prev_frame
        for coord in range(2):
            self.control[coord] += delta_mouse[coord] * self.mouse_scale
        self.mouse_pos_prev_frame[:] = mouse_pos

        if self.window.cursor_touch_border():
            self.window.move_back_cursor_to_the_middle()
            #cself.mouse_pos_prev_frame[:] = 0.5

        self.update_history()
        self.update_wave()

        if self.selected.any():
            self.n_frame_since_selected += 1
            if self.n_frame_since_selected == self.n_frame_selected:
                self.reset()
        else:
            self.update_deriv()

    def draw_trans_obj(self):

        visual_pos = np.zeros_like(self.pos)
        for coord in range(2):
            visual_pos[:, coord] = self.pos[:, coord] + self.control[coord]

        # Convert coordinates from (-100, 100) to (0, 1)
        visual_pos[:] = np.clip(visual_pos, -100, 100)
        visual_pos[:] = 0.5 + visual_pos / (2*100)

        for coord in range(2):
            visual_pos[:, coord] = self.margin + (1 - self.margin*2) * visual_pos[:, coord]

        radius = self.base_radius + (self.max_radius - self.base_radius) * self.var_ratio

        color = np.zeros(self.n_target, dtype=object)
        color[:] = self.color_still
        color[self.selected] = self.color_selected

        self.window.clear()

        for i in range(self.n_target):
            Circle(window=self.window,
                   position=visual_pos[i],
                   color=color[i],
                   radius=radius[i]).draw()

            # Line(window=self.window,
            #      color="black",
            #      start_position=pos[i],
            #      stop_position=pos[i] + delta_mouse * self.line_scale).draw()
            #
            # Line(window=self.window,
            #      color="red",
            #      start_position=pos[i],
            #      stop_position=pos[i] + delta_noise * self.line_scale).draw()

        self.window.update()

    def update_history(self):

        self.hist_control = np.roll(self.hist_control, -1)
        self.hist_control[:, -1] = self.control

        self.hist_pos = np.roll(self.hist_pos, -1)
        self.hist_pos[:, :, -1] = self.pos

        self.hist_model = np.roll(self.hist_model, -1)

        for i in range(self.n_target):
            for coord in range(2):
                pos = self.pos[i, coord]
                hm = self.hist_model[i, coord, -1]
                new_val = (1 - self.lag_tau) * hm + self.lag_tau * pos
                self.hist_model[i, coord, -1] = new_val

    def update_deriv(self):

        # For computation
        self.var_ratio *= 1e5

        e_var = np.zeros(self.n_target)
        for i in range(self.n_target):
            var = np.zeros(2)
            for coord in range(2):
                h = self.hist_pos[i, coord]
                m = np.mean(h)
                diff = h - m
                var[coord] = (diff**2).sum()
            e_var[i] = distance(np.zeros(2), var)

        a_var = np.zeros(self.n_target)
        hc = self.hist_control

        for i in range(self.n_target):
            var = np.zeros(2)
            for coord in range(2):
                hm = self.hist_model[i, coord, :]
                add = hc + hm
                var[coord] = (add**2).sum()

            a_var[i] = distance(np.zeros(2), var)

        var = np.zeros(2)
        for coord in range(2):
            for j in range(1, self.n_frame_var):
                c_now = self.hist_control[coord, j]
                c_prev_frame = self.hist_control[coord, j-1]

                diff = c_now - c_prev_frame
                var[coord] += diff**2

        c_var = distance(np.zeros(2), var)

        if c_var > 2000:
            c_var = 2000

        var_rat = 1.0 - np.sqrt((e_var + 22000) / a_var + 1e-05)
        var_rat += c_var/1000

        need_add = var_rat < self.add_threshold
        need_decay = var_rat > self.decay_threshold
        self.var_ratio[need_add] += var_rat[need_add] * self.add_coeff
        self.var_ratio[need_decay] /= self.decay_coeff

        # Normalize
        norm_ratio = np.sum(self.var_ratio + 0.4)
        self.var_ratio[:] = ((self.var_ratio + 0.4) / norm_ratio) * 10e4

        # Convert into probability
        self.var_ratio /= 1e5

        self.selected[:] = self.var_ratio > self.selection_threshold

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
            self.pos[i] += total_d

        self.pos = np.clip(self.pos, -100, 100)


def main():

    Model()


if __name__ == "__main__":
    main()
