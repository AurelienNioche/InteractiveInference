import numpy as np
from scipy.special import softmax

from graphic.window import Window
from graphic.shape import Line, Circle

np.seterr(all='raise')


def distance(pos1, pos2):
    return np.sqrt(((pos1 - pos2) ** 2).sum())


class Model:

    def __init__(self):

        self.fps = 30

        self.n_target = 2

        # --- Model parameters ---
        self.add_coeff = 5.0  # `ADD_COEFF`
        self.decay_coeff = 1.15  # `DECAY_COEFF`
        self.decay_threshold = -0.9  # `DECAY_THRESH`
        self.add_threshold = -2.0  # `ADD_THRESH`

        self.n_sin = 4  # `N_SIN`
        self.freq_min = 1
        self.freq_max = 51
        self.phase_min = 0
        self.phase_max = 2*np.pi

        self.phase_inc = 0.003    # `phase_inc`
        self.disturbance_scale = 655.5
        self.lag_tau = 0.08

        # self.time_window_sec = XXX
        # self.time_window = int(self.time_window_sec * self.window.fps)
        self.n_frame_var = 20    # `VAR_WIN`

        self.n_frame_selected = 20

        # self.var_ratio_min = 0.0
        # self.var_ratio_max = 6.0

        self.seed = 123

        self.scale_noise = 0.03

        self.mouse_scale = 0.5

        self.init_frames = 5

        # --- Graphic parameters ---

        self.hide_cursor = False

        # self.init_positions = np.array([(0.5, 0.5), ])  # np.array([(0.3, 0.3), (0.7, 0.7)])

        self.line_scale = 4.0

        self.base_radius = 5
        self.var_radius = 50

        self.width_circle_line = 2

        self.max_radius = self.base_radius + self.var_radius

        self.color_still = "chartreuse1"
        self.color_selected = "red"

        # ---------------------------------------- #

        self.var_ratio = np.zeros(self.n_target)

        self.window = Window(fps=self.fps, hide_cursor=self.hide_cursor)

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

        self.selected = False
        self.n_frame_since_selected = 0
        self.disturbance_phase = 0  # Will be incremented
        self.control = np.zeros(2)
        self.pos = np.zeros((self.n_target, 2))

        self.margin = 0.05

        self.mouse_pos_prev_frame = np.zeros(2)

        self.initialize()
        while True:
            self.loop()

    def initialize(self):

        # To be sure that the mouse can be captured correctly...
        for i in range(self.init_frames):
            self.window.clear()
            self.window.update()

        self.window.move_back_cursor_to_the_middle()

        # self.old_mouse_pos[:] = self.window.mouse_position - 0.5
        # self.old_noise[:] = 0

        self.color[:] = self.color_still
        self.radius[:] = self.base_radius

        self.control[:] = 0
        self.mouse_pos_prev_frame[:] = self.window.mouse_position

        for i in range(self.n_target):
            self.pos[i] = 0.5, 0.5
            for coord in range(2):
                self.hist_pos[i, coord, :] = self.pos[i, coord]
                self.hist_model[i, coord, :] = self.pos[i, coord]

    def loop(self):

        self.update_objs()
        self.draw_trans_obj()

    def update_objs(self):

        mouse_pos = self.window.mouse_position

        delta_mouse = mouse_pos - self.mouse_pos_prev_frame

        self.control += delta_mouse * 50  # self.mouse_scale

        self.hist_control = np.roll(self.hist_control, -1)
        self.hist_control[:, -1] = self.control

        self.disturbance_phase += self.phase_inc

        self.update_wave()
        self.update_deriv()

        self.normalize()

        self.mouse_pos_prev_frame = mouse_pos

    def draw_trans_obj(self):

        p_val = np.zeros(self.n_target)
        selected = np.zeros(self.n_target, dtype=bool)

        for i in range(self.n_target):
            p_val[i] = self.var_ratio[i] / 1000.0
            selected[i] = p_val[i] >= 95.5

        visual_pos = np.zeros_like(self.pos)
        for coord in range(2):
            visual_pos[:, coord] = self.pos[:, coord] + self.control[coord]

        # Convert coordinates from (-100, 100) to (0, 1)
        visual_pos[:] = np.clip(visual_pos, -100, 100)
        visual_pos[:] = 0.5 + visual_pos / (2*100)

        color = np.zeros(self.n_target, dtype=object)
        color[:] = self.color_still
        color[selected] = self.color_selected

        radius = self.base_radius + (self.max_radius - self.base_radius) * (self.var_ratio / 1e5)

        self.window.clear()

        for i in range(self.n_target):
            Circle(window=self.window,
                   position=visual_pos[i],
                   color=color[i],
                   radius=radius[i]).draw()

        self.window.update()

    def update_deriv(self):

        self.hist_pos = np.roll(self.hist_pos, -1)
        self.hist_pos[:, :, -1] = self.pos

        self.hist_model = np.roll(self.hist_model, -1)

        for i in range(self.n_target):
            for coord in range(2):
                pos = self.pos[i, coord]
                hm = self.hist_model[i, coord, -1]
                new_val = (1 - self.lag_tau) * hm + self.lag_tau * pos
                self.hist_model[i, coord, -1] = new_val

        # c = self.pos + self.control

        e_var = np.zeros(self.n_target)
        for i in range(self.n_target):
            h = self.hist_pos[i]
            m = np.mean(h, axis=-1)
            var = np.zeros(2)
            for coord in range(2):
                diff = h[coord] - m[coord]
                var[coord] = np.sum(diff**2, axis=-1)
            e_var[i] = distance(np.zeros(2), var)

        # m = np.zeros((self.n_target, 2))
        # for i in range(self.n_target):
        #     hc, hm = self.hist_control[i], self.hist_model[i]
        #     m = np.mean(hc + hm)

        a_var = np.zeros(self.n_target)
        for i in range(self.n_target):
            hc, hm = self.hist_control[i], self.hist_model[i]
            add = hc + hm
            var = np.sum(add**2)
            a_var[i] = distance(np.zeros(2), var)

        for i in range(self.n_target):
            var = 0
            for j in range(1, self.n_frame_var):
                c_now = self.hist_control[i, j]
                c_prev_frame = self.hist_control[i, j - 1]
                diff = c_now - c_prev_frame
                var += diff**2

            cvar = distance(np.zeros(2), var)

            var_rat = 1.0 - np.sqrt((e_var[i]+22000) / a_var[i]+1e-5)

            if cvar > 2000:
                cvar = 2000

            var_rat += cvar/1000

            if var_rat < self.add_threshold:
                self.var_ratio[i] += var_rat * self.add_coeff
            elif var_rat > self.decay_threshold:
                self.var_ratio[i] /= self.decay_coeff

        # np.clip(self.var_ratio, self.var_ratio_min, self.var_ratio_max)

    def update_wave(self):

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

    def normalize(self):

        norm_ratio = np.sum(self.var_ratio + 0.4)
        self.var_ratio[:] = ((self.var_ratio + 0.4) / norm_ratio) * 10e4

    # def update_graphics(self, pos, delta_mouse, delta_noise):
    #
    #     for i in range(self.n_target):
    #
    #         Circle(window=self.window,
    #                position=position, color=self.color[i],
    #                radius=self.radius[i]).draw()
            # Circle(window=self.window, position=pos[i], color=self.color[i],
            #        radius=self.max_radius,
            #        width=self.width_circle_line).draw()

            # Line(window=self.window,
            #      color="black",
            #      start_position=pos[i],
            #      stop_position=pos[i] + delta_mouse * self.line_scale).draw()
            #
            # Line(window=self.window,
            #      color="red",
            #      start_position=pos[i],
            #      stop_position=pos[i] + delta_noise * self.line_scale).draw()

        # self.window.update()


def main():

    Model()


if __name__ == "__main__":
    main()
