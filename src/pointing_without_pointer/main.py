import numpy as np
from graphic.window import Window
from graphic.shape import Circle, Line
from scipy.special import softmax


class SinusoidalNoise:

    def __init__(
            self, fps, rng,
            num_f,
            f_min,
            f_max):

        self.fps = fps
        self.t = 0

        self.f = rng.uniform(size=(2, num_f)) * (f_max - f_min) + f_min
        self.p = rng.uniform(size=(2, num_f)) * 2 * np.pi

    def new(self):

        x = self.oscillate(self.t, self.f[0], self.p[0])
        y = self.oscillate(self.t, self.f[1], self.p[1])

        self.t += 1 / self.fps
        pos = np.asarray([x, y])
        return pos

    @staticmethod
    def oscillate(t, freq, phase):
        return np.sin(2 * np.pi * freq * t + phase).sum()


def main():

    hide_cursor = False

    center_positions = np.array([(0.3, 0.3), (0.7, 0.7)])

    line_scale = 4.0

    base_radius = 5
    var_radius = 50

    max_radius = base_radius + var_radius

    alpha, beta = 0.5, 0.01

    time_stay_selected = 1.0

    min_gamma = 0.0
    max_gamma = 6.0

    v_threshold = 0.9

    time_window_sec = 4.0

    seed = 123

    scale_noise = 0.03
    scale_mouse = 0.4   # Golden value: 0.2?

    color_still = "chartreuse1"
    color_selected = "red"

    num_f = 40
    f_min = 0.05
    f_max = 0.2

    init_frames = 5

    # ---------------------------------------- #

    n_target = len(center_positions)

    gamma = np.zeros(n_target)

    window = Window(hide_cursor=hide_cursor)

    rng = np.random.default_rng(seed=seed)

    time_window = int(time_window_sec * window.fps)
    hist_pos = np.zeros((n_target, 2,  time_window))
    hist_f = np.zeros((n_target, 2,  time_window))

    noise_maker = \
        [SinusoidalNoise(
            fps=window.fps, rng=rng,
            num_f=num_f,
            f_min=f_min,
            f_max=f_max)
         for _ in range(n_target)]

    # To be sure that the mouse can be captured correctly...
    for i in range(init_frames):
        window.clear()
        window.update()

    window.move_back_cursor_to_the_middle()

    old_mouse_pos = window.mouse_position - 0.5

    old_noise = np.zeros((n_target, 2))

    selected = False
    time_since_selected = 0

    color = [color_still for _ in range(n_target)]
    radius = [base_radius for _ in range(n_target)]

    while True:

        window.clear()

        noise = np.zeros((n_target, 2))
        f = np.zeros((n_target, 2))
        pos = np.zeros((n_target, 2))

        mouse_pos = window.mouse_position
        mouse_pos *= scale_mouse

        for i in range(n_target):

            noise[i] = noise_maker[i].new()

        noise[:] *= scale_noise

        f[:] = center_positions[:] + noise[:]

        pos[:] = f[:] + mouse_pos

        pos = np.clip(pos, 0.0, 1.0)

        hist_pos = np.roll(hist_pos, -1)
        hist_f = np.roll(hist_f, -1)
        hist_pos[:, :, -1] = pos
        hist_f[:, :, -1] = f

        for i in range(n_target):

            mean_pos = hist_pos[i].mean(axis=-1)
            mean_f = hist_f[i].mean(axis=-1)

            rms_pos = np.sqrt(((pos - mean_pos) ** 2).sum())
            rms_f = np.sqrt(((f - mean_f) ** 2).sum())

            r = rms_pos / rms_f

            increase = r < v_threshold

            if increase:
                gamma[i] += alpha * (1-r)
            else:
                gamma[i] *= (1-beta)

        np.clip(gamma, min_gamma, max_gamma)

        p = softmax(gamma)

        delta_mouse = mouse_pos - old_mouse_pos

        if not selected:

            radius[:] = base_radius + var_radius * p[:]

            for i in range(n_target):

                selected_i = p[i] > 0.99

                if selected_i:
                    selected = True
                    time_since_selected = 0
                    color[i] = color_selected

        else:
            time_since_selected += 1.0/window.fps
            if time_since_selected >= time_stay_selected:
                selected = False
                color = [color_still for _ in range(n_target)]
                gamma[:] = 0
                hist_f[:] = 0
                hist_pos[:] = 0

        for i in range(n_target):

            Circle(window=window, position=pos[i], color=color[i],
                   radius=radius[i]).draw()
            Circle(window=window, position=pos[i], color=color[i], radius=max_radius,
                   width=2).draw()

            Line(window=window,
                 color="black",
                 start_position=pos[i],
                 stop_position=pos[i] + delta_mouse*line_scale).draw()

            delta_noise = noise[i] - old_noise[i]

            Line(window=window,
                 color="red",
                 start_position=pos[i],
                 stop_position=pos[i] + delta_noise*line_scale).draw()

        window.update()

        old_mouse_pos[:] = mouse_pos
        old_noise[:] = noise


if __name__ == "__main__":
    main()
