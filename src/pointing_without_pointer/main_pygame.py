import pygame
import sys
import numpy as np
from scipy.special import expit
from scipy.signal import butter, sosfiltfilt


class Window:

    def __init__(self, size=(800, 600), fps=30, background="white"):

        pygame.init()
        self.size = size
        self.surface = pygame.display.set_mode(size, 0, 32)
        pygame.display.set_caption('animation')
        self.fps = fps  # frames per second setting
        self.fps_clock = pygame.time.Clock()

        self.background = pygame.Color(background)

        pygame.mouse.set_visible(False)

    def clear(self):

        self.surface.fill(self.background)

    def update(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        self.keep_cursor_inside_window()
        pygame.display.update()
        self.fps_clock.tick(self.fps)

    @staticmethod
    def keep_cursor_inside_window():
        x, y = pygame.mouse.get_pos()
        if not pygame.mouse.get_focused():
            pygame.event.set_grab(True)
            pygame.mouse.set_pos(x, y)

    @property
    def mouse_position(self):

        x_scaled, y_scaled = pygame.mouse.get_pos()
        x_max, y_max = self.surface.get_size()
        x = x_scaled/x_max
        y = y_scaled/y_max

        return x, y


class Visual:

    def __init__(self, window, position):
        self.window = window
        self.position = position

    @property
    def coordinates(self):

        x, y = self.position
        x_max, y_max = self.window.surface.get_size()
        print(x_max, y_max)
        x_scaled = x*x_max
        y_scaled = y*y_max

        return x_scaled, y_scaled


class TextLine(Visual):

    def __init__(self, window,
                 text="Hello world!",
                 position=(0.5, 0.5),
                 font='meslolgmforpowerline',
                 fontsize=32,
                 color="black",
                 background=None,
                 antialiasing=True):

        super().__init__(window=window, position=position)
        self.fontsize = fontsize
        self.font_obj = pygame.font.SysFont(f'{font}.ttf', fontsize)
        self.text = text
        self.color = color
        self.background = background
        self.antialiasing = antialiasing

        self.window.surface.fill(pygame.Color("white"))

    def update(self, text):

        self.text = text
        self.draw()

    def draw(self):

        text_list = self.text.split("\n")

        coord = self.coordinates
        for i, text in enumerate(text_list):

            text_surface_obj = self.font_obj.render(text,
                                                    self.antialiasing,
                                                    pygame.Color(self.color),
                                                    self.background)
            text_rect_obj = text_surface_obj.get_rect()
            text_rect_obj.center = coord[0], coord[1] + i*self.fontsize

            self.window.surface.blit(text_surface_obj, text_rect_obj)


class Circle(Visual):

    def __init__(self, window, position=(0.5, 0.5), color="black", radius=10):

        super().__init__(window=window, position=position)
        self.color = color
        self.radius = radius

    def draw(self):
        pygame.draw.circle(self.window.surface,
                           color=pygame.Color(self.color),
                           center=self.coordinates, radius=self.radius,
                           width=0)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.draw()

def low_pass_filtered_white_noise(rng,
                                  n_sample_noise,
                                  sd_noise,
                                  cutoff, fs, order, ):

    def butter_lowpass_filter(data):
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        flt = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        y = sosfiltfilt(flt, data)
        return y

    smp = rng.normal(scale=sd_noise, size=(2, n_sample_noise))
    x_noise = butter_lowpass_filter(smp[0])
    y_noise = butter_lowpass_filter(smp[1])
    return x_noise, y_noise


def main():

    init_x, init_y = 0.5, 0.5

    base_radius = 5
    var_radius = 40

    # max_radius = base_radius + var_radius

    alpha, beta = 0.3, 0.1

    v = 0.8  # Threshold

    time_window_sec = 0.5
    samples_noise_sec = 6.0

    sd_noise = 0.01

    mvt_size_noise = 1.0
    mvt_size_mouse = 0.8

    cutoff = 0.5  # desired cutoff frequency of the filter, Hz
    order = 3 # sin wave can be approx represented as quadratic

    window = Window()

    time_window = int(time_window_sec * window.fps)
    n_sample_noise = int(samples_noise_sec * window.fps)

    rng = np.random.default_rng()

    x_noise, y_noise = None, None

    gamma = 0

    hist = {k: np.random.uniform(0, 1, size=time_window) for k in ('x', 'y', 'fx', 'fy')}

    circle = Circle(window)
    text = TextLine(window)
    label_position_mouse = TextLine(window, position=(0.5, 0.4))

    fx, fy = init_x, init_y

    i = 0

    while True:

        window.clear()

        if i == 0:

            x_noise, y_noise = low_pass_filtered_white_noise(
                rng=rng,
                n_sample_noise=n_sample_noise,
                cutoff=cutoff,
                fs=window.fps,
                order=order,
                sd_noise=sd_noise)

            # x_noise = np.ones(n_sample_noise)
            # x_noise[:] = 0.01  # 0.0001
            # x_noise[int(len(x_noise) / 2):] = - 0.01  # -0.0001
            # y_noise = np.zeros(n_sample_noise)

        mouse_x, mouse_y = window.mouse_position

        fx = fx + x_noise[i] * mvt_size_noise
        fy = fy + y_noise[i] * mvt_size_noise

        x = fx + (mouse_x - 0.5) * mvt_size_mouse
        y = fy + (mouse_y - 0.5) * mvt_size_mouse

        x = min(max(x, 0), 1)
        y = min(max(y, 0), 1)

        values = {
            'x': x,
            'y': y,
            'fx': fx,
            'fy': fy
        }

        for k, val in hist.items():
            hist[k] = np.roll(val, -1)
            hist[k][-1] = values[k]

        mean = {k: hist[k].mean() for k in hist.keys()}
        rms_pos = np.sqrt(((x - mean['x']) ** 2 + (y - mean['y']) ** 2).mean())
        rms_f = np.sqrt(((fx - mean['fx']) ** 2 + (fy - mean['fy']) ** 2).mean())

        r = rms_pos / rms_f

        need_increase = r < v
        if need_increase:
            gamma += alpha * r
        else:
            gamma -= beta * r

        min_gamma = 0
        max_gamma = 5.5
        gamma = min(max_gamma, gamma)
        gamma = max(min_gamma, gamma)
        p = expit(gamma)

        radius = base_radius + var_radius * (p - expit(min_gamma))

        if p > 0.99:
            color = "red"
        else:
            color = "chartreuse1"

        circle.update(position=(x, y), color=color, radius=radius)
        text.update(f"r={r:.2f}\ngamma={gamma}\np={p:.2f}")

        label_position_mouse.update(f"mouse position = {mouse_x, mouse_y}")

        window.update()

        i += 1
        i %= n_sample_noise


if __name__ == "__main__":
    main()
