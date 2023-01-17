import numpy as np


class Environment:

    def __init__(self, n_target):
        self.n_target = n_target

    def update_target_positions(self, shift, screen_size):

        screen_width, screen_height = screen_size
        x_shift = shift * screen_width
        area_width, area_height = screen_width/self.n_target, screen_height
        pos = np.zeros((self.n_target, 4))
        for i in range(self.n_target):
            x = area_width * i + x_shift
            if x >= screen_width:
                x -= screen_width
            exceed = x + area_width - screen_width
            first_width = area_width - max(0, exceed)
            second_width = area_width - first_width if exceed > 0 else 0
            pos[i] = x, first_width, second_width, screen_height
        return pos
