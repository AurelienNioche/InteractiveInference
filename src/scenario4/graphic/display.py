import numpy as np

from .components.window import Window
from .components.element import Image, Rectangle


class Display(Window):

    def __init__(self,
                 colors,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.colors = colors
        self.n_target = len(colors)

        self.fish = Image(window=self,
                          path="materials/fish.png",
                          size=100)

        self.pos = np.zeros(self.n_target)  # Left-aligned
        for i in range(self.n_target):
            self.pos[i] = self.area_width()*i

    def reset(self):
        pass

    def area_width(self):
        return 0.5 * self.size(0)

    def area_height(self):
        return self.size(1)

    def implement_assistant_action(self, assistant_action):
        x_shift = assistant_action[0] * self.size(0)

        self.pos = np.zeros(self.n_target)  # Left-aligned
        for i in range(self.n_target):
            self.pos[i] = self.area_width()*i + x_shift

    def implement_user_action(self, user_action):
        self.fish.update(position=user_action)

    def draw(self):
        width, height = self.area_width(), self.area_height()
        for i in range(self.n_target):
            x = self.pos[i]
            if x >= self.size(0):
                x -= self.size(0)
            exceed = x + width - self.size(0)
            first_width = width - max(0, exceed)
            Rectangle(
                window=self,
                position=(x, 0),
                color=self.colors[i],
                size=(first_width, height)).draw()

            if exceed > 0:
                second_width = width - first_width
                Rectangle(
                    window=self,
                    position=(0, 0),
                    color=self.colors[i],
                    size=(second_width, height)).draw()

        self.fish.draw()

    def update(self, user_action, assistant_action):

        self.implement_user_action(user_action)
        self.implement_assistant_action(assistant_action)
        self.graphic_update()

    @property
    def target_positions(self):
        return self.pos

    def fish_is_in(self, target, target_positions=None, fish_position=None):

        if fish_position is None:
            fish_position = self.fish.position

        if target_positions is None:
            target_positions = self.pos

        x = target_positions[target]
        fish_x, fish_y = fish_position
        if x >= self.size(0):
            x -= self.size(0)
        exceed = x + self.area_width() - self.size(0)
        first_width = self.area_width - max(0, exceed)
        if x <= fish_x <= x+first_width:
            return True
        if exceed > 0:
            second_width = self.area_width() - first_width
            if 0 <= fish_x <= second_width:
                return True
        return False


    def target_center(self, target):
        x0, x1 = self.pos[target]
        x_center, y_center = (x0 + x1) / 2, self.size(1) / 2
        return x_center, y_center
