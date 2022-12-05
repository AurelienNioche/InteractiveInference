import numpy as np

from .components.window import Window
from .components.element import Image, Rectangle, Circle


class Environment(Window):

    def __init__(self,
                 colors,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.colors = colors
        self.n_target = len(colors)

        self.fish = Image(window=self,
                          path="materials/fish.png",
                          size=100)

        # First rectangle left x, width; second rectangle width (starts at 0)
        self.pos = np.zeros((self.n_target, 3))

        self.x_shift = 0

    def reset(self, fish_init_position):

        self.fish.update(position=fish_init_position)
        self.x_shift = 0
        self.update_target_positions()

    @property
    def area_width(self):
        return self.size(0) / self.n_target

    @property
    def area_height(self):
        return self.size(1)

    def implement_assistant_action(self, assistant_action):
        if assistant_action is None:
            return

        self.x_shift = assistant_action[0] * self.size(0)
        self.update_target_positions()

    def update_target_positions(self):

        width, height = self.area_width, self.area_height
        self.pos[:] = 0
        for i in range(self.n_target):
            x = self.area_width*i + self.x_shift
            if x >= self.size(0):
                x -= self.size(0)
            exceed = x + width - self.size(0)
            first_width = width - max(0, exceed)
            second_width = width - first_width if exceed > 0 else None
            self.pos[i] = x, first_width, second_width

    def implement_user_action(self, user_action):
        """
        "User action" is the fish jump
        """
        if user_action is None:
            return
        self.fish.update(position=self.fish.position + user_action)

    def draw(self, aim=None):
        height = self.area_height
        for i in range(self.n_target):
            x, first_width, second_width = self.pos[i]
            Rectangle(
                window=self,
                position=(x, 0),
                color=self.colors[i],
                size=(first_width, height)).draw()
            if not np.isnan(second_width):
                Rectangle(
                    window=self,
                    position=(0, 0),
                    color=self.colors[i],
                    size=(second_width, height)).draw()

        self.fish.draw()

        if aim is not None:
            Circle(window=self, position=aim).draw()

    def update(self, user_action, assistant_action, *args, **kwargs):

        self.implement_user_action(user_action)
        self.implement_assistant_action(assistant_action)
        self.graphic_update(*args, **kwargs)

    @property
    def target_positions(self):
        return self.pos

    def fish_is_in(self, target=0, target_positions=None, fish_position=None):

        if fish_position is None:
            fish_position = self.fish.position

        if target_positions is None:
            target_positions = self.pos

        x, first_width, second_width = target_positions[target]
        fish_x, fish_y = fish_position
        if x <= fish_x <= x+first_width:
            return True
        elif not np.isnan(second_width) and 0 <= fish_x <= second_width:
            return True
        else:
            return False

    def fish_aim(self, target=0, target_positions=None, fish_position=None):

        if fish_position is None:
            fish_position = self.fish.position

        if target_positions is None:
            target_positions = self.pos

        if self.fish_is_in(target=target, target_positions=target_positions, fish_position=fish_position):
            return fish_position

        x, first_width, second_width = target_positions[target]
        x_fish, y_fish = fish_position
        x_to_look = np.zeros(2)
        x_to_look[0] = x

        if np.isnan(second_width):
            x_to_look[1] = x+first_width
        else:
            x_to_look[1] = second_width

        diff = (x_to_look - x_fish)**2

        x_center = x_to_look[diff.argmin()]
        y_center = y_fish
        return x_center, y_center
