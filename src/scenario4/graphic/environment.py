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

        self.fish_icon = Image(
            window=self,
            path="materials/fish.png",
            size=100)

        # First rectangle left x, width; second rectangle width (starts at 0)
        self.pos = None

    def reset(self, fish_init_position, init_shift):
        self.fish_icon.update(position=fish_init_position)
        self.pos = self.update_target_positions(shift=init_shift)

    @property
    def area_width(self):
        return self.size(0) / self.n_target

    @property
    def area_height(self):
        return self.size(1)

    def implement_assistant_action(self, assistant_action):
        if assistant_action is None:
            return
        self.pos = self.update_target_positions(shift=assistant_action)

    def update_target_positions(self, shift):

        screen_width = self.size(0)
        x_shift = shift*screen_width
        area_width, area_height = self.area_width, self.area_height
        pos = np.zeros((self.n_target, 3))
        for i in range(self.n_target):
            x = area_width*i + x_shift
            if x >= screen_width:
                x -= screen_width
            exceed = x + area_width - screen_width
            first_width = area_width - max(0, exceed)
            second_width = area_width - first_width if exceed > 0 else 0
            pos[i] = x, first_width, second_width
        return pos

    def implement_user_action_csq(self, new_fish_position):
        if new_fish_position is None:
            return
        self.fish_icon.update(
            position=new_fish_position)

    def draw(self, aim=None):
        height = self.area_height
        for i in range(self.n_target):
            x, first_width, second_width = self.pos[i]
            Rectangle(
                window=self,
                position=(x.item(), 0),
                color=self.colors[i],
                size=(first_width.item(), height)).draw()
            if second_width:
                Rectangle(
                    window=self,
                    position=(0, 0),
                    color=self.colors[i],
                    size=(second_width.item(), height)).draw()

        self.fish_icon.draw()

        if aim is not None:
            Circle(window=self, position=aim.item()).draw()

    def update(self, new_fish_position, assistant_action, *args, **kwargs):

        self.implement_user_action_csq(new_fish_position)
        self.implement_assistant_action(assistant_action)
        self.graphic_update(*args, **kwargs)

    @property
    def target_positions(self):
        return self.pos
