import numpy as np

from .components.window import Window
from .components.element import Image, Rectangle, Circle


class Display(Window):

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

        # First rectangle left x, width; second rectangle width (starts at 0); height
        self.pos = None

    def draw(self, aim=None):

        if self.pos is None:
            return

        for i in range(self.n_target):
            x, first_width, second_width, height = self.pos[i]
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

    def update(self, fish_position, target_positions):

        self.fish_icon.update(
            position=fish_position)
        self.pos = target_positions

