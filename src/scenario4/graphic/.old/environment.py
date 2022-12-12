import torch

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
        self.pos = torch.zeros((self.n_target, 3))

    def reset(self, fish_init_position, init_shift):

        self.fish.update(position=fish_init_position.numpy())
        self.update_target_positions(shift=init_shift)

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
        pos = torch.zeros((self.n_target, 3))
        for i in range(self.n_target):
            x = area_width*i + x_shift
            if x >= screen_width:
                x -= screen_width
            exceed = x + area_width - screen_width
            first_width = area_width - max(0, exceed)
            second_width = area_width - first_width if exceed > 0 else 0
            pos[i, 0] = x
            pos[i, 1] = first_width
            pos[i, 2] = second_width
        return pos

    def implement_user_action(self, user_action):
        """
        "User action" is the fish jump
        """
        if user_action is None:
            return
        self.fish.update(position=self.update_fish_position(fish_jump=user_action, fish_position=self.fish_position)
                         .detach().numpy())

    @staticmethod
    def update_fish_position(fish_jump, fish_position):
        return fish_position + fish_jump

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

        self.fish.draw()

        if aim is not None:
            Circle(window=self, position=aim.item()).draw()

    def update(self, user_action, assistant_action, *args, **kwargs):

        self.implement_user_action(user_action)
        self.implement_assistant_action(assistant_action)
        self.graphic_update(*args, **kwargs)

    @property
    def target_positions(self):
        return self.pos

    @property
    def fish_position(self):
        return torch.from_numpy(self.fish.position)

    @staticmethod
    def fish_is_in(target, target_positions, fish_position):

        x = target_positions[target, 0]
        print("x", x)
        first_width = target_positions[target, 1]
        second_width = target_positions[target, 2]

        fish_x = fish_position[0]
        fish_is_in = x <= fish_x <= x+first_width or (second_width and 0 <= fish_x <= second_width)
        if x.requires_grad:
            return fish_is_in
        else:
            return fish_is_in.item()

    @staticmethod
    def fish_aim(target, target_positions, fish_position):

        if Environment.fish_is_in(target=target, target_positions=target_positions, fish_position=fish_position):
            return fish_position

        x = target_positions[target, 0]
        first_width = target_positions[target, 1]
        second_width = target_positions[target, 2]
        x_fish = fish_position[0]
        y_fish = fish_position[1]

        first_diff = (x - x_fish)**2

        if second_width:
            second_diff = (second_width - x_fish)**2
            if first_diff >= second_diff:
                x_center = second_width
            else:
                x_center = x
        else:
            second_diff = ((x+first_width) - x_fish)**2
            if first_diff >= second_diff:
                x_center = x+first_width
            else:
                x_center = x

        y_center = y_fish
        return x_center, y_center
