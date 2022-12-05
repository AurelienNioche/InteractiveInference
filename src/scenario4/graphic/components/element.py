import pygame
import numpy as np


class Visual:

    def __init__(self, window, position):
        self.window = window
        if position is None:
            position = window.center
        self.position = position

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.draw()

    def draw(self):
        pass


class Image(Visual):

    def __init__(self, window, path, position=None, size=None):
        super().__init__(window=window, position=position)
        self.image, self.size = Image.load(path, size)

    @classmethod
    def load(cls, path, size):
        raw = pygame.image.load(path)
        try:
            img = raw.convert_alpha()
        except:
            img = raw.convert()
        if size is None:
            pass
        elif isinstance(size, float) or isinstance(size, int):
            original_size = np.asarray(img.get_size())
            ratio = size / original_size[0]
            img = pygame.transform.scale(img, ratio * original_size)
        else:
            img = pygame.transform.scale(img, size)

        size = np.asarray(img.get_size())
        return img, size

    def draw(self):
        width, height = self.size
        left_top_corner = self.position[0]-width/2, self.position[1]-height/2
        self.window.surface.blit(self.image, left_top_corner)


class Text(Visual):

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

    def draw(self):

        text_list = self.text.split("\n")
        for i, text in enumerate(text_list):

            text_surface_obj = self.font_obj.render(text,
                                                    self.antialiasing,
                                                    pygame.Color(self.color),
                                                    self.background)
            text_rect_obj = text_surface_obj.get_rect()
            text_rect_obj.center = self.position[0], self.position[1] + i*self.fontsize

            self.window.surface.blit(text_surface_obj, text_rect_obj)


class Circle(Visual):

    def __init__(self, window, position=None, color="black", radius=10, width=0):
        super().__init__(window=window, position=position)
        self.color = color
        self.radius = radius
        self.width = width

    def draw(self):
        pygame.draw.circle(surface=self.window.surface,
                           color=pygame.Color(self.color),
                           center=self.position, radius=self.radius,
                           width=self.width)


class Line:

    def __init__(self, window,
                 start_position,
                 stop_position,
                 color="black",
                 width=2):

        self.window = window
        self.color = color
        self.start_position = start_position
        self.stop_position = stop_position
        self.width = width

        self.draw()

    def draw(self):
        pygame.draw.line(surface=self.window.surface,
                         color=pygame.Color(self.color),
                         start_pos=self.start_position,
                         end_pos=self.stop_position,
                         width=self.width)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.draw()


class Rectangle(Visual):

    def __init__(self, window, position=None,
                 color="black",
                 size=(100, 100)):
        super().__init__(window=window, position=position)
        self.color = color
        self.size = np.asarray(size)

    def draw(self):
        top_left_corner = self.position
        pygame.draw.rect(surface=self.window.surface,
                         color=pygame.Color(self.color),
                         rect=(*top_left_corner, *self.size))
