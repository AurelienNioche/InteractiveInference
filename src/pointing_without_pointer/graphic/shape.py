import pygame


class Visual:

    def __init__(self, window, position):
        self.window = window
        self.position = position

    @property
    def coordinates(self):

        x, y = self.position
        x_max, y_max = self.window.surface.get_size()
        x_scaled = x*x_max
        y_scaled = y*y_max

        return x_scaled, y_scaled

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.draw()

    def draw(self):
        pass


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

    def __init__(self, window, position=(0.5, 0.5), color="black", radius=10, width=0):

        super().__init__(window=window, position=position)
        self.color = color
        self.radius = radius
        self.width = width

        self.draw()

    def draw(self):
        pygame.draw.circle(self.window.surface,
                           color=pygame.Color(self.color),
                           center=self.coordinates, radius=self.radius,
                           width=self.width)


class Line:

    def __init__(self, window,
                 start_position=(-0.2, -0.2),
                 stop_position=(+0.2, +0.2),
                 color="black",
                 width=2):

        self.window = window
        self.color = color
        self.start_position = start_position
        self.stop_position = stop_position
        self.width = width

        self.draw()

    def draw(self):
        pygame.draw.line(self.window.surface,
                         pygame.Color(self.color),
                         self.coordinates(self.start_position),
                         self.coordinates(self.stop_position),
                         width=self.width)

    def coordinates(self, position):

        x, y = position
        x_max, y_max = self.window.surface.get_size()
        x_scaled = x*x_max
        y_scaled = y*y_max

        return x_scaled, y_scaled

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.draw()
