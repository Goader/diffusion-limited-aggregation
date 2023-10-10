class Particle:
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius

    def move(self, dx: float, dy: float):
        self.x += dx
        self.y += dy

    def constrain(self, width: float, height: float):
        self.x = max(min(self.x, width), 0.)
        self.y = max(min(self.y, height), 0.)
