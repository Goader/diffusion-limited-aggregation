import random
import math

from .config import DLAConfig
from .particle import Particle


class RandomGenerator:
    def __init__(self, config: DLAConfig):
        self.config = config
        random.seed(config.seed)

    def particle(self) -> Particle:
        return Particle(
            random.random() * self.config.width,
            random.random() * self.config.height,
            self.config.particle_radius
        )

    def particle_on_border(self) -> Particle:
        xx = random.random()
        yy = random.choice([0., 1.])

        if random.random() > 0.5:
            xx, yy = yy, xx
        return Particle(xx * self.config.width, yy * self.config.height, self.config.particle_radius)

    def move(self) -> tuple[float, float]:
        angle = random.random() * 2 * math.pi
        # TODO does it need to be optimized?
        return self.config.move_radius * math.cos(angle), self.config.move_radius * math.sin(angle)

    def should_stick(self) -> bool:
        return random.random() < self.config.stickiness
