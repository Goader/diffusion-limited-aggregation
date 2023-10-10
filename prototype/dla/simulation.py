from .config import DLAConfig
from .particle import Particle
from .random_generator import RandomGenerator
from .tree import Tree


class Simulation:
    def __init__(self, config: DLAConfig):
        self.config = config

        self.generator = RandomGenerator(config)

        init_particle = Particle(config.width / 2, config.height / 2, config.particle_radius)
        self.tree = Tree([init_particle])
        self.particles = []

    def setup(self):
        for _ in range(self.config.n_particles):
            particle = self.generator.particle_on_border()
            self.particles.append(particle)

    def step(self):
        for particle in self.particles:
            dx, dy = self.generator.move()
            particle.move(dx, dy)
            particle.constrain(self.config.width, self.config.height)

            if self.tree.is_colliding(particle):
                if self.generator.should_stick():
                    self.tree.stick(particle)
                    self.particles.remove(particle)

                    if self.config.respawn_particles \
                            and len(self.particles) + len(self.tree) < self.config.max_particles:
                        self.particles.append(self.generator.particle_on_border())

    def is_finished(self) -> bool:
        return len(self.particles) == 0
