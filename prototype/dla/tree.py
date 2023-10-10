from .particle import Particle


class Tree:
    def __init__(self, particles: list[Particle]):
        self.particles = particles

        # self.kdtree = cKDTree()  # FIXME aabb tree instea d? select a data structure

    def __len__(self):
        return len(self.particles)

    def __iter__(self):
        return iter(self.particles)

    def stick(self, particle: Particle):
        self.particles.append(particle)

    def is_colliding(self, particle: Particle) -> bool:
        px, py = particle.x, particle.y
        for other in self.particles:
            ox, oy = other.x, other.y
            distance = (px - ox)**2 + (py - oy)**2
            if distance <= (particle.radius + other.radius)**2:
                return True

        return False
