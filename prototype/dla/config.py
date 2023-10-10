from dataclasses import dataclass


@dataclass
class DLAConfig:
    width: int = 300
    height: int = 300
    stickiness: float = 0.5
    move_radius: float = 2.0
    particle_radius: float = 4.0
    n_particles: int = 200
    respawn_particles: bool = True
    max_particles: int = 300
    seed: int = None
