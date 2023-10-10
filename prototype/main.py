from argparse import ArgumentParser

import dla


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--width', type=int, default=300, help='width of the canvas')
    parser.add_argument('--height', type=int, default=300, help='height of the canvas')
    parser.add_argument('--stickiness', type=float, default=0.5, help='probability of sticking to the tree')
    parser.add_argument('--move-radius', type=float, default=2.0, help='radius of the movement')
    parser.add_argument('--particle-radius', type=float, default=4.0, help='radius of the particle')
    parser.add_argument('--n-particles', type=int, default=200, help='number of particles')
    parser.add_argument('--respawn-particles', action='store_true', help='respawn particles')
    parser.add_argument('--max-particles', type=int, default=300, help='maximum number of particles')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    config = dla.DLAConfig(
        width=args.width,
        height=args.height,
        stickiness=args.stickiness,
        move_radius=args.move_radius,
        particle_radius=args.particle_radius,
        n_particles=args.n_particles,
        respawn_particles=args.respawn_particles,
        max_particles=args.max_particles,
        seed=args.seed,
    )
    simulation = dla.Simulation(config)
    simulation.setup()

    iteration = 0
    try:
        while not simulation.is_finished():
            simulation.step()
            iteration += 1

            if iteration % 250 == 0:
                print(f'Iteration: {iteration}, particles: {len(simulation.particles)}, tree: {len(simulation.tree)}')
    except KeyboardInterrupt:
        pass
    finally:
        # plot the tree and particles
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots()
        ax.set_xlim(0, config.width)
        ax.set_ylim(0, config.height)

        patches = []
        for particle in simulation.particles:
            patches.append(Circle((particle.x, particle.y), particle.radius, color='blue'))
        for particle in simulation.tree:
            patches.append(Circle((particle.x, particle.y), particle.radius, color='red'))

        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

        plt.show()
