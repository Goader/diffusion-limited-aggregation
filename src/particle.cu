//
// Created by goader on 11/30/23.
//

#include "particle.cuh"

__global__ void setupRandomStatesKernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__device__ void randomUniform(float* x, curandState* state) {
    *x = curand_uniform(state);
}

__device__ void randomMove(float moveRadius, float* dx, float* dy, curandState* state) {
    float angle = curand_uniform(state) * 2 * M_PI;
    *dx = moveRadius * cos(angle);
    *dy = moveRadius * sin(angle);
}

__global__ void moveParticlesKernel(Particle* particles,
                                    SimulationConfig config,
                                    curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // guard against out of bounds access and inactive particles
    auto particle = particles + idx;
    if (idx > config.numParticles || !particle->isActive) {
        return;
    }

    // generate random move
    float dx, dy;
    randomMove(config.moveRadius, &dx, &dy, &states[idx]);

    // move the particle
    particle->x += dx;
    particle->y += dy;

    // set the sticky flag for the future collision check (allows to avoid creating NxN random states)
    float u;
    randomUniform(&u, &states[idx]);
    particle->isSticky = u < config.stickiness;

    // clip the coordinates to stay within the bounds of the simulation
    particle->x = fmax(0.f, fmin(particle->x, static_cast<float>(config.width)));
    particle->y = fmax(0.f, fmin(particle->y, static_cast<float>(config.height)));
}

__global__ void checkCollisionsKernel(Particle* particles,
                                      SimulationConfig config,
                                      bool* allFrozen) {
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // guard against out of bounds access and inactive particles
    if (xIdx >= config.numParticles || yIdx >= config.numParticles || xIdx >= yIdx) {
        return;
    }

    auto particleX = particles + xIdx;
    auto particleY = particles + yIdx;

    // allows to skip aggregation check if all particles are frozen
    if (particleX->isActive || particleY->isActive) *allFrozen = false;

    // one of the particles must be active, and the other must be frozen
    if (!(particleX->isActive ^ particleY->isActive)) {
        return;
    }

    auto frozenParticle = !particleX->isActive ? particleX : particleY;
    auto activeParticle = particleX->isActive ? particleX : particleY;

    // if the active particle is not sticky, it cannot freeze
    if (!activeParticle->isSticky) {
        return;
    }

    auto squaredDistance = pow(frozenParticle->x - activeParticle->x, 2)
                              + pow(frozenParticle->y - activeParticle->y, 2);
    auto freezeRadiusSquared = pow(config.particleRadius * 2, 2);

    // if the active particle is within the freeze radius of the frozen particle, freeze it
    if (squaredDistance > freezeRadiusSquared) {
        return;
    }

    activeParticle->isActive = false;

    // adjust the particle's position to be on the surface of the frozen particle
    // todo: implement this
    // todo: do we need to have the previous position of the particle?
    // fixme: that would mean we could be changing the xy at the same time
    // fixme: as the other threads may do that too
}
