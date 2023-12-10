//
// Created by goader on 11/30/23.
//

#include "particle.cuh"

__global__ void setupRandomStatesKernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}


__device__ void randomMove(float moveRadius, float* dx, float* dy, curandState* state) {
    float angle = curand_uniform(state) * 2 * M_PI;
    *dx = moveRadius * cos(angle);
    *dy = moveRadius * sin(angle);
}

__global__ void moveParticlesKernel(Particle* particles,
                                    int numParticles,
//                                    const SimulationConfig& config,
                                    curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // guard against out of bounds access and inactive particles
    auto particle = particles + idx;
    if (idx < numParticles && particle->isActive) {
        // generate random move
        float dx, dy;
        randomMove(2.0, &dx, &dy, &states[idx]);

        // move the particle
        particle->x += dx;
        particle->y += dy;

        // clip the coordinates to stay within the bounds of the simulation
        particle->x = fmax(0.f, fmin(particle->x, static_cast<float>(400)));
        particle->y = fmax(0.f, fmin(particle->y, static_cast<float>(400)));
    }
}


__global__ void checkCollisionsKernel(Particle* particles,
                                      int numParticles,
//                                      const SimulationConfig& config,
                                      bool* allFrozen) {
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // guard against out of bounds access and inactive particles
    if (xIdx < numParticles && yIdx < numParticles && xIdx < yIdx) {
        auto particleX = particles + xIdx;
        auto particleY = particles + yIdx;

        // one of the particles must be active, and the other must be frozen
        if (particleX->isActive ^ particleY->isActive) {
            auto frozenParticle = !particleX->isActive ? particleX : particleY;
            auto activeParticle = particleX->isActive ? particleX : particleY;

            auto squaredDistance = pow(frozenParticle->x - activeParticle->x, 2)
                                   + pow(frozenParticle->y - activeParticle->y, 2);

            // if the active particle is within the freeze radius of the frozen particle, freeze it
            auto freezeRadiusSquared = pow(1.0 * 2, 2);
            if (squaredDistance <= freezeRadiusSquared) {
                activeParticle->isActive = false;

                // adjust the particle's position to be on the surface of the frozen particle
                // todo: implement this
                // todo: do we need to have the previous position of the particle?
            }
        }

        // allows to skip aggregation check if all particles are frozen
        if (particleX->isActive) *allFrozen = false;
        if (particleY->isActive) *allFrozen = false;
    }
}
