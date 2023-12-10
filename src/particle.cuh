//
// Created by goader on 11/30/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_PARTICLE_CUH
#define DIFFUSION_LIMITED_AGGREGATION_PARTICLE_CUH

#include "simulation_config.cuh"
#include "random_engine.cuh"
#include <cmath>
#include <curand_kernel.h>

struct Particle {
    float x, y;
    bool isActive;
};


__global__ void setupRandomStatesKernel(curandState* states, unsigned long seed);

__device__ void randomMove(float moveRadius, float* dx, float* dy, curandState* state);

__global__ void moveParticlesKernel(Particle* particles,
                                    int numParticles,
//                                    const SimulationConfig& config,
                                    curandState* states);

__global__ void checkCollisionsKernel(Particle* particles,
                                      int numParticles,
//                                      const SimulationConfig& config,
                                      bool* allFrozen);

#endif //DIFFUSION_LIMITED_AGGREGATION_PARTICLE_CUH
