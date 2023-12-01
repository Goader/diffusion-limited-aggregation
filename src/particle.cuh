//
// Created by goader on 11/30/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_PARTICLE_CUH
#define DIFFUSION_LIMITED_AGGREGATION_PARTICLE_CUH

#include "simulation_config.h"
#include "random_engine.cuh"


class Particle {
    float x, y;
    bool isActive;

    const SimulationConfig& config;
    RandomEngine& rng;

public:
    __host__ __device__ explicit Particle(const SimulationConfig& config, RandomEngine& rng);
    __host__ __device__ void move();
    __host__ __device__ void freeze();
    __host__ __device__ void clipX();
    __host__ __device__ void clipY();
};


#endif //DIFFUSION_LIMITED_AGGREGATION_PARTICLE_CUH
