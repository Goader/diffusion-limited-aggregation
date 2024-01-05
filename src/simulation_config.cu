//
// Created by goader on 11/5/23.
//

#include "simulation_config.cuh"


__host__ __device__ SimulationConfig::SimulationConfig(const int width,
                                                       const int height,
                                                       const float stickiness,
                                                       const float moveRadius,
                                                       const float particleRadius,
                                                       const int numParticles,
                                                       const int seed)
        : width(width),
          height(height),
          stickiness(stickiness),
          moveRadius(moveRadius),
          particleRadius(particleRadius),
          numParticles(numParticles),
          seed(seed) {}

__host__ __device__ SimulationConfig::SimulationConfig(const SimulationConfig &config) = default;
