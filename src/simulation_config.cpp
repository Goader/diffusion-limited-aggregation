//
// Created by goader on 11/5/23.
//

#include "simulation_config.h"

SimulationConfig::SimulationConfig(const int width,
                                   const int height,
                                   const float stickiness,
                                   const float moveRadius,
                                   const float particleRadius,
                                   const int numParticles,
                                   const int maxParticles,
                                   const bool respawnParticles,
                                   const int seed)
        : width(width),
          height(height),
          stickiness(stickiness),
          moveRadius(moveRadius),
          particleRadius(particleRadius),
          numParticles(numParticles),
          maxParticles(maxParticles),
          respawnParticles(respawnParticles),
          seed(seed) {}
