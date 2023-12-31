#ifndef DIFFUSION_LIMITED_AGGREGATION_VISUALIZATION_H
#define DIFFUSION_LIMITED_AGGREGATION_VISUALIZATION_H

#include <vector>
#include "particle.cuh"
#include "simulation_config.cuh"


struct GLParticle {
    float x, y;
};


void visualizeSimulation(const std::vector<Particle>& particles, const SimulationConfig& config, int lastStep);

#endif //DIFFUSION_LIMITED_AGGREGATION_VISUALIZATION_H
