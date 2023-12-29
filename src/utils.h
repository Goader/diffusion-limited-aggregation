//
// Created by goader on 11/5/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_UTILS_H
#define DIFFUSION_LIMITED_AGGREGATION_UTILS_H

#include "simulation_config.cuh"
#include "particle.cuh"
#include <string>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

SimulationConfig parseConfig(const std::string& filename);

std::vector<Particle> parseInitialParticles(const std::string& filename);


#endif //DIFFUSION_LIMITED_AGGREGATION_UTILS_H
