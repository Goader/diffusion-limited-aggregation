//
// Created by goader on 11/5/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH
#define DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH

#include "simulation_config.h"
#include "random_engine.cuh"
#include "particle.cuh"
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/functional.h>


class Simulation {
    const SimulationConfig config;
    RandomEngine rng;
    thrust::device_vector<Particle*> dev_particlesActive;
    thrust::device_vector<Particle*> dev_particlesFrozen;

    const unsigned int BLOCK_SIZE = 256;
    unsigned int numBlocks;

    public:
        explicit Simulation(const SimulationConfig& config);
        ~Simulation();
        void initParticles();
        void step();
        [[nodiscard]] bool isFinished() const;
};

#endif //DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH
