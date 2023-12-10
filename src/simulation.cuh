//
// Created by goader on 11/5/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH
#define DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH

#include "simulation_config.cuh"
#include "random_engine.cuh"
#include "particle.cuh"
#include <vector>


class Simulation {
    const SimulationConfig config;
    RandomEngine rng;
    bool h_allFrozen = false;
    bool* d_allFrozen;

    const unsigned int BLOCK_SIZE_1D = 1024;
    const unsigned int BLOCK_SIZE_2D = 32;
    unsigned int numBlocks1d;
    unsigned int numBlocks2d;

    Particle *h_particles;
    Particle *d_particles;
    curandState *d_states;

    public:
        explicit Simulation(const SimulationConfig& config);
        ~Simulation();
        void initParticles();
        void setupCuda();
        void step();
        std::vector<Particle> getParticles();
        [[nodiscard]] bool isFinished() const;
};

#endif //DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH
