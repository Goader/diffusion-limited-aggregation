//
// Created by goader on 11/5/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH
#define DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH

#include "simulation_config.h"
#include "random_engine.cuh"
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/functional.h>


class Simulation {
    const SimulationConfig config;
    RandomEngine rng;

    thrust::device_vector<float> particlesX;
    thrust::device_vector<float> particlesY;
    thrust::device_vector<bool> particlesActive;

    public:
        explicit Simulation(const SimulationConfig& config);
        void step();
        [[nodiscard]] std::vector<int> getParticlesX() const;
        [[nodiscard]] std::vector<int> getParticlesY() const;
        [[nodiscard]] bool isFinished() const;
};

#endif //DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CUH
