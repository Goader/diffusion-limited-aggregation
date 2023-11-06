//
// Created by goader on 11/5/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CONFIG_H
#define DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CONFIG_H


class SimulationConfig {
    public:
        const int width;
        const int height;
        const float stickiness;
        const float moveRadius;
        const float particleRadius;
        const int numParticles;
        const int maxParticles;
        const bool respawnParticles;
        const int seed;


        SimulationConfig(const int width,
                         const int height,
                         const float stickiness,
                         const float moveRadius,
                         const float particleRadius,
                         const int numParticles,
                         const int maxParticles,
                         const bool respawnParticles,
                         const int seed);
};


#endif //DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CONFIG_H
