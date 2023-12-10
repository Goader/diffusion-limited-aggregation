//
// Created by goader on 11/5/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CONFIG_CUH
#define DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CONFIG_CUH

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


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


        __host__ __device__ SimulationConfig(int width,
                                             int height,
                                             float stickiness,
                                             float moveRadius,
                                             float particleRadius,
                                             int numParticles,
                                             int maxParticles,
                                             bool respawnParticles,
                                             int seed);

        __host__ __device__ SimulationConfig(const SimulationConfig& config);
};


#endif //DIFFUSION_LIMITED_AGGREGATION_SIMULATION_CONFIG_CUH
