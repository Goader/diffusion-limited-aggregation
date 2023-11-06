//
// Created by goader on 11/6/23.
//

#ifndef DIFFUSION_LIMITED_AGGREGATION_RANDOM_ENGINE_CUH
#define DIFFUSION_LIMITED_AGGREGATION_RANDOM_ENGINE_CUH

#include "simulation_config.h"
#include <thrust/random.h>

class RandomEngine {
    thrust::default_random_engine engine;
    thrust::uniform_real_distribution<float> particleXDist;
    thrust::uniform_real_distribution<float> particleYDist;
    thrust::uniform_real_distribution<float> angleDist;

    public:
        explicit RandomEngine(const SimulationConfig& config);
        float generateParticleX();
        float generateParticleY();
        float generateAngle();
};


#endif //DIFFUSION_LIMITED_AGGREGATION_RANDOM_ENGINE_CUH
