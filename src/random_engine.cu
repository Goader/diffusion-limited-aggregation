//
// Created by goader on 11/6/23.
//

#include "random_engine.cuh"

RandomEngine::RandomEngine(const SimulationConfig& config) {
    engine = thrust::default_random_engine(config.seed);
    particleXDist = thrust::uniform_real_distribution<float>(0, (float) config.width);
    particleYDist = thrust::uniform_real_distribution<float>(0, (float) config.height);
    angleDist = thrust::uniform_real_distribution<float>(0, 2 * M_PI);
}

float RandomEngine::generateParticleX() {
    return particleXDist(engine);
}

float RandomEngine::generateParticleY() {
    return particleYDist(engine);
}

float RandomEngine::generateAngle() {
    return angleDist(engine);
}
