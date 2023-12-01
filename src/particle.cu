//
// Created by goader on 11/30/23.
//

#include "particle.cuh"
#include <cmath>


__host__ __device__ Particle::Particle(const SimulationConfig& config, RandomEngine& rng)
        : config(config), rng(rng) {
    isActive = true;
    x = rng.generateParticleX();
    y = rng.generateParticleY();
}

__host__ __device__ void Particle::move() {
    float angle = rng.generateAngle();
    x += config.moveRadius * std::cos(angle);  // todo: check if fast enough (mykolus has an idea)
    y += config.moveRadius * std::sin(angle);
    clipX();
    clipY();
}

__host__ __device__ void Particle::freeze() {
    isActive = false;
}

__host__ __device__ void Particle::clipX() {
    x = std::max(0.f, std::min(x, static_cast<float>(config.width)));
}

__host__ __device__ void Particle::clipY() {
    x = std::max(0.f, std::min(y, static_cast<float>(config.height)));
}
