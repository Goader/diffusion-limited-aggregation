//
// Created by goader on 11/5/23.
//

#include "simulation.cuh"


Simulation::Simulation(const SimulationConfig& config) : config(config) {
    rng = RandomEngine(config);
    numBlocks = (config.numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

Simulation::~Simulation() {
    cudaFree(thrust::raw_pointer_cast(dev_particlesActive.data()));
}

__global__ void initParticlesKernel(Particle* particles, RandomEngine rng, const SimulationConfig& config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < config.numParticles) {
        new (&particles[idx]) Particle(config, rng);
    }
}

void Simulation::initParticles() {
    Particle* rawPtr;
    cudaMalloc(&rawPtr, config.numParticles * sizeof(Particle));

    initParticlesKernel<<<numBlocks, BLOCK_SIZE>>>(rawPtr, rng, config);

    std::vector<Particle*> host_particles(config.numParticles);
    for (int i = 0; i < config.numParticles; ++i) {
        host_particles[i] = rawPtr + i;
    }
    dev_particlesActive = thrust::device_vector<Particle*>(host_particles.begin(), host_particles.end());
}

__global__ void moveParticlesKernel(Particle* particles, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        particles[idx].move();
    }
}

void Simulation::step() {
    moveParticlesKernel<<<numBlocks, BLOCK_SIZE>>>(dev_particlesActive, config.numParticles);
    cudaDeviceSynchronize();
    // todo: handle state updates needed after particle movement (collisions)
}

bool Simulation::isFinished() const {
//    bool anyActive = thrust::reduce(
//        particlesActive.begin(),
//        particlesActive.end(),
//        false,
//        thrust::logical_or<bool>()
//    );
    return false;
}
