//
// Created by goader on 11/5/23.
//

#include "simulation.cuh"


Simulation::Simulation(const SimulationConfig& config) : config(config), rng(config) {
    h_particles = new Particle[config.numParticles];

    d_allFrozen = nullptr;
    d_particles = nullptr;
    d_states = nullptr;

    d_forceFieldX = nullptr;
    d_forceFieldY = nullptr;
    d_obstacles = nullptr;

    numBlocks1d = (config.numParticles + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    numBlocks2d = (config.numParticles + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
}

Simulation::~Simulation() {
    delete[] h_particles;
    cudaFree(d_particles);
    cudaFree(d_states);
    cudaFree(d_allFrozen);
    cudaFree(d_forceFieldX);
    cudaFree(d_forceFieldY);
    cudaFree(d_obstacles);
}

void Simulation::initParticles(std::vector<Particle> initialParticles) {
    size_t frozenParticles = initialParticles.size();

    for (int i = 0; i < frozenParticles; i++) {
        h_particles[i].oldX = initialParticles[i].oldX;
        h_particles[i].oldY = initialParticles[i].oldY;
        h_particles[i].x = initialParticles[i].x;
        h_particles[i].y = initialParticles[i].y;
        h_particles[i].isActive = initialParticles[i].isActive;
        h_particles[i].frozenAtStep = initialParticles[i].frozenAtStep;
        h_particles[i].collidedParticleIdx = initialParticles[i].collidedParticleIdx;
    }

    for (size_t i = frozenParticles; i < config.numParticles; i++) {
        auto x = rng.generateParticleX();
        auto y = rng.generateParticleY();
        h_particles[i].oldX = x;
        h_particles[i].oldY = y;
        h_particles[i].x = x;
        h_particles[i].y = y;
        h_particles[i].isActive = true;
        h_particles[i].frozenAtStep = -100;
        h_particles[i].collidedParticleIdx = -1;
    }
}

void Simulation::setupCudaForceField(float* forceFieldX, float* forceFieldY) {
        cudaMalloc(&d_forceFieldX, config.width * config.height * sizeof(float));
        cudaMemcpy(d_forceFieldX, forceFieldX, config.width * config.height * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_forceFieldY, config.width * config.height * sizeof(float));
        cudaMemcpy(d_forceFieldY, forceFieldY, config.width * config.height * sizeof(float), cudaMemcpyHostToDevice);
}


void Simulation::setupCudaObstacles(std::vector<Obstacle> obstacles) {
    size_t numObstacles = obstacles.size();
    
    auto h_obstacles = new Obstacle[numObstacles];
    for (int i = 0; i < numObstacles; i++) {
        h_obstacles[i].xTopLeft = obstacles[i].xTopLeft;
        h_obstacles[i].yTopLeft = obstacles[i].yTopLeft;
        h_obstacles[i].recHeight = obstacles[i].recHeight;
        h_obstacles[i].recWidth = obstacles[i].recWidth;
    }

    cudaError_t err;

    err = cudaMalloc(&d_obstacles, numObstacles * sizeof(Obstacle));
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        delete[] h_obstacles;
        return;
    }
    
    err = cudaMemcpy(d_obstacles, h_obstacles, numObstacles * sizeof(Obstacle), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_obstacles);
        delete[] h_obstacles;
        return;
    }

    delete[] h_obstacles;
}


// must be called after initParticles
void Simulation::setupCuda() {
    cudaMalloc(&d_allFrozen, sizeof(bool));

    cudaMalloc(&d_states, config.numParticles * sizeof(curandState));
    setupRandomStatesKernel<<<numBlocks1d, BLOCK_SIZE_1D>>>(d_states, config.seed);

//    cudaDeviceSynchronize();
//    cudaError_t error = cudaGetLastError();
//    if(error != cudaSuccess)
//    {
//        // print the CUDA error message and exit
//        printf("CUDA error: %s\n", cudaGetErrorString(error));
//        exit(-1);
//    }
//    else {
//        printf("Success!\n");
//    }

    cudaMalloc(&d_particles, config.numParticles * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles,
               config.numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
}

void Simulation::step() {
    moveParticlesKernel<<<numBlocks1d, BLOCK_SIZE_1D>>>(
            d_particles,
            config,
            d_states,
            d_forceFieldX,
            d_forceFieldY,
            d_obstacles
    );

    cudaMemset(d_allFrozen, 1, sizeof(bool));  // set d_allFrozen to true
    dim3 gridDims(numBlocks2d, numBlocks2d); dim3 blockDims(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    checkCollisionsKernel<<<gridDims, blockDims>>>(
            d_particles,
            config
    );

    freezeParticlesKernel<<<numBlocks1d, BLOCK_SIZE_1D>>>(
            d_particles,
            config,
            d_states,
            d_allFrozen,
            current_step
    );
    cudaDeviceSynchronize();  // waiting for the d_allFrozen to be updated
    cudaMemcpy(&h_allFrozen, d_allFrozen, sizeof(bool), cudaMemcpyDeviceToHost);

    current_step++;
}

std::vector<Particle> Simulation::getParticles() {
    cudaDeviceSynchronize();
    cudaMemcpy(h_particles, d_particles,
               config.numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

    // copy the particles to a vector
    std::vector<Particle> particles;
    particles.reserve(config.numParticles);
    for (int i = 0; i < config.numParticles; i++) {
        particles.push_back(h_particles[i]);
    }
    return particles;
}

int Simulation::getCurrentStep() const {
    return current_step;
}

bool Simulation::isFinished() const {
    return h_allFrozen;
}
