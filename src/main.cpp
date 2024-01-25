//
// Created by goader on 11/5/23.
//

#include <filesystem>
#include <iostream>

#include <GL/glew.h>  // include first
#include <GLFW/glfw3.h>


#include "utils.h"
#include "simulation.cuh"
#include "visualization.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(argv[1])) {
        std::cout << "Config file " << argv[1] << " does not exist." << std::endl;
        return 1;
    }

    if (!std::filesystem::is_regular_file(argv[1])) {
        std::cout << "Config file " << argv[1] << " is not a regular file." << std::endl;
        return 1;
    }

    if (std::string(argv[1]).substr(std::string(argv[1]).find_last_of('.') + 1) != "json") {
        std::cout << "Config file " << argv[1] << " is not a JSON file." << std::endl;
        return 1;
    }

    SimulationConfig config = parseConfig(argv[1]);
    std::vector<Particle> initialParticles = parseInitialParticles(argv[1]);
    auto [forceFieldX, forceFieldY] = parseForceField(argv[1], config.width, config.height);
    std::vector<Obstacle> obstacles = parseObstacles(argv[1]);

    auto simulation = Simulation(config);

    simulation.initParticles(initialParticles, obstacles);
    simulation.setupCudaForceField(forceFieldX, forceFieldY);
    simulation.setupCudaObstacles(obstacles);
    simulation.setupCuda();

    // clean up host force fields
    delete[] forceFieldX;
    delete[] forceFieldY;

    auto start_time = std::chrono::high_resolution_clock::now();
    while (!simulation.isFinished()) {
        simulation.step();

        int step = simulation.getCurrentStep();

        // count frozen particles
//        auto particles = simulation.getParticles();
//        int frozenParticles = 0;
//        for (auto particle : particles) {
//            if (!particle.isActive) {
//                frozenParticles++;
//            }
//        }

//        std::cout << "Particles[0]: " << particles[0].x << ", " << particles[0].y << ", " << particles[0].isActive << std::endl;
//        std::cout << "Particles[1]: " << particles[1].x << ", " << particles[1].y << ", " << particles[1].isActive << std::endl;

//        if (step < 1000 && step % 1 == 0) {
//            std::cout << "Step " << step << std::endl;
//            std::cout << "Frozen particles: " << frozenParticles << std::endl;
//        }
//
        if (step % 1000 == 0) {
            std::cout << "Step " << step << std::endl;
//            std::cout << "Frozen particles: " << frozenParticles << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    int lastStep = simulation.getCurrentStep();

    std::cout << "Simulation finished in " << duration / 1000 << "s " << duration % 1000 << "ms." << std::endl;
    std::cout << "Number of steps: " << lastStep + 1 << std::endl;


    // write to csv
    std::ofstream csvFile;
    csvFile.open("output.csv");
    csvFile << "x,y" << std::endl;
    for (auto particle : simulation.getParticles()) {
        csvFile << particle.x << "," << particle.y << std::endl;
    }
    csvFile.close();

    // visualize
    visualizeSimulation(simulation.getParticles(), config, lastStep);

    return 0;
}
