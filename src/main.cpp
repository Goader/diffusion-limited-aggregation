//
// Created by goader on 11/5/23.
//

#include <filesystem>
#include <iostream>
#include "utils.h"
#include "simulation.cuh"

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
    auto simulation = Simulation(config);

    while (!simulation.isFinished()) {
        simulation.step();
    }

    return 0;
}
