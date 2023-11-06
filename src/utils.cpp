//
// Created by goader on 11/5/23.
//

#include "utils.h"

using json = nlohmann::json;

SimulationConfig parseConfig(const std::string& filename) {
    std::ifstream file(filename);
    json j;
    file >> j;
    return SimulationConfig(j["width"],
                            j["height"],
                            j["stickiness"],
                            j["moveRadius"],
                            j["particleRadius"],
                            j["numParticles"],
                            j["maxParticles"],
                            j["respawnParticles"],
                            j["seed"]);
}
