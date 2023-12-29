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
                            j["seed"]);
}

std::vector<Particle> parseInitialParticles(const std::string& filename) {
    std::ifstream file(filename);
    json j;
    file >> j;

    auto initialParticles = j["initialParticles"];
    std::vector<Particle> initialParticlesVector;
    for (auto& initialParticle : initialParticles) {
        initialParticlesVector.push_back(
                Particle(initialParticle["x"], initialParticle["y"],
                         initialParticle["x"], initialParticle["y"],
                         false, -1, -1)
        );
    }

    return initialParticlesVector;
}
