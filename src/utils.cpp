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
                            j["obstacleRectangles"].size(),
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

std::pair<float*, float*> parseForceField(const std::string& filename, int width, int height) {
    std::ifstream file(filename);
    json j;
    file >> j;

    float* forceFieldX = new float[width * height]();  // initialize to 0
    float* forceFieldY = new float[width * height]();  // initialize to 0

    auto forceRectangles = j["forceRectangles"];
    for (auto& rectangle : forceRectangles) {
        int xTopLeft = rectangle["xTopLeft"];
        int yTopLeft = rectangle["yTopLeft"];
        int recWidth = rectangle["recWidth"];
        int recHeight = rectangle["recHeight"];
        float dragRadius = rectangle["dragRadius"];
        float dragAngle = rectangle["dragAngle"];

        float dx = dragRadius * cos(dragAngle);
        float dy = dragRadius * sin(dragAngle);

        for (int y = yTopLeft; y < yTopLeft + recHeight; ++y) {
            for (int x = xTopLeft; x < xTopLeft + recWidth; ++x) {
                forceFieldX[y * width + x] = dx;
                forceFieldY[y * width + x] = dy;
            }
        }
    }

    return {forceFieldX, forceFieldY};
}

std::vector<Obstacle> parseObstacles(const std::string& filename) {
    std::ifstream file(filename);
    json j;
    file >> j;

    auto obstacleRectangles = j["obstacleRectangles"];
    std::vector<Obstacle> obstacles;
    for (auto& rectangle : obstacleRectangles) {
        int xTopLeft = rectangle["xTopLeft"];
        int yTopLeft = rectangle["yTopLeft"];
        int recWidth = rectangle["recWidth"];
        int recHeight = rectangle["recHeight"];
        obstacles.push_back(Obstacle(xTopLeft, yTopLeft, recWidth, recHeight));
    }

    return obstacles;
}
