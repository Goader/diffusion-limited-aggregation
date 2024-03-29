//
// Created by goader on 11/30/23.
//

#include "particle.cuh"

__global__ void setupRandomStatesKernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__device__ void randomUniform(float* x, curandState* state) {
    *x = curand_uniform(state);
}

__device__ void randomMove(float moveRadius, float* dx, float* dy, curandState* state) {
    float angle = curand_uniform(state) * 2 * M_PI;
    *dx = moveRadius * cos(angle);
    *dy = moveRadius * sin(angle);
}

__device__ void surfaceCollisionPoint(float xa, float ya,  // active particle
                                      float xf, float yf,  // frozen particle
                                      float dx, float dy,  // move vector
                                      float r,             // particle radius
                                      float* x, float* y) {

    // link to the math: https://www.sciencedirect.com/science/article/pii/S0010465511001238#br0150
    // math notion considering the names of the variables:
    // $X_1 = (xa, ya)$
    // $X_d = (xf, yf)$
    // $A = (dx, dy)$
    // $C = \alpha A$
    // $X_2 = X_1 + A$
    // $X_f = X_1 + C$

    // $B = X_2 - X_d$
    // $D = B + C$

    // Considering that $|D| = 2r$:
    // $4r^2 = B \cdot B + 2B \cdot C + C \cdot C$
    // $4r^2 = B \cdot B + 2B \cdot \alpha A + \alpha^2 A \cdot A$

    // $\theta = 2 B \cdot A$,    $\theta$ is $b$ in the $ax^2 + bx + c = 0$ equation
    // $\psi = B \cdot B - 4r^2$, $\psi$ is $c$ in the $ax^2 + bx + c = 0$ equation
    // $\chi = A \cdot A$,        $\chi$ is $a$ in the $ax^2 + bx + c = 0$ equation

    // $\alpha = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ is the solution of the equation
    // $\alpha = \frac{-\theta \pm \sqrt{\theta^2 - 4\chi\psi}}{2\chi}$

    auto x2 = xa + dx, y2 = ya + dy;
    auto bx = x2 - xf, by = y2 - yf;

    auto theta = 2 * (bx * dx + by * dy);
    auto psi = bx * bx + by * by - 4 * r * r;
    auto chi = dx * dx + dy * dy;

    auto alpha = (-theta - sqrt(theta * theta - 4 * chi * psi)) / (2 * chi);

    *x = x2 + alpha * dx;
    *y = y2 + alpha * dy;
}

__global__ void moveParticlesKernel(Particle* particles,
                                    SimulationConfig config,
                                    curandState* states, 
                                    float* forceFieldX, 
                                    float* forceFieldY,
                                    Obstacle* obstacles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // guard against out of bounds access and inactive particles
    auto particle = particles + idx;
    if (idx > config.numParticles || !particle->isActive) {
        return;
    }

    // generate random move
    float dx, dy;
    randomMove(config.moveRadius, &dx, &dy, &states[idx]);

    int xGridIdx = static_cast<int>(particle->x);
    int yGridIdx = static_cast<int>(particle->y);
    int gridIdx = yGridIdx * config.width + xGridIdx;

    // add force drag
    dx += forceFieldX[gridIdx];
    dy += forceFieldY[gridIdx];

    // save the old position
    particle->oldX = particle->x;
    particle->oldY = particle->y;

    // move the particle
    particle->x += dx;
    particle->y += dy;

    // handle obstacle collision and reflection
    for (int i = 0; i < config.numObstacles; i++) {
        auto obstacle = obstacles + i;

        float obstacleLeftX = static_cast<float>(obstacle->xTopLeft);
        float obstacleRightX = static_cast<float>(obstacle->xTopLeft + obstacle->recWidth);
        float obstacleTopY = static_cast<float>(obstacle->yTopLeft);
        float obstacleBottomY = static_cast<float>(obstacle->yTopLeft + obstacle->recHeight);

        // check if the particle is within the obstacle
        if (particle->x >= obstacleLeftX && particle->x <= obstacleRightX &&
            particle->y >= obstacleTopY && particle->y <= obstacleBottomY) {

            // if an obstacle has been inside for at least 2 steps (e.g. due to an initial position in config)
            if (particle->oldX >= obstacleLeftX && particle->oldX <= obstacleRightX &&
                particle->oldY >= obstacleTopY && particle->oldY <= obstacleBottomY) {
                break;
            }

            // calculate the line equation of the particle's trajectory
            float slope = dy / dx;
            float intercept = particle->oldY - slope * particle->oldX;

            float collisionX, collisionY;
            float minDistance = 10000.0f;
            float edgeX[4] = {obstacleLeftX, obstacleRightX, particle->x, particle->x};
            float edgeY[4] = {particle->y, particle->y, obstacleTopY, obstacleBottomY};

            // check intersection with each edge of the obstacle
            // (the particle will collide with the nearest edge along its trajectory)
            for (int j = 0; j < 4; j++) {
                float x = edgeX[j];
                float y = edgeY[j];

                if (j < 2) {y = slope * x + intercept;}  // vertical edges
                else {x = (y - intercept) / slope;}  // horizontal edges

                // Check if the intersection is within the bounds of the edge and along the trajectory
                if (x >= fmin(obstacleLeftX, particle->oldX) && x <= fmax(obstacleRightX, particle->oldX) &&
                    y >= fmin(obstacleTopY, particle->oldY) && y <= fmax(obstacleBottomY, particle->oldY)) {
                    float distance = hypot(x - particle->oldX, y - particle->oldY);
                    if (distance < minDistance) {
                        minDistance = distance;
                        collisionX = x;
                        collisionY = y;
                    }
                }
            }

            // update old coordinates to the collision point
            particle->oldX = collisionX;
            particle->oldY = collisionY;

            if (collisionX == obstacleLeftX || collisionX == obstacleRightX) {dx = -dx;}  // reflect horizontally
            else if (collisionY == obstacleTopY || collisionY == obstacleBottomY) {dy = -dy;}  // reflect vertically
            else {break;} // should not happen

            // update new coordinates
            float remainingDist = config.moveRadius - minDistance;
            particle->x = collisionX + dx * remainingDist;
            particle->y = collisionY + dy * remainingDist;
            
            // break after reflecting from obstacle
            break;
        }
    }

    // clip the coordinates to stay within the bounds of the simulation
    particle->x = fmax(0.f, fmin(particle->x, static_cast<float>(config.width)));
    particle->y = fmax(0.f, fmin(particle->y, static_cast<float>(config.height)));
}

__global__ void checkCollisionsKernel(Particle* particles,
                                      SimulationConfig config) {
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // guard against out of bounds access and duplication
    if (xIdx >= config.numParticles || yIdx >= config.numParticles || xIdx >= yIdx) {
        return;
    }

    auto particleX = particles + xIdx;
    auto particleY = particles + yIdx;

    // one of the particles must be active, and the other must be frozen
    if (!(particleX->isActive ^ particleY->isActive)) {
        return;
    }

    auto frozenIdx = !particleX->isActive ? xIdx : yIdx;
    auto frozenParticle = !particleX->isActive ? particleX : particleY;
    auto activeParticle = particleX->isActive ? particleX : particleY;

    auto squaredDistance = pow(frozenParticle->x - activeParticle->x, 2)
                              + pow(frozenParticle->y - activeParticle->y, 2);
    auto freezeRadiusSquared = pow(config.particleRadius * 2, 2);  // todo: make it a parameter (?)

    // if the particle is not within the freeze radius of the frozen particle, it cannot freeze
    if (squaredDistance > freezeRadiusSquared) {
        return;
    }

    // if the active particle is within the freeze radius of the frozen particle, freeze it
    // fixme should we somehow add constraint on the closest particle?
    activeParticle->collidedParticleIdx = frozenIdx;
}

__global__ void freezeParticlesKernel(Particle* particles,
                                      SimulationConfig config,
                                      curandState* states,
                                      bool* allFrozen,
                                      int currentStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // guard against out of bounds access and inactive particles
    auto particle = particles + idx;
    if (idx > config.numParticles || !particle->isActive) {
        return;
    }

    // if the particle has not collided, it cannot freeze
    if (particle->collidedParticleIdx == -1) {
        // allows to skip aggregation check if all particles are frozen
        *allFrozen = false;
        return;
    }

    auto activeParticle = particle;
    auto frozenParticle = particles + particle->collidedParticleIdx;

    // randomize the stickiness of the particle
    float u;
    randomUniform(&u, &states[idx]);
    auto isSticky = u < config.stickiness;

    float collision_x, collision_y;
    surfaceCollisionPoint(activeParticle->oldX, activeParticle->oldY,
                          frozenParticle->x, frozenParticle->y,
                          activeParticle->x - activeParticle->oldX, activeParticle->y - activeParticle->oldY,
                          config.particleRadius,
                          &collision_x, &collision_y);

    // if the particle is not sticky, it will bounce off the frozen particle
    if (!isSticky) {
        // real distance passed by the particle
        auto dx = collision_x - activeParticle->oldX,
             dy = collision_y - activeParticle->oldY;
        auto distancePassed = sqrt(dx * dx + dy * dy);
        auto distanceLeft = config.moveRadius - distancePassed;

        // normalizing the vector
        dx = dx / distancePassed;
        dy = dy / distancePassed;

        // collision surface vector
        auto surface_x = collision_x - frozenParticle->x,
             surface_y = collision_y - frozenParticle->y;

        // normalizing the vector
        auto surface_length = sqrt(surface_x * surface_x + surface_y * surface_y);
        surface_x = surface_x / surface_length;
        surface_y = surface_y / surface_length;

        // reflection vector
        auto reflection_x = dx - 2 * (dx * surface_x + dy * surface_y) * surface_x,
             reflection_y = dy - 2 * (dx * surface_x + dy * surface_y) * surface_y;

        // move the particle along the reflection vector
        activeParticle->x = collision_x + reflection_x * distanceLeft;
        activeParticle->y = collision_y + reflection_y * distanceLeft;

        particle->collidedParticleIdx = -1;  // reset the collided particle index
        *allFrozen = false;
        return;
    }

    // if the particle is sticky, it will freeze to the frozen particle
    activeParticle->isActive = false;
    activeParticle->frozenAtStep = currentStep;

    // adjust the particle's position to be on the surface of the frozen particle
    activeParticle->x = collision_x;
    activeParticle->y = collision_y;
}
