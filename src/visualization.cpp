#include "visualization.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>
#include <stack>

// Shader sources
const GLchar* vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec2 position;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
    }
)glsl";

const GLchar* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(0.5, 0.0, 0.5, 1.0); // Purple color
    }
)glsl";


GLuint compileShader(GLenum type, const GLchar* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    return shader;
}


// fixme: ChatGPT generated

void visualizeSimulation(const std::vector<Particle>& particles, const SimulationConfig& config, int lastStep) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(config.width, config.height, "Diffusion Limited Aggregation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }

    // Compile shaders
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Particle) * particles.size(), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, x));
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0);

    // Sort particles by their frozenAtStep in descending order
    std::vector<Particle> sortedParticles = particles;
    std::sort(sortedParticles.begin(), sortedParticles.end(), [](const Particle& a, const Particle& b) {
        return a.frozenAtStep > b.frozenAtStep;
    });

    // Create a stack for particles
    std::stack<Particle> particleStack;
    for (const auto& particle : sortedParticles) {
        particleStack.push(particle);
    }

    // Visualization loop
    for (int step = 0; step <= lastStep && !glfwWindowShouldClose(window); step++) {
        glClear(GL_COLOR_BUFFER_BIT);

        if (step % 1000 == 0) {
            std::cout << "Vis step " << step << std::endl;
        }

        std::vector<Particle> particlesToDraw;

        while (!particleStack.empty() && particleStack.top().frozenAtStep == step) {
            particlesToDraw.push_back(particleStack.top());
            particleStack.pop();
        }

        if (!particlesToDraw.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Particle) * particlesToDraw.size(), particlesToDraw.data());

            glUseProgram(shaderProgram);
            glBindVertexArray(vao);

            glDrawArrays(GL_POINTS, 0, particlesToDraw.size());

            glBindVertexArray(0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    while(!glfwWindowShouldClose(window)) {}

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
}
