#include "visualization.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>
#include <stack>



const SimulationConfig* globalConfig;


const GLchar* vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec2 position;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        gl_PointSize = 8.0;
    }
)glsl";


const GLchar* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;
    void main() {
        float radius = 0.5; // normalized radius of the circle
        vec2 center = gl_PointCoord - vec2(0.5, 0.5); // shift point coord to the center
        if (length(center) > radius) {
            discard; // discard fragments outside the radius
        }
        FragColor = vec4(0.0, 1.0, 0.0, 1.0); // green color
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


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Calculate the new viewport dimensions while maintaining the aspect ratio
    float aspectRatio = static_cast<float>(globalConfig->width) / static_cast<float>(globalConfig->height);
    int viewportWidth, viewportHeight;

    if (static_cast<float>(width) / height > aspectRatio) {  // Window is too wide
        viewportHeight = height;
        viewportWidth = static_cast<int>(height * aspectRatio);  // Window is too tall
    } else {
        viewportWidth = width;
        viewportHeight = static_cast<int>(width / aspectRatio);
    }
    int viewportX = (width - viewportWidth) / 2;
    int viewportY = (height - viewportHeight) / 2;
    // Update the viewport
    glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
}


void visualizeSimulation(const std::vector<Particle>& particles, const SimulationConfig& config, int lastStep) {
    
    globalConfig = &config;

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

    if (GLEW_VERSION_1_3) {
        std::cout << "OpenGL 1.3 supported" << std::endl;
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
    glEnable(GL_PROGRAM_POINT_SIZE);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLParticle) * particles.size(), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GLParticle), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0);

    // Sort particles by their frozenAtStep in descending order
    std::vector<Particle> sortedParticles = particles;
    std::sort(sortedParticles.begin(), sortedParticles.end(), [](const Particle& a, const Particle& b) {
        return a.frozenAtStep > b.frozenAtStep;
    });

    std::cout << "Sorted top: " << sortedParticles.front().frozenAtStep << " end: " << sortedParticles.back().frozenAtStep << std::endl;

    // Create a stack for particles
    std::stack<Particle> particleStack;
    for (const auto& particle : sortedParticles) {
        particleStack.push(particle);
    }

    std::vector<GLParticle> particlePositionsToDraw;

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  // white background

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  // resize callback
    glViewport(0, 0, config.width, config.height);

    glUseProgram(shaderProgram);

    const int SKIP_STEPS_SIZE = 8;

    // Visualization loop
    for (int step = -1; step <= lastStep + SKIP_STEPS_SIZE && !glfwWindowShouldClose(window); step += SKIP_STEPS_SIZE) {
        bool newParticlesAdded = false;

        if (step % 1000 == 0) {
            std::cout << "Vis step " << step << "   Stack size: " << particleStack.size() << std::endl;
        }

        while (!particleStack.empty() && particleStack.top().frozenAtStep <= step) {
            auto &p = particleStack.top();
            particlePositionsToDraw.push_back({(p.x / config.width) * 2.0f - 1.0f, (p.y / config.height) * 2.0f - 1.0f});
            particleStack.pop();
            newParticlesAdded = true;
        }

        if (newParticlesAdded) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLParticle) * particlePositionsToDraw.size(), particlePositionsToDraw.data());
        }

        glClear(GL_COLOR_BUFFER_BIT);
        
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, particlePositionsToDraw.size());
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    std::cout << "Simulation's visualization finished" << std::endl;

    // Keep displaying the final state until the window is closed
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLParticle) * particlePositionsToDraw.size(), particlePositionsToDraw.data());

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(vao);

        glDrawArrays(GL_POINTS, 0, particlePositionsToDraw.size());

        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
}
