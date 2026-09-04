#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <string>

// ==========================================
// 1. MINIMAL 4x4 MATRIX MATH LIBRARY
// ==========================================
void mat4_identity(float* m) {
    for(int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

void mat4_perspective(float* m, float fov, float aspect, float near, float far) {
    float f = 1.0f / tan(fov * 0.5f);
    mat4_identity(m);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0f;
    m[14] = (2.0f * far * near) / (near - far);
    m[15] = 0.0f;
}

void mat4_translate(float* m, float x, float y, float z) {
    mat4_identity(m);
    m[12] = x; m[13] = y; m[14] = z;
}

void mat4_rotateX(float* m, float angle) {
    mat4_identity(m);
    float c = cos(angle), s = sin(angle);
    m[5] = c; m[6] = s;
    m[9] = -s; m[10] = c;
}

void mat4_rotateY(float* m, float angle) {
    mat4_identity(m);
    float c = cos(angle), s = sin(angle);
    m[0] = c; m[2] = -s;
    m[8] = s; m[10] = c;
}

void mat4_multiply(float* out, const float* a, const float* b) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c * 4 + r] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
            }
        }
    }
}

// ==========================================
// 2. GLOBAL VARIABLES & DATA
// ==========================================
GLFWwindow* g_window = nullptr;
GLuint g_vao = 0, g_vbo = 0, g_ebo = 0, g_shaderProgram = 0;

float g_cameraPosition[3] = {0.0f, 0.0f, 5.0f};
float g_cameraRotation[3] = {0.0f, 0.0f, 0.0f}; // Pitch, Yaw, Roll (Degrees)
float g_speed = 0.1f;

// 24 Vertices (Position X,Y,Z + Color R,G,B)
const float vertices[] = {
    // Front face
    -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,   0.5f, -0.5f,  0.5f,  0.0f, 1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  -0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 0.0f,
    // Back face
    -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,  -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,   0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 0.0f,
    // Top face
    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f, 1.0f,   0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,  -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,
    // Bottom face
    -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,  -0.5f, -0.5f, -0.5f,  0.0f, 0.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 0.0f, 0.0f,   0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,
    // Right face
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,   0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 0.0f,   0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 0.0f,
    // Left face
    -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.0f,  -0.5f, -0.5f, -0.5f,  0.0f, 0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f,  -0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 0.0f
};

const GLuint indices[] = {
    0, 1, 2, 2, 3, 0,   4, 5, 6, 6, 7, 4,   8, 9, 10, 10, 11, 8,
    12, 13, 14, 14, 15, 12,  16, 17, 18, 18, 19, 16,  20, 21, 22, 22, 23, 20
};

// ==========================================
// 3. SHADERS (Modern OpenGL Requirement)
// ==========================================
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;
    uniform mat4 uMVP;
    out vec3 vColor;
    void main() {
        gl_Position = uMVP * vec4(aPos, 1.0);
        vColor = aColor;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 vColor;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(vColor, 1.0);
    }
)";

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
    }
    return shader;
}

void setupShaders() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
    g_shaderProgram = glCreateProgram();
    glAttachShader(g_shaderProgram, vertexShader);
    glAttachShader(g_shaderProgram, fragmentShader);
    glLinkProgram(g_shaderProgram);
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

// ==========================================
// 4. SETUP & DRAW
// ==========================================
void loadVertices() {
    glGenVertexArrays(1, &g_vao);
    glGenBuffers(1, &g_vbo);
    glGenBuffers(1, &g_ebo);

    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void drawScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(g_shaderProgram);

    // 1. Calculate Matrices
    float proj[16], view[16], model[16], temp[16], mvp[16];
    float rotX[16], rotY[16], trans[16];
    
    // Convert degrees to radians for math
    float pitchRad = g_cameraRotation[0] * 3.14159265f / 180.0f;
    float yawRad = g_cameraRotation[1] * 3.14159265f / 180.0f;

    mat4_perspective(proj, 3.14159265f / 3.0f, 800.0f / 600.0f, 0.1f, 100.0f);
    mat4_identity(model);
    
    // Camera View Matrix: Translate * RotateY * RotateX
    mat4_rotateX(rotX, -pitchRad);
    mat4_rotateY(rotY, -yawRad);
    mat4_translate(trans, -g_cameraPosition[0], -g_cameraPosition[1], -g_cameraPosition[2]);
    
    mat4_multiply(temp, rotY, rotX);
    mat4_multiply(view, trans, temp);
    
    mat4_multiply(mvp, proj, view); // MVP = Projection * View * Model

    // 2. Pass MVP to Shader
    GLint mvpLoc = glGetUniformLocation(g_shaderProgram, "uMVP");
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, mvp);

    // 3. Draw
    glBindVertexArray(g_vao);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glfwSwapBuffers(g_window);
}

void handleInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    static double lastX = 800.0 / 2.0, lastY = 600.0 / 2.0;
    double currentX, currentY;
    glfwGetCursorPos(window, &currentX, &currentY);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        g_cameraRotation[1] += (currentX - lastX) * 0.2f; // Yaw
        g_cameraRotation[0] += (lastY - currentY) * 0.2f; // Pitch
        // Clamp pitch to prevent flipping
        if (g_cameraRotation[0] > 89.0f) g_cameraRotation[0] = 89.0f;
        if (g_cameraRotation[0] < -89.0f) g_cameraRotation[0] = -89.0f;
    }
    lastX = currentX; lastY = currentY;

    // 6DOF Movement based on Yaw direction
    float yawRad = g_cameraRotation[1] * 3.14159265f / 180.0f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        g_cameraPosition[0] -= sin(yawRad) * g_speed;
        g_cameraPosition[2] -= cos(yawRad) * g_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        g_cameraPosition[0] += sin(yawRad) * g_speed;
        g_cameraPosition[2] += cos(yawRad) * g_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        g_cameraPosition[0] -= cos(yawRad) * g_speed;
        g_cameraPosition[2] += sin(yawRad) * g_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        g_cameraPosition[0] += cos(yawRad) * g_speed;
        g_cameraPosition[2] -= sin(yawRad) * g_speed;
    }
}

int main() {
    if (!glfwInit()) return -1;

    // CRITICAL: Request Modern OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on Mac
    #endif

    g_window = glfwCreateWindow(800, 600, "Modern 6DOF Camera", nullptr, nullptr);
    if (!g_window) {
        std::cerr << "Failed to create GLFW window. Your GPU might not support OpenGL 3.3." << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(g_window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
    glfwSwapInterval(1);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);

    setupShaders();
    loadVertices();

    while (!glfwWindowShouldClose(g_window)) {
        handleInput(g_window);
        drawScene();
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}