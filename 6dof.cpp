// descent6dof.cpp
//
// Descent-style 6DOF camera demo using GLFW and legacy OpenGL.
//
// Controls:
//   Mouse       : look / pitch / yaw, when captured
//   L           : toggle mouse capture
//   W / S       : forward / backward
//   A / D       : strafe left / right
//   Space       : move up
//   Ctrl or C   : move down
//   Q / E       : roll left / right
//   Shift       : boost
//   R           : reset camera
//   Esc         : quit
//g++ -std=c++11 6dof.cpp -o app   -I/usr/local/include   -L/usr/local/lib   -lglfw   -framework OpenGL   -framework Cocoa   -framework IOKit   -framework CoreVideo -L/Users/macbook2015/Desktop/brew/lib -I/Users/macbook2015/Desktop/brew/include
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>

//------------------------------------------------------------------------------
// Global input state
//------------------------------------------------------------------------------

static double mouseXOffset = 0.0;
static double mouseYOffset = 0.0;

static double lastMouseX = 0.0;
static double lastMouseY = 0.0;

static bool firstMouse = true;
static bool mouseLookEnabled = true;
static bool resetRequested = false;

//------------------------------------------------------------------------------
// Minimal vector/quaternion math
//------------------------------------------------------------------------------

struct Vec3
{
    float x, y, z;

    Vec3(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f)
        : x(x_), y(y_), z(z_)
    {
    }
};

static inline Vec3 operator+(Vec3 a, Vec3 b)
{
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline Vec3 operator-(Vec3 a, Vec3 b)
{
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline Vec3 operator-(Vec3 a)
{
    return Vec3(-a.x, -a.y, -a.z);
}

static inline Vec3 operator*(Vec3 a, float s)
{
    return Vec3(a.x * s, a.y * s, a.z * s);
}

static inline Vec3 operator*(float s, Vec3 a)
{
    return a * s;
}

static inline Vec3& operator+=(Vec3& a, Vec3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

static inline Vec3& operator-=(Vec3& a, Vec3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

static inline Vec3& operator*=(Vec3& a, float s)
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}

static inline float dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3 cross(Vec3 a, Vec3 b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline float lengthSquared(Vec3 a)
{
    return dot(a, a);
}

static inline float length(Vec3 a)
{
    return std::sqrt(lengthSquared(a));
}

static inline Vec3 normalize(Vec3 a)
{
    float len = length(a);
    if (len > 1e-6f)
        return a * (1.0f / len);

    return Vec3(0.0f, 0.0f, 0.0f);
}

struct Quat
{
    float w, x, y, z;

    Quat(float w_ = 1.0f, float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f)
        : w(w_), x(x_), y(y_), z(z_)
    {
    }
};

static inline Quat operator*(Quat a, Quat b)
{
    return Quat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    );
}

static inline float length(Quat q)
{
    return std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

static inline Quat normalize(Quat q)
{
    float len = length(q);
    if (len > 1e-6f)
    {
        float inv = 1.0f / len;
        return Quat(q.w * inv, q.x * inv, q.y * inv, q.z * inv);
    }

    return Quat();
}

static inline Quat axisAngle(Vec3 axis, float angleRadians)
{
    axis = normalize(axis);

    float half = angleRadians * 0.5f;
    float s = std::sin(half);
    float c = std::cos(half);

    return Quat(c, axis.x * s, axis.y * s, axis.z * s);
}

static inline Vec3 rotate(Quat q, Vec3 v)
{
    // Optimized quaternion rotation:
    // v' = v + q.w * t + cross(q.xyz, t), where t = 2 * cross(q.xyz, v)
    Vec3 u(q.x, q.y, q.z);
    Vec3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}

//------------------------------------------------------------------------------
// GLFW callbacks
//------------------------------------------------------------------------------

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    if (key == GLFW_KEY_L && action == GLFW_PRESS)
    {
        mouseLookEnabled = !mouseLookEnabled;

        glfwSetInputMode(
            window,
            GLFW_CURSOR,
            mouseLookEnabled ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL
        );

        firstMouse = true;
        mouseXOffset = 0.0;
        mouseYOffset = 0.0;
    }

    if (key == GLFW_KEY_R && action == GLFW_PRESS)
    {
        resetRequested = true;
    }
}

static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    (void)window;

    if (!mouseLookEnabled)
    {
        lastMouseX = xpos;
        lastMouseY = ypos;
        return;
    }

    if (firstMouse)
    {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    mouseXOffset += xpos - lastMouseX;
    mouseYOffset += ypos - lastMouseY;

    lastMouseX = xpos;
    lastMouseY = ypos;
}

//------------------------------------------------------------------------------
// Simple immediate-mode drawing helpers
//------------------------------------------------------------------------------

static void drawWireCube(float size)
{
    const float h = size * 0.5f;

    const float v[8][3] =
    {
        { -h, -h, -h }, // 0
        {  h, -h, -h }, // 1
        {  h,  h, -h }, // 2
        { -h,  h, -h }, // 3
        { -h, -h,  h }, // 4
        {  h, -h,  h }, // 5
        {  h,  h,  h }, // 6
        { -h,  h,  h }  // 7
    };

    static const int edges[12][2] =
    {
        // Bottom face
        { 0, 1 },
        { 1, 2 },
        { 2, 3 },
        { 3, 0 },

        // Top face
        { 4, 5 },
        { 5, 6 },
        { 6, 7 },
        { 7, 4 },

        // Connect bottom/top
        { 0, 4 },
        { 1, 5 },
        { 2, 6 },
        { 3, 7 }
    };

    glBegin(GL_LINES);
    for (int i = 0; i < 12; ++i)
    {
        glVertex3fv(v[edges[i][0]]);
        glVertex3fv(v[edges[i][1]]);
    }
    glEnd();
}

static void drawGrid(
    float y,
    float halfExtent,
    float step,
    float r,
    float g,
    float b
)
{
    if (step <= 0.0f)
        return;

    glColor3f(r, g, b);

    glBegin(GL_LINES);

    for (float x = -halfExtent; x <= halfExtent + 0.001f; x += step)
    {
        glVertex3f(x, y, -halfExtent);
        glVertex3f(x, y,  halfExtent);
    }

    for (float z = -halfExtent; z <= halfExtent + 0.001f; z += step)
    {
        glVertex3f(-halfExtent, y, z);
        glVertex3f( halfExtent, y, z);
    }

    glEnd();
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main(void)
{
    if (!glfwInit())
    {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return -1;
    }

    // Compatibility-profile legacy OpenGL.
    // This demo uses glBegin/glEnd-style rendering for simplicity.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        1280,
        800,
        "Descent-style 6DOF Camera",
        nullptr,
        nullptr
    );

    if (!window)
    {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);

    // Start with mouse captured.
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

    std::printf("Descent-style 6DOF camera demo\n");
    std::printf("------------------------------\n");
    std::printf("Mouse       : look\n");
    std::printf("L           : toggle mouse capture\n");
    std::printf("W/S         : forward/backward\n");
    std::printf("A/D         : strafe left/right\n");
    std::printf("Space       : up\n");
    std::printf("Ctrl or C   : down\n");
    std::printf("Q/E         : roll\n");
    std::printf("Shift       : boost\n");
    std::printf("R           : reset camera\n");
    std::printf("Esc         : quit\n\n");

    // Camera state.
    Vec3 position(0.0f, 0.0f, 18.0f);
    Vec3 velocity(0.0f, 0.0f, 0.0f);
    Quat orientation; // Identity quaternion.

    double lastTime = glfwGetTime();

    const float mouseSensitivity = 0.0022f;
    const float rollSpeed = 2.0f; // radians/sec

    while (!glfwWindowShouldClose(window))
    {
        double currentTime = glfwGetTime();
        float dt = float(currentTime - lastTime);
        lastTime = currentTime;

        // Avoid huge simulation steps if the window was dragged/minimized.
        if (dt > 0.05f)
            dt = 0.05f;

        if (resetRequested)
        {
            position = Vec3(0.0f, 0.0f, 18.0f);
            velocity = Vec3(0.0f, 0.0f, 0.0f);
            orientation = Quat();
            resetRequested = false;
            mouseXOffset = 0.0;
            mouseYOffset = 0.0;
        }

        //----------------------------------------------------------------------
        // Rotation input
        //----------------------------------------------------------------------

        float yaw = 0.0f;
        float pitch = 0.0f;

        if (mouseLookEnabled)
        {
            // Moving mouse right turns right.
            // Moving mouse down looks down.
            yaw = float(-mouseXOffset * mouseSensitivity);
            pitch = float(-mouseYOffset * mouseSensitivity);
        }

        mouseXOffset = 0.0;
        mouseYOffset = 0.0;

        float rollInput = 0.0f;

        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            rollInput += 1.0f;

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            rollInput -= 1.0f;

        float roll = rollInput * rollSpeed * dt;

        // Apply rotations around the camera's current local axes.
        // This is what gives the full 6DOF Descent-like feel.
        if (yaw != 0.0f)
        {
            Vec3 up = rotate(orientation, Vec3(0.0f, 1.0f, 0.0f));
            orientation = normalize(axisAngle(up, yaw) * orientation);
        }

        if (pitch != 0.0f)
        {
            Vec3 right = rotate(orientation, Vec3(1.0f, 0.0f, 0.0f));
            orientation = normalize(axisAngle(right, pitch) * orientation);
        }

        if (roll != 0.0f)
        {
            Vec3 forward = rotate(orientation, Vec3(0.0f, 0.0f, -1.0f));
            orientation = normalize(axisAngle(forward, roll) * orientation);
        }

        //----------------------------------------------------------------------
        // Movement input
        //----------------------------------------------------------------------

        Vec3 forward = rotate(orientation, Vec3(0.0f, 0.0f, -1.0f));
        Vec3 right   = rotate(orientation, Vec3(1.0f, 0.0f,  0.0f));
        Vec3 up      = rotate(orientation, Vec3(0.0f, 1.0f,  0.0f));

        Vec3 desiredDir(0.0f, 0.0f, 0.0f);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            desiredDir += forward;

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            desiredDir -= forward;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            desiredDir += right;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            desiredDir -= right;

        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            desiredDir += up;

        if (
            glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS
        )
        {
            desiredDir -= up;
        }

        if (lengthSquared(desiredDir) > 1e-6f)
            desiredDir = normalize(desiredDir);

        bool boost =
            glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

        float acceleration = boost ? 135.0f : 45.0f;

        // Simple inertia/damping for a floaty Descent-like movement feel.
        velocity += desiredDir * (acceleration * dt);
        velocity *= std::exp(-3.0f * dt);
        position += velocity * dt;

        //----------------------------------------------------------------------
        // Render
        //----------------------------------------------------------------------

        int fbWidth = 0;
        int fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

        if (fbWidth < 1 || fbHeight < 1)
        {
            // Window is minimized or has zero size.
            glfwWaitEventsTimeout(0.1);
            continue;
        }

        glViewport(0, 0, fbWidth, fbHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Projection matrix.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        const float pi = 3.14159265358979323846f;
        const float fovY = 70.0f * pi / 180.0f;

        float aspect = float(fbWidth) / float(fbHeight);
        float tanHalfFov = std::tan(fovY * 0.5f);

        float zNear = 0.1f;
        float zFar = 1200.0f;

        glFrustum(
            -aspect * zNear * tanHalfFov,
             aspect * zNear * tanHalfFov,
            -zNear * tanHalfFov,
             zNear * tanHalfFov,
             zNear,
             zFar
        );

        // View matrix.
        //
        // OpenGL camera convention:
        //   +X = right
        //   +Y = up
        //   -Z = forward
        //
        // We build the inverse camera transform manually from camera axes.
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        Vec3 back = -forward;

        GLfloat view[16] =
        {
            right.x, up.x, back.x, 0.0f,
            right.y, up.y, back.y, 0.0f,
            right.z, up.z, back.z, 0.0f,
            -dot(right, position),
            -dot(up, position),
            -dot(back, position),
            1.0f
        };

        glLoadMatrixf(view);

        //----------------------------------------------------------------------
        // Draw scene
        //----------------------------------------------------------------------

        glLineWidth(1.5f);

        float time = float(currentTime);

        // Floor and ceiling grids.
        drawGrid(-14.0f, 140.0f, 14.0f, 0.16f, 0.18f, 0.24f);
        drawGrid( 14.0f, 140.0f, 14.0f, 0.16f, 0.18f, 0.24f);

        // Origin marker cube.
        glPushMatrix();
        glColor3f(1.0f, 0.75f, 0.25f);
        drawWireCube(4.0f);
        glPopMatrix();

        // A "mine tunnel" made of wireframe cubes.
        for (int i = 0; i < 36; ++i)
        {
            float fi = float(i);
            float z = -fi * 12.0f;

            glPushMatrix();
            glTranslatef(0.0f, 0.0f, z);
            glRotatef(fi * 4.0f, 0.0f, 0.0f, 1.0f);

            float pulse = 0.5f + 0.5f * std::sin(fi * 0.35f);
            glColor3f(
                0.20f + 0.35f * pulse,
                0.55f + 0.25f * pulse,
                0.85f
            );

            drawWireCube(10.0f);
            glPopMatrix();
        }

        // Some floating debris cubes.
        for (int i = 0; i < 60; ++i)
        {
            float fi = float(i);

            float x = 20.0f * std::sin(fi * 1.7f);
            float y = 10.0f * std::cos(fi * 2.3f);
            float z = -25.0f - fi * 18.0f;

            float size = 2.0f + float(i % 6);

            glPushMatrix();
            glTranslatef(x, y, z);
            glRotatef(time * 8.0f + fi * 19.0f, 0.0f, 1.0f, 0.0f);

            float green = 0.35f + 0.65f * (0.5f + 0.5f * std::sin(fi * 0.8f));
            glColor3f(0.85f, green, 0.45f);

            drawWireCube(size);
            glPopMatrix();
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}