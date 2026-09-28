// maze3d.cpp
//
// OpenGL 4.1 core labyrinth maze game.
// Dependencies: GLFW, GLEW, libpng.
// No GLM.

#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <png.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static constexpr float kPi = 3.14159265358979323846f;

static float radians(float deg)
{
    return deg * kPi / 180.0f;
}

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;

    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
};

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

static Vec2 operator+(const Vec2& a, const Vec2& b)
{
    return Vec2(a.x + b.x, a.y + b.y);
}

static Vec2 operator-(const Vec2& a, const Vec2& b)
{
    return Vec2(a.x - b.x, a.y - b.y);
}

static Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static Vec3 operator*(const Vec3& a, float s)
{
    return Vec3(a.x * s, a.y * s, a.z * s);
}

static Vec3 operator*(float s, const Vec3& a)
{
    return a * s;
}

static Vec3 operator/(const Vec3& a, float s)
{
    return Vec3(a.x / s, a.y / s, a.z / s);
}

static float dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static float length(const Vec3& v)
{
    return std::sqrt(dot(v, v));
}

static Vec3 normalize(const Vec3& v)
{
    float l = length(v);
    if (l < 1e-6f)
        return Vec3(0.0f, 1.0f, 0.0f);
    return v / l;
}

struct Mat4
{
    float m[16] = {};
};

static Mat4 identity()
{
    Mat4 r;
    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 multiply(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
            {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

static Mat4 translate(const Vec3& t)
{
    Mat4 r = identity();
    r.m[12] = t.x;
    r.m[13] = t.y;
    r.m[14] = t.z;
    return r;
}

static Mat4 scale(const Vec3& s)
{
    Mat4 r;
    r.m[0] = s.x;
    r.m[5] = s.y;
    r.m[10] = s.z;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 perspectiveMatrix(float fovyRad, float aspect, float zNear, float zFar)
{
    Mat4 r;
    float f = 1.0f / std::tan(fovyRad * 0.5f);

    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zFar + zNear) / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    r.m[15] = 0.0f;

    return r;
}

static Mat4 orthoMatrix(float left, float right, float bottom, float top, float zNear, float zFar)
{
    Mat4 r;
    r.m[0] = 2.0f / (right - left);
    r.m[5] = 2.0f / (top - bottom);
    r.m[10] = -2.0f / (zFar - zNear);
    r.m[12] = -(right + left) / (right - left);
    r.m[13] = -(top + bottom) / (top - bottom);
    r.m[14] = -(zFar + zNear) / (zFar - zNear);
    r.m[15] = 1.0f;
    return r;
}

// Custom view matrix.
// Camera looks along +front in world space.
// OpenGL clip space expects camera looking along -Z, so third basis vector is -front.
static Mat4 viewMatrix(const Vec3& eye, const Vec3& front)
{
    Vec3 f = normalize(front);
    Vec3 r = normalize(cross(Vec3(0.0f, 1.0f, 0.0f), f));
    if (length(r) < 1e-5f)
        r = Vec3(1.0f, 0.0f, 0.0f);

    Vec3 u = normalize(cross(f, r));

    Mat4 v = identity();

    // Row 0: right
    v.m[0] = r.x;
    v.m[4] = r.y;
    v.m[8] = r.z;
    v.m[12] = -dot(r, eye);

    // Row 1: up
    v.m[1] = u.x;
    v.m[5] = u.y;
    v.m[9] = u.z;
    v.m[13] = -dot(u, eye);

    // Row 2: -forward
    v.m[2] = -f.x;
    v.m[6] = -f.y;
    v.m[10] = -f.z;
    v.m[14] = dot(f, eye);

    v.m[15] = 1.0f;
    return v;
}

struct CubeVertex
{
    Vec3 pos;
    Vec3 normal;
};

struct Rect
{
    int x = 0;
    int y = 0;
    int w = 1;
    int h = 1;
};

struct AppState
{
    GLFWwindow* window = nullptr;

    bool preview = true;
    bool won = false;

    std::vector<uint8_t> gray; // 0 = black, 255 = white after thresholding
    int mazeW = 0;
    int mazeH = 0;
    bool invert = false;

    float cellSize = 1.0f;
    float wallHeight = 3.0f;

    Vec2 start{0.0f, 0.0f};
    Vec2 finish{0.0f, 0.0f};
    bool hasStart = false;
    bool hasFinish = false;

    Vec2 hover{-1.0f, -1.0f};

    Vec3 pos{0.0f, 1.7f, 0.0f};
    float yaw = 90.0f;     // degrees, 90 = +Z
    float pitch = 0.0f;    // degrees
    float sensitivity = 0.12f;
    float moveSpeed = 3.2f;

    bool keys[512] = {};
    double lastX = 0.0;
    double lastY = 0.0;

    GLuint progWall = 0;
    GLuint progObj = 0;
    GLuint progPreview = 0;

    GLuint vaoPlain = 0;
    GLuint vaoInstanced = 0;
    GLuint vaoQuad = 0;

    GLuint vboCube = 0;
    GLuint vboInstance = 0;
    GLuint vboQuad = 0;

    GLuint textureMaze = 0;

    std::vector<Vec2> wallCells;
    int instanceCount = 0;
    int cubeVertexCount = 36;
};

static AppState app;

static const char* wallVS = R"glsl(
#version 410 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aCell;

uniform mat4 uView;
uniform mat4 uProj;
uniform float uCellSize;
uniform float uWallHeight;

out vec3 vNormal;
out vec3 vWorldPos;

void main()
{
    vec3 s = vec3(uCellSize, uWallHeight, uCellSize);
    vec3 offset = vec3(
        (aCell.x + 0.5) * uCellSize,
        uWallHeight * 0.5,
        (aCell.y + 0.5) * uCellSize
    );

    vec3 world = offset + aPos * s;
    vWorldPos = world;
    vNormal = aNormal;

    gl_Position = uProj * uView * vec4(world, 1.0);
}
)glsl";

static const char* wallFS = R"glsl(
#version 410 core

in vec3 vNormal;
in vec3 vWorldPos;

out vec4 fragColor;

uniform vec3 uCameraPos;
uniform vec3 uFogColor;

void main()
{
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.35, 0.85, 0.25));
    float diff = max(dot(n, lightDir), 0.0);
    float shade = 0.35 + 0.65 * diff;

    vec3 base = vec3(0.54, 0.50, 0.45);
    vec3 color = base * shade;

    float dist = length(vWorldPos - uCameraPos);
    float fog = clamp(dist / 90.0, 0.0, 1.0);
    color = mix(color, uFogColor, fog * 0.65);

    fragColor = vec4(color, 1.0);
}
)glsl";

static const char* objectVS = R"glsl(
#version 410 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vWorldPos;

void main()
{
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(uModel) * aNormal);
    gl_Position = uProj * uView * world;
}
)glsl";

static const char* objectFS = R"glsl(
#version 410 core

in vec3 vNormal;
in vec3 vWorldPos;

out vec4 fragColor;

uniform vec3 uBaseColor;
uniform float uAlpha;
uniform vec3 uCameraPos;
uniform vec3 uFogColor;

void main()
{
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.35, 0.85, 0.25));
    float diff = max(dot(n, lightDir), 0.0);
    float shade = 0.38 + 0.62 * diff;

    vec3 color = uBaseColor * shade;

    float dist = length(vWorldPos - uCameraPos);
    float fog = clamp(dist / 100.0, 0.0, 1.0);
    color = mix(color, uFogColor, fog * 0.55);

    fragColor = vec4(color, uAlpha);
}
)glsl";

static const char* previewVS = R"glsl(
#version 410 core

layout(location = 0) in vec2 aPos;

uniform mat4 uProj;
uniform vec2 uMazeSize;

out vec2 vUV;

void main()
{
    vUV = aPos;
    vec2 world = aPos * uMazeSize;
    gl_Position = uProj * vec4(world, 0.0, 1.0);
}
)glsl";

static const char* previewFS = R"glsl(
#version 410 core

in vec2 vUV;

out vec4 fragColor;

uniform sampler2D uMaze;
uniform vec2 uMazeSize;
uniform bool uInvert;

uniform vec2 uStart;
uniform vec2 uFinish;
uniform bool uHasStart;
uniform bool uHasFinish;
uniform vec2 uHover;

void main()
{
    vec2 cell = vUV * uMazeSize;
    float g = texture(uMaze, vUV).r;

    bool wall = (g < 0.5);
    if (uInvert)
        wall = !wall;

    vec3 col = wall ? vec3(0.10, 0.10, 0.11) : vec3(0.88, 0.88, 0.86);

    // Grid lines.
    vec2 f = fract(cell);
    if (f.x < 0.035 || f.y < 0.035 || f.x > 0.965 || f.y > 0.965)
        col *= 0.72;

    // Finish marker.
    if (uHasFinish)
    {
        vec2 fc = uFinish + vec2(0.5);
        if (length(cell - fc) < 0.45)
            col = vec3(0.10, 0.90, 0.25);
    }

    // Start marker.
    if (uHasStart)
    {
        vec2 sc = uStart + vec2(0.5);
        if (length(cell - sc) < 0.45)
            col = vec3(0.15, 0.45, 1.00);
    }

    // Hover highlight.
    if (uHover.x >= 0.0 && uHover.y >= 0.0)
    {
        vec2 hc = uHover + vec2(0.5);
if (abs(cell.x - hc.x) < 0.5 && abs(cell.y - hc.y) < 0.5)
            col = mix(col, vec3(1.0, 0.92, 0.20), 0.28);
    }

    fragColor = vec4(col, 1.0);
}
)glsl";

static GLuint compileShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<size_t>(std::max(len, 1)));
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log.data() << "\n";
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

static GLuint createProgram(const char* vsSource, const char* fsSource)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource);
    if (!vs || !fs)
        return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<size_t>(std::max(len, 1)));
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log.data() << "\n";
        glDeleteProgram(prog);
        prog = 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static void setUniformMatrix4(GLuint prog, const char* name, const Mat4& m)
{
    GLint loc = glGetUniformLocation(prog, name);
    if (loc != -1)
        glUniformMatrix4fv(loc, 1, GL_FALSE, m.m);
}

static void setUniformVec2(GLuint prog, const char* name, const Vec2& v)
{
    GLint loc = glGetUniformLocation(prog, name);
    if (loc != -1)
        glUniform2f(loc, v.x, v.y);
}

static void setUniformVec3(GLuint prog, const char* name, const Vec3& v)
{
    GLint loc = glGetUniformLocation(prog, name);
    if (loc != -1)
        glUniform3f(loc, v.x, v.y, v.z);
}

static void setUniformFloat(GLuint prog, const char* name, float v)
{
    GLint loc = glGetUniformLocation(prog, name);
    if (loc != -1)
        glUniform1f(loc, v);
}

static void setUniformInt(GLuint prog, const char* name, int v)
{
    GLint loc = glGetUniformLocation(prog, name);
    if (loc != -1)
        glUniform1i(loc, v);
}

static bool loadPNGGray(const std::string& filename, int& outW, int& outH, std::vector<uint8_t>& outGray)
{
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp)
        return false;

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png)
    {
        std::fclose(fp);
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_read_struct(&png, nullptr, nullptr);
        std::fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return false;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = static_cast<int>(png_get_image_width(png, info));
    int height = static_cast<int>(png_get_image_height(png, info));
    if (width <= 0 || height <= 0)
    {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return false;
    }

    png_byte colorType = png_get_color_type(png, info);
    png_byte bitDepth = png_get_bit_depth(png, info);

    if (bitDepth == 16)
        png_set_strip_16(png);

    if (colorType == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    int rowBytes = static_cast<int>(png_get_rowbytes(png, info));
    int channels = static_cast<int>(png_get_channels(png, info));

    std::vector<std::vector<png_byte>> rows(height, std::vector<png_byte>(rowBytes));
    std::vector<png_bytep> rowPtrs(height);
    for (int y = 0; y < height; ++y)
        rowPtrs[y] = rows[y].data();

    png_read_image(png, rowPtrs.data());

    outW = width;
    outH = height;
    outGray.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 255);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            unsigned int r = 0, g = 0, b = 0;
            size_t off = static_cast<size_t>(y) * static_cast<size_t>(rowBytes)
                       + static_cast<size_t>(x) * static_cast<size_t>(channels);

            if (channels >= 3)
            {
                r = rows[y][off + 0];
                g = rows[y][off + 1];
                b = rows[y][off + 2];
            }
            else if (channels == 1)
            {
                r = g = b = rows[y][off];
            }
            else
            {
                r = g = b = rows[y][off];
            }

            unsigned int lum = (r * 299u + g * 587u + b * 114u) / 1000u;
            outGray[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                (lum < 128u) ? 0 : 255;
        }
    }

    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    return true;
}

static bool insideCell(int x, int y)
{
    return x >= 0 && y >= 0 && x < app.mazeW && y < app.mazeH;
}

static bool isWallCell(int x, int y)
{
    if (!insideCell(x, y))
        return true;

    uint8_t g = app.gray[static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x)];
    bool wall = (g < 128);
    if (app.invert)
        wall = !wall;
    return wall;
}

static void uploadMazeTexture()
{
    if (app.mazeW <= 0 || app.mazeH <= 0 || app.gray.empty())
        return;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app.textureMaze);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R8,
        app.mazeW,
        app.mazeH,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        app.gray.data()
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

static void rebuildWallInstances()
{
    app.wallCells.clear();
    app.wallCells.reserve(static_cast<size_t>(app.mazeW) * static_cast<size_t>(app.mazeH) / 4u + 1u);

    for (int y = 0; y < app.mazeH; ++y)
    {
        for (int x = 0; x < app.mazeW; ++x)
        {
            if (isWallCell(x, y))
                app.wallCells.push_back(Vec2(static_cast<float>(x), static_cast<float>(y)));
        }
    }

    app.instanceCount = static_cast<int>(app.wallCells.size());

    glBindBuffer(GL_ARRAY_BUFFER, app.vboInstance);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(app.wallCells.size() * sizeof(Vec2)),
        app.wallCells.empty() ? nullptr : app.wallCells.data(),
        GL_DYNAMIC_DRAW
    );
}

static void makeCellOpen(int x, int y)
{
    if (!insideCell(x, y))
        return;

    size_t idx = static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x);

    // Make this cell open under current invert setting.
    app.gray[idx] = app.invert ? 0 : 255;

    uploadMazeTexture();
    rebuildWallInstances();
}

static void findDefaultStartFinish()
{
    app.hasStart = false;
    app.hasFinish = false;

    bool found = false;
    for (int y = 0; y < app.mazeH && !found; ++y)
    {
        for (int x = 0; x < app.mazeW; ++x)
        {
            if (!isWallCell(x, y))
            {
                app.start = Vec2(static_cast<float>(x), static_cast<float>(y));
                app.hasStart = true;
                found = true;
                break;
            }
        }
    }

    if (!app.hasStart)
    {
        int cx = app.mazeW / 2;
        int cy = app.mazeH / 2;
        makeCellOpen(cx, cy);
        app.start = Vec2(static_cast<float>(cx), static_cast<float>(cy));
        app.hasStart = true;
    }

    found = false;
    for (int y = app.mazeH - 1; y >= 0 && !found; --y)
    {
        for (int x = app.mazeW - 1; x >= 0; --x)
        {
            if (!isWallCell(x, y))
            {
                app.finish = Vec2(static_cast<float>(x), static_cast<float>(y));
                app.hasFinish = true;
                found = true;
                break;
            }
        }
    }

    if (!app.hasFinish)
    {
        int cx = std::max(0, app.mazeW - 2);
        int cy = std::max(0, app.mazeH - 2);
        makeCellOpen(cx, cy);
        app.finish = Vec2(static_cast<float>(cx), static_cast<float>(cy));
        app.hasFinish = true;
    }
}

static void ensureValidMarkers()
{
    if (app.hasStart && isWallCell(static_cast<int>(app.start.x), static_cast<int>(app.start.y)))
        app.hasStart = false;

    if (app.hasFinish && isWallCell(static_cast<int>(app.finish.x), static_cast<int>(app.finish.y)))
        app.hasFinish = false;

    if (!app.hasStart || !app.hasFinish)
        findDefaultStartFinish();
}

static bool loadMazeFromFile(const std::string& path)
{
    int w = 0, h = 0;
    std::vector<uint8_t> g;

    if (!loadPNGGray(path, w, h, g))
        return false;

    app.mazeW = w;
    app.mazeH = h;
    app.gray = std::move(g);

    uploadMazeTexture();
    rebuildWallInstances();
    findDefaultStartFinish();
    return true;
}

static void generateDefaultMaze()
{
    app.mazeW = 21;
    app.mazeH = 11;
    app.gray.assign(static_cast<size_t>(app.mazeW) * static_cast<size_t>(app.mazeH), 255);

    auto setWall = [&](int x, int y)
    {
        if (insideCell(x, y))
            app.gray[static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x)] = 0;
    };

    auto setOpen = [&](int x, int y)
    {
        if (insideCell(x, y))
            app.gray[static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x)] = 255;
    };

    // Border.
    for (int x = 0; x < app.mazeW; ++x)
    {
        setWall(x, 0);
        setWall(x, app.mazeH - 1);
    }
    for (int y = 0; y < app.mazeH; ++y)
    {
        setWall(0, y);
        setWall(app.mazeW - 1, y);
    }

    // Some interior walls with gaps.
    for (int y = 1; y <= 8; ++y) setWall(5, y);
    setOpen(5, 4);

    for (int y = 2; y <= 9; ++y) setWall(10, y);
    setOpen(10, 6);

    for (int y = 1; y <= 8; ++y) setWall(15, y);
    setOpen(15, 3);

    for (int x = 6; x <= 9; ++x) setWall(x, 3);
    for (int x = 11; x <= 14; ++x) setWall(x, 7);
}

static std::vector<CubeVertex> makeCubeVertices()
{
    std::vector<CubeVertex> v;
    v.reserve(36);

    auto push = [&](const Vec3& p, const Vec3& n)
    {
        v.push_back(CubeVertex{p, n});
    };

    auto face = [&](const Vec3& n, const Vec3& u, const Vec3& vv)
    {
        Vec3 c = n * 0.5f;
        Vec3 p0 = c - u * 0.5f - vv * 0.5f;
        Vec3 p1 = c + u * 0.5f - vv * 0.5f;
        Vec3 p2 = c + u * 0.5f + vv * 0.5f;
        Vec3 p3 = c - u * 0.5f + vv * 0.5f;

        push(p0, n);
        push(p1, n);
        push(p2, n);

        push(p0, n);
        push(p2, n);
        push(p3, n);
    };

    face(Vec3(0.0f, 0.0f, 1.0f), Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    face(Vec3(0.0f, 0.0f, -1.0f), Vec3(-1.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    face(Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f), Vec3(0.0f, 1.0f, 0.0f));
    face(Vec3(-1.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 1.0f, 0.0f));
    face(Vec3(0.0f, 1.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -1.0f));
    face(Vec3(0.0f, -1.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));

    return v;
}

static void createGeometry()
{
    std::vector<CubeVertex> cube = makeCubeVertices();
    app.cubeVertexCount = static_cast<int>(cube.size());

    glGenBuffers(1, &app.vboCube);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboCube);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(cube.size() * sizeof(CubeVertex)), cube.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &app.vboInstance);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboInstance);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    // Plain cube VAO for floor/markers.
    glGenVertexArrays(1, &app.vaoPlain);
    glBindVertexArray(app.vaoPlain);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboCube);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CubeVertex), reinterpret_cast<void*>(offsetof(CubeVertex, pos)));
    glVertexAttribDivisor(0, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CubeVertex), reinterpret_cast<void*>(offsetof(CubeVertex, normal)));
    glVertexAttribDivisor(1, 0);

    // Instanced cube VAO for walls.
    glGenVertexArrays(1, &app.vaoInstanced);
    glBindVertexArray(app.vaoInstanced);

    glBindBuffer(GL_ARRAY_BUFFER, app.vboCube);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CubeVertex), reinterpret_cast<void*>(offsetof(CubeVertex, pos)));
    glVertexAttribDivisor(0, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CubeVertex), reinterpret_cast<void*>(offsetof(CubeVertex, normal)));
    glVertexAttribDivisor(1, 0);

    glBindBuffer(GL_ARRAY_BUFFER, app.vboInstance);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), nullptr);
    glVertexAttribDivisor(2, 1);

    // Preview quad VAO. Unit quad in [0,1]^2.
    static const float quad[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
    };

    glGenBuffers(1, &app.vboQuad);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboQuad);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    glGenVertexArrays(1, &app.vaoQuad);
    glBindVertexArray(app.vaoQuad);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboQuad);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glVertexAttribDivisor(0, 0);

    glBindVertexArray(0);
}

static bool createShaders()
{
    app.progWall = createProgram(wallVS, wallFS);
    app.progObj = createProgram(objectVS, objectFS);
    app.progPreview = createProgram(previewVS, previewFS);

    return app.progWall && app.progObj && app.progPreview;
}

static Rect getPreviewRectWindow()
{
    int ww = 1, wh = 1;
    if (app.window)
        glfwGetWindowSize(app.window, &ww, &wh);

    Rect r;
    r.x = 0;
    r.y = 0;
    r.w = ww;
    r.h = wh;

    if (app.mazeW <= 0 || app.mazeH <= 0 || ww <= 0 || wh <= 0)
        return r;

    float mazeAspect = static_cast<float>(app.mazeW) / static_cast<float>(app.mazeH);
    float winAspect = static_cast<float>(ww) / static_cast<float>(wh);

    if (winAspect > mazeAspect)
    {
        r.h = wh;
        r.w = static_cast<int>(std::round(static_cast<float>(wh) * mazeAspect));
        r.x = (ww - r.w) / 2;
        r.y = 0;
    }
    else
    {
        r.w = ww;
        r.h = static_cast<int>(std::round(static_cast<float>(ww) / mazeAspect));
        r.x = 0;
        r.y = (wh - r.h) / 2;
    }

    if (r.w < 1) r.w = 1;
    if (r.h < 1) r.h = 1;
    return r;
}

static Rect windowRectToFramebuffer(const Rect& wr)
{
    int ww = 1, wh = 1;
    int fw = 1, fh = 1;

    if (app.window)
    {
        glfwGetWindowSize(app.window, &ww, &wh);
        glfwGetFramebufferSize(app.window, &fw, &fh);
    }

    if (ww <= 0 || wh <= 0 || fw <= 0 || fh <= 0)
        return Rect{0, 0, std::max(1, fw), std::max(1, fh)};

    float sx = static_cast<float>(fw) / static_cast<float>(ww);
    float sy = static_cast<float>(fh) / static_cast<float>(wh);

    Rect fr;
    fr.x = static_cast<int>(std::round(static_cast<float>(wr.x) * sx));
    fr.y = static_cast<int>(std::round(static_cast<float>(fh) - (static_cast<float>(wr.y) + static_cast<float>(wr.h)) * sy));
    fr.w = static_cast<int>(std::round(static_cast<float>(wr.w) * sx));
    fr.h = static_cast<int>(std::round(static_cast<float>(wr.h) * sy));

    if (fr.w < 1) fr.w = 1;
    if (fr.h < 1) fr.h = 1;

    return fr;
}

static void updateHover(double mouseX, double mouseY)
{
    Rect r = getPreviewRectWindow();

    if (mouseX < r.x || mouseX >= r.x + r.w || mouseY < r.y || mouseY >= r.y + r.h)
    {
        app.hover = Vec2(-1.0f, -1.0f);
        return;
    }

    float fx = static_cast<float>(mouseX - r.x) / static_cast<float>(r.w);
    float fy = static_cast<float>(mouseY - r.y) / static_cast<float>(r.h);

    int cx = static_cast<int>(std::floor(fx * static_cast<float>(app.mazeW)));
    int cy = static_cast<int>(std::floor(fy * static_cast<float>(app.mazeH)));

    cx = std::clamp(cx, 0, app.mazeW - 1);
    cy = std::clamp(cy, 0, app.mazeH - 1);

    app.hover = Vec2(static_cast<float>(cx), static_cast<float>(cy));
}

static float eyeHeight()
{
    float h = app.wallHeight - 0.25f;
    h = std::clamp(h, 0.5f, 1.7f);
    return h;
}

static Vec3 cameraFront()
{
    float yr = radians(app.yaw);
    float pr = radians(app.pitch);
    return Vec3(
        std::cos(yr) * std::cos(pr),
        std::sin(pr),
        std::sin(yr) * std::cos(pr)
    );
}

static void resetPlayer()
{
    float e = eyeHeight();

    if (app.hasStart)
    {
        app.pos = Vec3(
            (app.start.x + 0.5f) * app.cellSize,
            e,
            (app.start.y + 0.5f) * app.cellSize
        );
    }
    else
    {
        app.pos = Vec3(0.5f * app.cellSize, e, 0.5f * app.cellSize);
    }

    app.yaw = 90.0f;
    app.pitch = 0.0f;
    app.won = false;
}

static void enterGame()
{
    if (app.mazeW <= 0 || app.mazeH <= 0)
        return;

    ensureValidMarkers();
    resetPlayer();

    app.preview = false;
    glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    int ww = 1, wh = 1;
    glfwGetWindowSize(app.window, &ww, &wh);
    double cx = ww * 0.5;
    double cy = wh * 0.5;
    glfwSetCursorPos(app.window, cx, cy);
    app.lastX = cx;
    app.lastY = cy;
}

static void enterPreview()
{
    app.preview = true;
    app.won = false;
    glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    double x = 0.0, y = 0.0;
    glfwGetCursorPos(app.window, &x, &y);
    updateHover(x, y);
}

static bool canStand(float x, float z)
{
    if (app.cellSize <= 0.001f)
        return false;

    float r = app.cellSize * 0.30f;
    if (r < 0.01f) r = 0.01f;

    float offs[3] = {-r, 0.0f, r};

    for (float ox : offs)
    {
        for (float oz : offs)
        {
            float px = x + ox;
            float pz = z + oz;

            int cx = static_cast<int>(std::floor(px / app.cellSize));
            int cy = static_cast<int>(std::floor(pz / app.cellSize));

            if (isWallCell(cx, cy))
                return false;
        }
    }

    return true;
}

static void update(float dt)
{
    if (app.preview)
        return;

    if (dt > 0.1f) dt = 0.1f;

    app.pos.y = eyeHeight();

    if (app.won)
        return;

    float yr = radians(app.yaw);
    Vec3 hf(std::cos(yr), 0.0f, std::sin(yr));
    Vec3 hr(std::sin(yr), 0.0f, -std::cos(yr));

    Vec3 move(0.0f, 0.0f, 0.0f);

    if (app.keys[GLFW_KEY_W] || app.keys[GLFW_KEY_UP])
        move = move + hf;
    if (app.keys[GLFW_KEY_S] || app.keys[GLFW_KEY_DOWN])
        move = move - hf;
    if (app.keys[GLFW_KEY_D] || app.keys[GLFW_KEY_RIGHT])
        move = move + hr;
    if (app.keys[GLFW_KEY_A] || app.keys[GLFW_KEY_LEFT])
        move = move - hr;

    float len = length(move);
    if (len > 1e-5f)
    {
        move = move / len;

        bool sprint = app.keys[GLFW_KEY_LEFT_SHIFT] || app.keys[GLFW_KEY_RIGHT_SHIFT];
        float speed = app.moveSpeed * (sprint ? 2.0f : 1.0f);
        move = move * (speed * dt);

        Vec3 np = app.pos;
        np.x += move.x;
        if (canStand(np.x, app.pos.z))
            app.pos.x = np.x;

        np = app.pos;
        np.z += move.z;
        if (canStand(app.pos.x, np.z))
            app.pos.z = np.z;
    }

    if (app.hasFinish)
    {
        Vec3 fp(
            (app.finish.x + 0.5f) * app.cellSize,
            0.0f,
            (app.finish.y + 0.5f) * app.cellSize
        );

        float dx = app.pos.x - fp.x;
        float dz = app.pos.z - fp.z;
        float dist = std::sqrt(dx * dx + dz * dz);

        if (dist < app.cellSize * 0.45f)
        {
            if (!app.won)
            {
                app.won = true;
                std::cout << "You reached the finish! Press R to restart or Tab for preview.\n";
            }
        }
    }
}

static void drawObject(
    const Mat4& model,
    const Vec3& baseColor,
    float alpha,
    const Mat4& view,
    const Mat4& proj,
    const Vec3& cameraPos,
    const Vec3& fogColor)
{
    glUseProgram(app.progObj);

    setUniformMatrix4(app.progObj, "uModel", model);
    setUniformMatrix4(app.progObj, "uView", view);
    setUniformMatrix4(app.progObj, "uProj", proj);
    setUniformVec3(app.progObj, "uBaseColor", baseColor);
    setUniformFloat(app.progObj, "uAlpha", alpha);
    setUniformVec3(app.progObj, "uCameraPos", cameraPos);
    setUniformVec3(app.progObj, "uFogColor", fogColor);

    glBindVertexArray(app.vaoPlain);
    glDrawArrays(GL_TRIANGLES, 0, app.cubeVertexCount);
}

static void render3D(float time)
{
    int fw = 1, fh = 1;
    glfwGetFramebufferSize(app.window, &fw, &fh);
    glViewport(0, 0, fw, fh);

    Vec3 fogColor = app.won
        ? Vec3(0.04f, 0.10f, 0.05f)
        : Vec3(0.05f, 0.06f, 0.08f);

    glClearColor(fogColor.x, fogColor.y, fogColor.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (app.mazeW <= 0 || app.mazeH <= 0)
        return;

    float aspect = static_cast<float>(fw) / static_cast<float>(std::max(1, fh));
    Mat4 proj = perspectiveMatrix(radians(70.0f), aspect, 0.05f, 500.0f);
    Mat4 view = viewMatrix(app.pos, cameraFront());

    // Floor.
    float worldW = static_cast<float>(app.mazeW) * app.cellSize;
    float worldH = static_cast<float>(app.mazeH) * app.cellSize;

    Mat4 floorModel = multiply(
        translate(Vec3(worldW * 0.5f, -0.05f, worldH * 0.5f)),
        scale(Vec3(worldW, 0.1f, worldH))
    );

    drawObject(floorModel, Vec3(0.24f, 0.24f, 0.26f), 1.0f, view, proj, app.pos, fogColor);

    // Walls, instanced.
    glUseProgram(app.progWall);
    setUniformMatrix4(app.progWall, "uView", view);
    setUniformMatrix4(app.progWall, "uProj", proj);
    setUniformFloat(app.progWall, "uCellSize", app.cellSize);
    setUniformFloat(app.progWall, "uWallHeight", app.wallHeight);
    setUniformVec3(app.progWall, "uCameraPos", app.pos);
    setUniformVec3(app.progWall, "uFogColor", fogColor);

    glBindVertexArray(app.vaoInstanced);
    glDrawArraysInstanced(GL_TRIANGLES, 0, app.cubeVertexCount, app.instanceCount);

    // Transparent markers.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    if (app.hasFinish)
    {
        float pulse = 0.12f + 0.05f * std::sin(time * 4.0f);
        Vec3 center(
            (app.finish.x + 0.5f) * app.cellSize,
            pulse * 0.5f,
            (app.finish.y + 0.5f) * app.cellSize
        );

        Vec3 color = app.won
            ? Vec3(1.0f, 0.85f, 0.20f)
            : Vec3(0.10f, 0.90f, 0.25f);

        Mat4 m = multiply(translate(center), scale(Vec3(app.cellSize * 0.88f, pulse, app.cellSize * 0.88f)));
        drawObject(m, color, 0.55f, view, proj, app.pos, fogColor);
    }

    if (app.hasStart)
    {
        float h = 0.10f;
        Vec3 center(
            (app.start.x + 0.5f) * app.cellSize,
            h * 0.5f,
            (app.start.y + 0.5f) * app.cellSize
        );

        Mat4 m = multiply(translate(center), scale(Vec3(app.cellSize * 0.88f, h, app.cellSize * 0.88f)));
        drawObject(m, Vec3(0.15f, 0.45f, 1.00f), 0.45f, view, proj, app.pos, fogColor);
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

static void renderPreview()
{
    int fw = 1, fh = 1;
    glfwGetFramebufferSize(app.window, &fw, &fh);

    Vec3 bg(0.03f, 0.035f, 0.045f);
    glViewport(0, 0, fw, fh);
    glClearColor(bg.x, bg.y, bg.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (app.mazeW <= 0 || app.mazeH <= 0)
        return;

    Rect wr = getPreviewRectWindow();
    Rect fr = windowRectToFramebuffer(wr);

    glViewport(fr.x, fr.y, fr.w, fr.h);
    glDisable(GL_DEPTH_TEST);

    glUseProgram(app.progPreview);

    Mat4 ortho = orthoMatrix(
        0.0f,
        static_cast<float>(app.mazeW),
        static_cast<float>(app.mazeH),
        0.0f,
        -1.0f,
        1.0f
    );

    setUniformMatrix4(app.progPreview, "uProj", ortho);
    setUniformVec2(app.progPreview, "uMazeSize", Vec2(static_cast<float>(app.mazeW), static_cast<float>(app.mazeH)));
    setUniformInt(app.progPreview, "uInvert", app.invert ? 1 : 0);
    setUniformVec2(app.progPreview, "uStart", app.start);
    setUniformVec2(app.progPreview, "uFinish", app.finish);
    setUniformInt(app.progPreview, "uHasStart", app.hasStart ? 1 : 0);
    setUniformInt(app.progPreview, "uHasFinish", app.hasFinish ? 1 : 0);
    setUniformVec2(app.progPreview, "uHover", app.hover);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app.textureMaze);
    setUniformInt(app.progPreview, "uMaze", 0);

    glBindVertexArray(app.vaoQuad);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, fw, fh);
}

static void render(float time)
{
    if (app.preview)
        renderPreview();
    else
        render3D(time);
}

static void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key >= 0 && key < 512)
        app.keys[key] = (action != GLFW_RELEASE);

    if (action != GLFW_PRESS)
        return;

    if (key == GLFW_KEY_ESCAPE)
    {
        if (app.preview)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        else
            enterPreview();
        return;
    }

    if (app.preview)
    {
        if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER || key == GLFW_KEY_SPACE)
        {
            enterGame();
        }
        else if (key == GLFW_KEY_I)
        {
            app.invert = !app.invert;
            rebuildWallInstances();
            ensureValidMarkers();
        }
        else if (key == GLFW_KEY_R)
        {
            findDefaultStartFinish();
        }
        else if (key == GLFW_KEY_COMMA)
        {
            app.cellSize = std::max(0.1f, app.cellSize - 0.1f);
        }
        else if (key == GLFW_KEY_PERIOD)
        {
            app.cellSize = std::min(10.0f, app.cellSize + 0.1f);
        }
        else if (key == GLFW_KEY_LEFT_BRACKET)
        {
            app.wallHeight = std::max(0.5f, app.wallHeight - 0.25f);
        }
        else if (key == GLFW_KEY_RIGHT_BRACKET)
        {
            app.wallHeight = std::min(20.0f, app.wallHeight + 0.25f);
        }
    }
    else
    {
        if (key == GLFW_KEY_TAB)
        {
            enterPreview();
        }
        else if (key == GLFW_KEY_R)
        {
            resetPlayer();
        }
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos)
{
    if (app.preview)
    {
        updateHover(xpos, ypos);
    }
    else
    {
        double dx = xpos - app.lastX;
        double dy = ypos - app.lastY;
        app.lastX = xpos;
        app.lastY = ypos;

        app.yaw -= static_cast<float>(dx) * app.sensitivity;
        app.pitch -= static_cast<float>(dy) * app.sensitivity;
        app.pitch = std::clamp(app.pitch, -89.0f, 89.0f);
    }
}

static void mouseButtonCallback(GLFWwindow* /*window*/, int button, int action, int /*mods*/)
{
    if (action != GLFW_PRESS || !app.preview)
        return;

    int cx = static_cast<int>(app.hover.x);
    int cy = static_cast<int>(app.hover.y);

    if (cx < 0 || cy < 0 || !insideCell(cx, cy))
        return;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (!isWallCell(cx, cy))
        {
            app.start = Vec2(static_cast<float>(cx), static_cast<float>(cy));
            app.hasStart = true;
        }
        else
        {
            std::cout << "Start must be on an open/white cell.\n";
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (!isWallCell(cx, cy))
        {
            app.finish = Vec2(static_cast<float>(cx), static_cast<float>(cy));
            app.hasFinish = true;
        }
        else
        {
            std::cout << "Finish must be on an open/white cell.\n";
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    {
        findDefaultStartFinish();
    }
}

static void scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset)
{
    if (!app.preview)
        return;

    float delta = static_cast<float>(yoffset) * 0.1f;
    app.cellSize = std::clamp(app.cellSize + delta, 0.1f, 10.0f);
}

static void framebufferSizeCallback(GLFWwindow* /*window*/, int /*width*/, int /*height*/)
{
    // Viewport is set every frame.
}

static bool init()
{
    if (!glfwInit())
    {
        std::cerr << "glfwInit failed\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    app.window = glfwCreateWindow(1280, 720, "OpenGL 4.1 Labyrinth Maze", nullptr, nullptr);
    if (!app.window)
    {
        std::cerr << "glfwCreateWindow failed\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(app.window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK)
    {
        std::cerr << "glewInit failed: " << glewGetErrorString(glewErr) << "\n";
        return false;
    }

    // Some GLEW versions generate a harmless invalid enum after init in core profile.
    while (glGetError() != GL_NO_ERROR) {}

    if (!GLEW_VERSION_4_1)
    {
        std::cerr << "Warning: OpenGL 4.1 not reported by GLEW. Trying anyway.\n";
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearDepthf(1.0f);
    glDisable(GL_CULL_FACE);

    if (!createShaders())
        return false;

    createGeometry();

    glGenTextures(1, &app.textureMaze);
    glBindTexture(GL_TEXTURE_2D, app.textureMaze);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glfwSetKeyCallback(app.window, keyCallback);
    glfwSetCursorPosCallback(app.window, cursorPosCallback);
    glfwSetMouseButtonCallback(app.window, mouseButtonCallback);
    glfwSetScrollCallback(app.window, scrollCallback);
    glfwSetFramebufferSizeCallback(app.window, framebufferSizeCallback);

    glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    return true;
}

static void cleanup()
{
    if (app.progWall) glDeleteProgram(app.progWall);
    if (app.progObj) glDeleteProgram(app.progObj);
    if (app.progPreview) glDeleteProgram(app.progPreview);

    if (app.vaoPlain) glDeleteVertexArrays(1, &app.vaoPlain);
    if (app.vaoInstanced) glDeleteVertexArrays(1, &app.vaoInstanced);
    if (app.vaoQuad) glDeleteVertexArrays(1, &app.vaoQuad);

    if (app.vboCube) glDeleteBuffers(1, &app.vboCube);
    if (app.vboInstance) glDeleteBuffers(1, &app.vboInstance);
    if (app.vboQuad) glDeleteBuffers(1, &app.vboQuad);

    if (app.textureMaze) glDeleteTextures(1, &app.textureMaze);

    if (app.window)
        glfwDestroyWindow(app.window);

    glfwTerminate();
}

int main(int argc, char** argv)
{
    std::string mazePath;

    float cliCellSize = app.cellSize;
    float cliWallHeight = app.wallHeight;

    bool cliHasStart = false;
    bool cliHasFinish = false;
    Vec2 cliStart;
    Vec2 cliFinish;

    if (argc > 1)
        mazePath = argv[1];

    if (argc > 2)
        cliCellSize = std::max(0.1f, static_cast<float>(std::atof(argv[2])));

    if (argc > 3)
        cliWallHeight = std::max(0.5f, static_cast<float>(std::atof(argv[3])));

    if (argc >= 7)
    {
        cliStart = Vec2(static_cast<float>(std::atoi(argv[4])), static_cast<float>(std::atoi(argv[5])));
        cliHasStart = true;

        cliFinish = Vec2(static_cast<float>(std::atoi(argv[6])), static_cast<float>(std::atoi(argv[7])));
        cliHasFinish = true;
    }

    if (!init())
        return 1;

    bool loaded = false;
    if (!mazePath.empty())
    {
        loaded = loadMazeFromFile(mazePath);
        if (!loaded)
            std::cerr << "Failed to load maze PNG: " << mazePath << "\nUsing built-in default maze.\n";
    }

    if (!loaded)
    {
        generateDefaultMaze();
        uploadMazeTexture();
        rebuildWallInstances();
        findDefaultStartFinish();
    }

    app.cellSize = cliCellSize;
    app.wallHeight = cliWallHeight;

    if (cliHasStart)
    {
        int sx = static_cast<int>(cliStart.x);
        int sy = static_cast<int>(cliStart.y);
        if (insideCell(sx, sy) && !isWallCell(sx, sy))
        {
            app.start = cliStart;
            app.hasStart = true;
        }
    }

    if (cliHasFinish)
    {
        int fx = static_cast<int>(cliFinish.x);
        int fy = static_cast<int>(cliFinish.y);
        if (insideCell(fx, fy) && !isWallCell(fx, fy))
        {
            app.finish = cliFinish;
            app.hasFinish = true;
        }
    }

    ensureValidMarkers();

    app.preview = true;
    glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    double x = 0.0, y = 0.0;
    glfwGetCursorPos(app.window, &x, &y);
    updateHover(x, y);

    float lastTime = static_cast<float>(glfwGetTime());

    while (!glfwWindowShouldClose(app.window))
    {
        float now = static_cast<float>(glfwGetTime());
        float dt = now - lastTime;
        lastTime = now;

        update(dt);
        render(now);

        glfwSwapBuffers(app.window);
        glfwPollEvents();
    }

    cleanup();
    return 0;
}