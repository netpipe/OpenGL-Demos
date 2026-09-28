// maze_heightmap.cpp
//
// OpenGL 4.1 core heightmap labyrinth maze.
// Dependencies: GLFW, GLEW, libpng.
// No GLM.
 //./maze_heightmap my_circular_maze.png [cellScale] [heightScale] [walkLimit01]
 //./maze maze.png 1.0 6.0 0.35

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

static float clampf(float v, float lo, float hi)
{
    return std::clamp(v, lo, hi);
}

static float lerpf(float a, float b, float t)
{
    return a + (b - a) * t;
}

static float smoothstepf(float edge0, float edge1, float x)
{
    if (edge1 <= edge0)
        return (x < edge0) ? 0.0f : 1.0f;

    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
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

struct Vertex
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

    std::vector<uint8_t> height; // 0..255 grayscale heightmap
    int mazeW = 0;
    int mazeH = 0;

    bool invert = false;

    float cellSize = 1.0f;       // horizontal world units per heightmap texel
    float heightScale = 5.0f;    // vertical world units for height 1.0
    float walkLimit01 = 0.35f;   // normalized height <= this is considered walkable
    float maxClimb = 0.9f;       // max vertical step allowed in world units
    float eyeHeight = 1.7f;

    Vec2 start{0.0f, 0.0f};
    Vec2 finish{0.0f, 0.0f};
    bool hasStart = false;
    bool hasFinish = false;

    Vec2 hover{-1.0f, -1.0f};

    Vec3 pos{0.0f, 1.7f, 0.0f};
    float yaw = 90.0f;
    float pitch = 0.0f;
    float sensitivity = 0.12f;
    float moveSpeed = 3.2f;

    bool keys[512] = {};
    double lastX = 0.0;
    double lastY = 0.0;

    GLuint progWorld = 0;
    GLuint progPreview = 0;

    GLuint vaoTerrain = 0;
    GLuint vboTerrain = 0;
    GLuint eboTerrain = 0;

    GLuint vaoMarker = 0;
    GLuint vboMarker = 0;

    GLuint vaoPreview = 0;
    GLuint vboPreview = 0;

    GLuint textureHeight = 0;

    int indexCount = 0;
    int markerVertexCount = 36;
};

static AppState app;

static const char* worldVS = R"glsl(
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

static const char* worldFS = R"glsl(
#version 410 core

in vec3 vNormal;
in vec3 vWorldPos;

out vec4 fragColor;

uniform vec3 uCameraPos;
uniform vec3 uFogColor;

uniform float uHeightScale;
uniform float uWalkLimit01;

uniform int uUseOverride;
uniform vec3 uOverrideColor;
uniform float uAlpha;

void main()
{
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.35, 0.85, 0.25));
    float diff = max(dot(n, lightDir), 0.0);
    float shade = 0.35 + 0.65 * diff;

    vec3 color;

    if (uUseOverride != 0)
    {
        color = uOverrideColor * shade;
    }
    else
    {
        float h = clamp(vWorldPos.y / max(uHeightScale, 0.0001), 0.0, 1.0);

        vec3 floorCol = vec3(0.16, 0.38, 0.20);
        vec3 pathCol  = vec3(0.62, 0.55, 0.40);
        vec3 wallCol  = vec3(0.38, 0.36, 0.34);
        vec3 highCol  = vec3(0.92, 0.92, 0.95);

        color = mix(floorCol, pathCol, smoothstep(0.0, uWalkLimit01 * 0.85, h));
        color = mix(color, wallCol, smoothstep(uWalkLimit01 * 0.92, uWalkLimit01 + 0.10, h));
        color = mix(color, highCol, smoothstep(0.72, 1.0, h));

        float contour = 1.0 - smoothstep(0.0, 0.018, abs(h - uWalkLimit01));
        color = mix(color, vec3(1.0, 0.82, 0.25), contour * 0.55);

        color *= shade;
    }

    float dist = length(vWorldPos - uCameraPos);
    float fog = clamp(dist / 120.0, 0.0, 1.0);
    color = mix(color, uFogColor, fog * 0.65);

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

uniform sampler2D uHeight;
uniform vec2 uMazeSize;

uniform int uInvert;
uniform float uWalkLimit01;

uniform vec2 uStart;
uniform vec2 uFinish;
uniform int uHasStart;
uniform int uHasFinish;
uniform vec2 uHover;

void main()
{
    vec2 cell = vUV * uMazeSize;
    float raw = texture(uHeight, vUV).r;
    float h = (uInvert != 0) ? (1.0 - raw) : raw;

    vec3 floorCol = vec3(0.16, 0.38, 0.20);
    vec3 pathCol  = vec3(0.62, 0.55, 0.40);
    vec3 wallCol  = vec3(0.26, 0.25, 0.27);
    vec3 highCol  = vec3(0.94, 0.94, 0.96);

    vec3 col = mix(floorCol, pathCol, smoothstep(0.0, uWalkLimit01 * 0.85, h));
    col = mix(col, wallCol, smoothstep(uWalkLimit01 * 0.92, uWalkLimit01 + 0.10, h));
    col = mix(col, highCol, smoothstep(0.72, 1.0, h));

    float contour = 1.0 - smoothstep(0.0, 0.022, abs(h - uWalkLimit01));
    col = mix(col, vec3(1.0, 0.85, 0.20), contour * 0.75);

    // Faint cell grid.
    vec2 f = fract(cell);
    if (f.x < 0.02 || f.y < 0.02 || f.x > 0.98 || f.y > 0.98)
        col *= 0.92;

    // Finish marker.
    if (uHasFinish != 0)
    {
        vec2 fc = uFinish + vec2(0.5);
        if (length(cell - fc) < 0.45)
            col = vec3(0.10, 0.95, 0.25);
    }

    // Start marker.
    if (uHasStart != 0)
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
            col = mix(col, vec3(1.0, 0.92, 0.20), 0.25);
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

static bool loadPNGHeight(const std::string& filename, int& outW, int& outH, std::vector<uint8_t>& outHeight)
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
    outHeight.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 255);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            unsigned int r = 0, g = 0, b = 0, a = 255;
            size_t off = static_cast<size_t>(y) * static_cast<size_t>(rowBytes)
                       + static_cast<size_t>(x) * static_cast<size_t>(channels);

            if (channels >= 3)
            {
                r = rows[y][off + 0];
                g = rows[y][off + 1];
                b = rows[y][off + 2];

                if (channels == 4)
                    a = rows[y][off + 3];
            }
            else if (channels == 1)
            {
                r = g = b = rows[y][off];
            }
            else
            {
                r = g = b = rows[y][off];
            }

            unsigned int lum = 0;
            if (a < 128u)
            {
                lum = 0u;
            }
            else
            {
                lum = (r * 299u + g * 587u + b * 114u) / 1000u;
            }

            outHeight[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                static_cast<uint8_t>(lum);
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

static float rawHeight01Cell(int x, int y)
{
    if (!insideCell(x, y))
        return 1.0f;

    size_t idx = static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x);
    return static_cast<float>(app.height[idx]) / 255.0f;
}

static float effectiveHeight01Cell(int x, int y)
{
    float h = rawHeight01Cell(x, y);
    if (app.invert)
        h = 1.0f - h;
    return h;
}

static bool cellWalkable(int x, int y)
{
    if (!insideCell(x, y))
        return false;

    float h = effectiveHeight01Cell(x, y);
    return h <= app.walkLimit01 + 0.001f;
}

static void uploadHeightTexture()
{
    if (app.mazeW <= 0 || app.mazeH <= 0 || app.height.empty())
        return;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app.textureHeight);
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
        app.height.data()
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

static void rebuildTerrainMesh()
{
    if (app.mazeW <= 0 || app.mazeH <= 0 || app.height.empty())
    {
        app.indexCount = 0;
        return;
    }

    const int W = app.mazeW;
    const int H = app.mazeH;

    std::vector<Vertex> verts(static_cast<size_t>(W) * static_cast<size_t>(H));

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            size_t idx = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);

            float h = effectiveHeight01Cell(x, y);

            Vec3 p(
                (static_cast<float>(x) + 0.5f) * app.cellSize,
                h * app.heightScale,
                (static_cast<float>(y) + 0.5f) * app.cellSize
            );

            float hL = (x > 0) ? effectiveHeight01Cell(x - 1, y) : h;
            float hR = (x < W - 1) ? effectiveHeight01Cell(x + 1, y) : h;
            float hU = (y > 0) ? effectiveHeight01Cell(x, y - 1) : h;
            float hD = (y < H - 1) ? effectiveHeight01Cell(x, y + 1) : h;

            float dx = (hR - hL) * app.heightScale / (2.0f * app.cellSize);
            float dz = (hD - hU) * app.heightScale / (2.0f * app.cellSize);

            Vec3 n = normalize(Vec3(-dx, 1.0f, -dz));

            verts[idx].pos = p;
            verts[idx].normal = n;
        }
    }

    std::vector<uint32_t> indices;
    if (W >= 2 && H >= 2)
    {
        indices.reserve(static_cast<size_t>(W - 1) * static_cast<size_t>(H - 1) * 6u);

        for (int y = 0; y < H - 1; ++y)
        {
            for (int x = 0; x < W - 1; ++x)
            {
                uint32_t v00 = static_cast<uint32_t>(y * W + x);
                uint32_t v10 = v00 + 1u;
                uint32_t v01 = v00 + static_cast<uint32_t>(W);
                uint32_t v11 = v01 + 1u;

                indices.push_back(v00);
                indices.push_back(v10);
                indices.push_back(v11);

                indices.push_back(v00);
                indices.push_back(v11);
                indices.push_back(v01);
            }
        }
    }

    app.indexCount = static_cast<int>(indices.size());

    glBindBuffer(GL_ARRAY_BUFFER, app.vboTerrain);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)),
        verts.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, app.eboTerrain);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(indices.size() * sizeof(uint32_t)),
        indices.empty() ? nullptr : indices.data(),
        GL_STATIC_DRAW
    );
}

static void makeCellWalkable(int x, int y)
{
    if (!insideCell(x, y))
        return;

    size_t idx = static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x);

    // Effective height 0 means walkable floor.
    // If inverted, raw 255 becomes effective 0.
    app.height[idx] = app.invert ? 255 : 0;

    uploadHeightTexture();
    rebuildTerrainMesh();
}

static void findDefaultStartFinish()
{
    app.hasStart = false;
    app.hasFinish = false;

    // Find first walkable cell as start.
    for (int y = 0; y < app.mazeH && !app.hasStart; ++y)
    {
        for (int x = 0; x < app.mazeW; ++x)
        {
            if (cellWalkable(x, y))
            {
                app.start = Vec2(static_cast<float>(x), static_cast<float>(y));
                app.hasStart = true;
                break;
            }
        }
    }

    if (!app.hasStart)
    {
        int cx = app.mazeW / 2;
        int cy = app.mazeH / 2;
        makeCellWalkable(cx, cy);
        app.start = Vec2(static_cast<float>(cx), static_cast<float>(cy));
        app.hasStart = true;
    }

    // Find walkable cell farthest from start as finish.
    int bx = -1;
    int by = -1;
    float bestD = -1.0f;

    for (int y = 0; y < app.mazeH; ++y)
    {
        for (int x = 0; x < app.mazeW; ++x)
        {
            if (!cellWalkable(x, y))
                continue;

            float dx = static_cast<float>(x) - app.start.x;
            float dy = static_cast<float>(y) - app.start.y;
            float d = dx * dx + dy * dy;

            if (d > bestD)
            {
                bestD = d;
                bx = x;
                by = y;
            }
        }
    }

    if (bx >= 0 && by >= 0 && !(bx == static_cast<int>(app.start.x) && by == static_cast<int>(app.start.y)))
    {
        app.finish = Vec2(static_cast<float>(bx), static_cast<float>(by));
        app.hasFinish = true;
    }
    else
    {
        int fx = (app.start.x < app.mazeW * 0.5f) ? (app.mazeW - 2) : 1;
        int fy = (app.start.y < app.mazeH * 0.5f) ? (app.mazeH - 2) : 1;

        fx = clampf(static_cast<float>(fx), 0.0f, static_cast<float>(app.mazeW - 1));
        fy = clampf(static_cast<float>(fy), 0.0f, static_cast<float>(app.mazeH - 1));

        makeCellWalkable(fx, fy);
        app.finish = Vec2(static_cast<float>(fx), static_cast<float>(fy));
        app.hasFinish = true;
    }
}

static void ensureValidMarkers()
{
    if (app.hasStart && !cellWalkable(static_cast<int>(app.start.x), static_cast<int>(app.start.y)))
        app.hasStart = false;

    if (app.hasFinish && !cellWalkable(static_cast<int>(app.finish.x), static_cast<int>(app.finish.y)))
        app.hasFinish = false;

    if (!app.hasStart || !app.hasFinish)
        findDefaultStartFinish();
}

static void blurHeightmap()
{
    if (app.mazeW <= 0 || app.mazeH <= 0 || app.height.empty())
        return;

    std::vector<uint8_t> src = app.height;

    for (int y = 0; y < app.mazeH; ++y)
    {
        for (int x = 0; x < app.mazeW; ++x)
        {
            int sum = 0;
            int count = 0;

            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (!insideCell(nx, ny))
                        continue;

                    size_t idx = static_cast<size_t>(ny) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(nx);
                    sum += src[idx];
                    ++count;
                }
            }

            size_t idx = static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x);
            app.height[idx] = static_cast<uint8_t>(sum / std::max(1, count));
        }
    }

    uploadHeightTexture();
    rebuildTerrainMesh();
    ensureValidMarkers();
}

static bool loadMazeFromFile(const std::string& path)
{
    int w = 0, h = 0;
    std::vector<uint8_t> hm;

    if (!loadPNGHeight(path, w, h, hm))
        return false;

    app.mazeW = w;
    app.mazeH = h;
    app.height = std::move(hm);

    uploadHeightTexture();
    rebuildTerrainMesh();
    findDefaultStartFinish();
    return true;
}

static void generateDefaultCircularHeightmap()
{
    app.mazeW = 161;
    app.mazeH = 161;
    app.height.assign(static_cast<size_t>(app.mazeW) * static_cast<size_t>(app.mazeH), 0);

    const float cx = static_cast<float>(app.mazeW - 1) * 0.5f;
    const float cy = static_cast<float>(app.mazeH - 1) * 0.5f;
    const float maxR = 78.0f;

    for (int y = 0; y < app.mazeH; ++y)
    {
        for (int x = 0; x < app.mazeW; ++x)
        {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float r = std::sqrt(dx * dx + dy * dy);
            float a = std::atan2(dy, dx);

            float h = 0.08f;

            if (r > maxR)
            {
                h = 1.0f;
            }
            else
            {
                // Concentric circular walls with alternating gaps.
                for (int ring = 0; ring < 5; ++ring)
                {
                    float ringR = 18.0f + static_cast<float>(ring) * 15.0f;
                    float d = std::fabs(r - ringR);

                    float wall = 1.0f - smoothstepf(0.0f, 2.8f, d);

                    float gapCenter = (ring % 2 == 0) ? 0.0f : kPi;
                    float da = std::atan2(std::sin(a - gapCenter), std::cos(a - gapCenter));
                    float gap = smoothstepf(0.35f, 0.85f, std::fabs(da));

                    wall *= gap;
                    h = std::max(h, wall);
                }

                // Soft outer boundary.
                float outer = smoothstepf(maxR - 4.0f, maxR, r);
                h = std::max(h, outer);
            }

            h = clampf(h, 0.0f, 1.0f);

            size_t idx = static_cast<size_t>(y) * static_cast<size_t>(app.mazeW) + static_cast<size_t>(x);
            app.height[idx] = static_cast<uint8_t>(h * 255.0f + 0.5f);
        }
    }

    uploadHeightTexture();
    rebuildTerrainMesh();
    findDefaultStartFinish();
}

static std::vector<Vertex> makeCubeVertices()
{
    std::vector<Vertex> v;
    v.reserve(36);

    auto push = [&](const Vec3& p, const Vec3& n)
    {
        v.push_back(Vertex{p, n});
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
    // Terrain buffers.
    glGenBuffers(1, &app.vboTerrain);
    glGenBuffers(1, &app.eboTerrain);

    glGenVertexArrays(1, &app.vaoTerrain);
    glBindVertexArray(app.vaoTerrain);

    glBindBuffer(GL_ARRAY_BUFFER, app.vboTerrain);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, pos)));
    glVertexAttribDivisor(0, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glVertexAttribDivisor(1, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, app.eboTerrain);

    // Marker cube buffers.
    std::vector<Vertex> cube = makeCubeVertices();
    app.markerVertexCount = static_cast<int>(cube.size());

    glGenBuffers(1, &app.vboMarker);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboMarker);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(cube.size() * sizeof(Vertex)), cube.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &app.vaoMarker);
    glBindVertexArray(app.vaoMarker);

    glBindBuffer(GL_ARRAY_BUFFER, app.vboMarker);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, pos)));
    glVertexAttribDivisor(0, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glVertexAttribDivisor(1, 0);

    // Preview quad VAO. Unit quad in [0,1]^2.
    static const float quad[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
    };

    glGenBuffers(1, &app.vboPreview);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboPreview);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    glGenVertexArrays(1, &app.vaoPreview);
    glBindVertexArray(app.vaoPreview);
    glBindBuffer(GL_ARRAY_BUFFER, app.vboPreview);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glVertexAttribDivisor(0, 0);

    glBindVertexArray(0);
}

static bool createShaders()
{
    app.progWorld = createProgram(worldVS, worldFS);
    app.progPreview = createProgram(previewVS, previewFS);

    return app.progWorld && app.progPreview;
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

static float sampleHeight01(float x, float z)
{
    if (app.mazeW <= 0 || app.mazeH <= 0 || app.height.empty())
        return 1.0f;

    float u = x / app.cellSize - 0.5f;
    float v = z / app.cellSize - 0.5f;

    u = clampf(u, 0.0f, static_cast<float>(app.mazeW - 1));
    v = clampf(v, 0.0f, static_cast<float>(app.mazeH - 1));

    int x0 = static_cast<int>(std::floor(u));
    int y0 = static_cast<int>(std::floor(v));

    int x1 = std::min(x0 + 1, app.mazeW - 1);
    int y1 = std::min(y0 + 1, app.mazeH - 1);

    float fx = u - static_cast<float>(x0);
    float fy = v - static_cast<float>(y0);

    auto he = [&](int ix, int iy) -> float
    {
        return effectiveHeight01Cell(ix, iy);
    };

    float h00 = he(x0, y0);
    float h10 = he(x1, y0);
    float h01 = he(x0, y1);
    float h11 = he(x1, y1);

    float a = lerpf(h00, h10, fx);
    float b = lerpf(h01, h11, fx);

    return lerpf(a, b, fy);
}

static float sampleWorldHeight(float x, float z)
{
    return sampleHeight01(x, z) * app.heightScale;
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
    if (app.hasStart)
    {
        app.pos.x = (app.start.x + 0.5f) * app.cellSize;
        app.pos.z = (app.start.y + 0.5f) * app.cellSize;
    }
    else
    {
        app.pos.x = 0.5f * app.cellSize;
        app.pos.z = 0.5f * app.cellSize;
    }

    app.pos.y = sampleWorldHeight(app.pos.x, app.pos.z) + app.eyeHeight;
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

static bool canStandAt(float x, float z, float currentGroundH)
{
    if (app.mazeW <= 0 || app.mazeH <= 0)
        return false;

    float worldW = static_cast<float>(app.mazeW) * app.cellSize;
    float worldH = static_cast<float>(app.mazeH) * app.cellSize;

    float radius = std::min(app.cellSize * 0.28f, 0.45f);
    if (radius < 0.01f) radius = 0.01f;

    float margin = radius + app.cellSize * 0.15f;

    if (x < margin || z < margin || x > worldW - margin || z > worldH - margin)
        return false;

    const float offs[5][2] = {
        {0.0f, 0.0f},
        {radius, 0.0f},
        {-radius, 0.0f},
        {0.0f, radius},
        {0.0f, -radius}
    };

    for (int i = 0; i < 5; ++i)
    {
        float px = x + offs[i][0];
        float pz = z + offs[i][1];

        float h01 = sampleHeight01(px, pz);
        if (h01 > app.walkLimit01 + 0.002f)
            return false;

        float wh = h01 * app.heightScale;
        if (std::fabs(wh - currentGroundH) > app.maxClimb)
            return false;
    }

    return true;
}

static void update(float dt)
{
    if (app.preview)
        return;

    if (dt > 0.1f) dt = 0.1f;

    float ground = sampleWorldHeight(app.pos.x, app.pos.z);
    app.pos.y = ground + app.eyeHeight;

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

        float nx = app.pos.x + move.x;
        if (canStandAt(nx, app.pos.z, ground))
            app.pos.x = nx;

        float ground2 = sampleWorldHeight(app.pos.x, app.pos.z);
        float nz = app.pos.z + move.z;
        if (canStandAt(app.pos.x, nz, ground2))
            app.pos.z = nz;

        app.pos.y = sampleWorldHeight(app.pos.x, app.pos.z) + app.eyeHeight;
    }

    if (app.hasFinish)
    {
        float fx = (app.finish.x + 0.5f) * app.cellSize;
        float fz = (app.finish.y + 0.5f) * app.cellSize;

        float dx = app.pos.x - fx;
        float dz = app.pos.z - fz;
        float dist = std::sqrt(dx * dx + dz * dz);

        float radius = std::max(app.cellSize * 0.45f, 0.25f);

        if (dist < radius && sampleHeight01(app.pos.x, app.pos.z) <= app.walkLimit01 + 0.01f)
        {
            if (!app.won)
            {
                app.won = true;
                std::cout << "You reached the finish! Press R to restart or Tab for preview.\n";
            }
        }
    }
}

static void drawTerrain(const Mat4& view, const Mat4& proj, const Vec3& cameraPos, const Vec3& fogColor)
{
    if (app.indexCount <= 0)
        return;

    glUseProgram(app.progWorld);

    setUniformMatrix4(app.progWorld, "uModel", identity());
    setUniformMatrix4(app.progWorld, "uView", view);
    setUniformMatrix4(app.progWorld, "uProj", proj);
    setUniformVec3(app.progWorld, "uCameraPos", cameraPos);
    setUniformVec3(app.progWorld, "uFogColor", fogColor);
    setUniformFloat(app.progWorld, "uHeightScale", app.heightScale);
    setUniformFloat(app.progWorld, "uWalkLimit01", app.walkLimit01);
    setUniformInt(app.progWorld, "uUseOverride", 0);
    setUniformVec3(app.progWorld, "uOverrideColor", Vec3(0.0f, 0.0f, 0.0f));
    setUniformFloat(app.progWorld, "uAlpha", 1.0f);

    glBindVertexArray(app.vaoTerrain);
    glDrawElements(GL_TRIANGLES, app.indexCount, GL_UNSIGNED_INT, nullptr);
}

static void drawMarker(
    const Mat4& model,
    const Vec3& color,
    float alpha,
    const Mat4& view,
    const Mat4& proj,
    const Vec3& cameraPos,
    const Vec3& fogColor)
{
    glUseProgram(app.progWorld);

    setUniformMatrix4(app.progWorld, "uModel", model);
    setUniformMatrix4(app.progWorld, "uView", view);
    setUniformMatrix4(app.progWorld, "uProj", proj);
    setUniformVec3(app.progWorld, "uCameraPos", cameraPos);
    setUniformVec3(app.progWorld, "uFogColor", fogColor);
    setUniformFloat(app.progWorld, "uHeightScale", app.heightScale);
    setUniformFloat(app.progWorld, "uWalkLimit01", app.walkLimit01);
    setUniformInt(app.progWorld, "uUseOverride", 1);
    setUniformVec3(app.progWorld, "uOverrideColor", color);
    setUniformFloat(app.progWorld, "uAlpha", alpha);

    glBindVertexArray(app.vaoMarker);
    glDrawArrays(GL_TRIANGLES, 0, app.markerVertexCount);
}

static void renderWorld(float time)
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

    drawTerrain(view, proj, app.pos, fogColor);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    if (app.hasFinish)
    {
        float fx = (app.finish.x + 0.5f) * app.cellSize;
        float fz = (app.finish.y + 0.5f) * app.cellSize;
        float fy = sampleWorldHeight(fx, fz);

        float pulse = 0.30f + 0.10f * std::sin(time * 4.0f);

        Vec3 color = app.won
            ? Vec3(1.0f, 0.85f, 0.20f)
            : Vec3(0.10f, 0.90f, 0.25f);

        Mat4 m = multiply(
            translate(Vec3(fx, fy + pulse * 0.5f, fz)),
            scale(Vec3(app.cellSize * 0.85f, pulse, app.cellSize * 0.85f))
        );

        drawMarker(m, color, 0.55f, view, proj, app.pos, fogColor);
    }

    if (app.hasStart)
    {
        float sx = (app.start.x + 0.5f) * app.cellSize;
        float sz = (app.start.y + 0.5f) * app.cellSize;
        float sy = sampleWorldHeight(sx, sz);

        float h = 0.12f;

        Mat4 m = multiply(
            translate(Vec3(sx, sy + h * 0.5f, sz)),
            scale(Vec3(app.cellSize * 0.85f, h, app.cellSize * 0.85f))
        );

        drawMarker(m, Vec3(0.15f, 0.45f, 1.00f), 0.45f, view, proj, app.pos, fogColor);
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
    setUniformFloat(app.progPreview, "uWalkLimit01", app.walkLimit01);
    setUniformVec2(app.progPreview, "uStart", app.start);
    setUniformVec2(app.progPreview, "uFinish", app.finish);
    setUniformInt(app.progPreview, "uHasStart", app.hasStart ? 1 : 0);
    setUniformInt(app.progPreview, "uHasFinish", app.hasFinish ? 1 : 0);
    setUniformVec2(app.progPreview, "uHover", app.hover);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app.textureHeight);
    setUniformInt(app.progPreview, "uHeight", 0);

    glBindVertexArray(app.vaoPreview);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, fw, fh);
}

static void render(float time)
{
    if (app.preview)
        renderPreview();
    else
        renderWorld(time);
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
            rebuildTerrainMesh();
            ensureValidMarkers();
        }
        else if (key == GLFW_KEY_R)
        {
            findDefaultStartFinish();
        }
        else if (key == GLFW_KEY_B)
        {
            blurHeightmap();
        }
        else if (key == GLFW_KEY_COMMA)
        {
            app.heightScale = std::max(0.01f, app.heightScale - 0.25f);
            rebuildTerrainMesh();
        }
        else if (key == GLFW_KEY_PERIOD)
        {
            app.heightScale = std::min(40.0f, app.heightScale + 0.25f);
            rebuildTerrainMesh();
        }
        else if (key == GLFW_KEY_MINUS)
        {
            app.walkLimit01 = std::max(0.02f, app.walkLimit01 - 0.02f);
        }
        else if (key == GLFW_KEY_EQUAL)
        {
            app.walkLimit01 = std::min(1.0f, app.walkLimit01 + 0.02f);
        }
        else if (key == GLFW_KEY_LEFT_BRACKET)
        {
            app.maxClimb = std::max(0.05f, app.maxClimb - 0.1f);
        }
        else if (key == GLFW_KEY_RIGHT_BRACKET)
        {
            app.maxClimb = std::min(8.0f, app.maxClimb + 0.1f);
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
        if (cellWalkable(cx, cy))
        {
            app.start = Vec2(static_cast<float>(cx), static_cast<float>(cy));
            app.hasStart = true;
        }
        else
        {
            std::cout << "Start must be on a walkable low area.\n";
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (cellWalkable(cx, cy))
        {
            app.finish = Vec2(static_cast<float>(cx), static_cast<float>(cy));
            app.hasFinish = true;
        }
        else
        {
            std::cout << "Finish must be on a walkable low area.\n";
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
    app.cellSize = clampf(app.cellSize + delta, 0.1f, 10.0f);
    rebuildTerrainMesh();
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

    app.window = glfwCreateWindow(1280, 720, "OpenGL 4.1 Heightmap Labyrinth Maze", nullptr, nullptr);
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

    glGenTextures(1, &app.textureHeight);
    glBindTexture(GL_TEXTURE_2D, app.textureHeight);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
    if (app.progWorld) glDeleteProgram(app.progWorld);
    if (app.progPreview) glDeleteProgram(app.progPreview);

    if (app.vaoTerrain) glDeleteVertexArrays(1, &app.vaoTerrain);
    if (app.vaoMarker) glDeleteVertexArrays(1, &app.vaoMarker);
    if (app.vaoPreview) glDeleteVertexArrays(1, &app.vaoPreview);

    if (app.vboTerrain) glDeleteBuffers(1, &app.vboTerrain);
    if (app.eboTerrain) glDeleteBuffers(1, &app.eboTerrain);
    if (app.vboMarker) glDeleteBuffers(1, &app.vboMarker);
    if (app.vboPreview) glDeleteBuffers(1, &app.vboPreview);

    if (app.textureHeight) glDeleteTextures(1, &app.textureHeight);

    if (app.window)
        glfwDestroyWindow(app.window);

    glfwTerminate();
}

int main(int argc, char** argv)
{
    std::string mazePath;

    float cliCellSize = app.cellSize;
    float cliHeightScale = app.heightScale;
    float cliWalkLimit01 = app.walkLimit01;

    if (argc > 1)
        mazePath = argv[1];

    if (argc > 2)
        cliCellSize = std::max(0.1f, static_cast<float>(std::atof(argv[2])));

    if (argc > 3)
        cliHeightScale = std::max(0.01f, static_cast<float>(std::atof(argv[3])));

    if (argc > 4)
        cliWalkLimit01 = clampf(static_cast<float>(std::atof(argv[4])), 0.02f, 1.0f);

    if (!init())
        return 1;

    bool loaded = false;
    if (!mazePath.empty())
    {
        loaded = loadMazeFromFile(mazePath);
        if (!loaded)
            std::cerr << "Failed to load heightmap PNG: " << mazePath << "\nUsing built-in circular demo maze.\n";
    }

    if (!loaded)
    {
        generateDefaultCircularHeightmap();
    }

    app.cellSize = cliCellSize;
    app.heightScale = cliHeightScale;
    app.walkLimit01 = cliWalkLimit01;

    rebuildTerrainMesh();
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