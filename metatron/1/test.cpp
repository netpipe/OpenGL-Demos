// metatron_cube.cpp
//
// OpenGL 4.1 core profile Metatron-style self-assembly demo.
// GLFW + GLEW, no GLM.
//
// Controls:
//   1..6   showroom stages
//   A      assembly mode A
//   B      assembly mode B
//   R      restart
//   Space  pause
//   G      toggle guide rings
//   F      toggle auto rotate
//   H      help
//

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <algorithm>
#include <sstream>

using namespace std;

static const float PI = 3.14159265358979323846f;
static const int NODE_COUNT = 15;
static const float WORLD_RADIUS = 2.3f;

//------------------------------------------------------------------------------
// Minimal math, no GLM.
//------------------------------------------------------------------------------

struct Vec3
{
    float x, y, z;

    Vec3(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f)
        : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s)       const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s)       const { return Vec3(x / s, y / s, z / s); }

    Vec3& operator+=(const Vec3& o)
    {
        x += o.x; y += o.y; z += o.z;
        return *this;
    }
};

inline Vec3 operator*(float s, const Vec3& v)
{
    return v * s;
}

struct Vec4
{
    float x, y, z, w;

    Vec4(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f, float w_ = 1.0f)
        : x(x_), y(y_), z(z_), w(w_) {}
};

inline float dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline float length(const Vec3& v)
{
    return sqrtf(dot(v, v));
}

inline Vec3 normalize(Vec3 v)
{
    float l = length(v);
    if (l < 1e-8f) return Vec3(0.0f, 0.0f, 0.0f);
    return v / l;
}

inline Vec3 mix(const Vec3& a, const Vec3& b, float t)
{
    return a + (b - a) * t;
}

inline float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline float smoothstep(float a, float b, float x)
{
    float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float easeOutCubic(float t)
{
    float u = 1.0f - t;
    return 1.0f - u * u * u;
}

inline float easeInOutCubic(float t)
{
    if (t < 0.5f)
        return 4.0f * t * t * t;
    float u = 2.0f * t - 2.0f;
    return 0.5f * u * u * u + 1.0f;
}

inline float easeOutBack(float x)
{
    const float c1 = 1.70158f;
    const float c3 = c1 + 1.0f;
    float p = x - 1.0f;
    return 1.0f + c3 * p * p * p + c1 * p * p;
}

inline Vec3 rotateAround(Vec3 v, Vec3 axis, float angle)
{
    axis = normalize(axis);
    if (length(axis) < 1e-7f) return v;

    float c = cosf(angle);
    float s = sinf(angle);

    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0f - c);
}

inline Vec3 slerp(Vec3 a, Vec3 b, float t)
{
    a = normalize(a);
    b = normalize(b);

    float d = clampf(dot(a, b), -1.0f, 1.0f);

    if (d > 0.9995f)
        return normalize(mix(a, b, t));

    if (d < -0.9995f)
    {
        Vec3 axis = cross(a, b);
        if (length(axis) < 1e-5f)
            axis = cross(a, Vec3(0.0f, 1.0f, 0.0f));
        if (length(axis) < 1e-5f)
            axis = Vec3(1.0f, 0.0f, 0.0f);

        axis = normalize(axis);
        return rotateAround(a, axis, PI * t);
    }

    float theta = acosf(d);
    float s = sinf(theta);
    if (fabsf(s) < 1e-6f)
        return normalize(mix(a, b, t));

    float wa = sinf((1.0f - t) * theta) / s;
    float wb = sinf(t * theta) / s;

    return normalize(a * wa + b * wb);
}

struct Mat4
{
    float m[16];

    Mat4()
    {
        for (int i = 0; i < 16; ++i)
            m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }

    static Mat4 zero()
    {
        Mat4 r;
        for (int i = 0; i < 16; ++i)
            r.m[i] = 0.0f;
        return r;
    }

    static Mat4 perspective(float fovY, float aspect, float zNear, float zFar)
    {
        Mat4 r = Mat4::zero();

        float f = 1.0f / tanf(fovY * 0.5f);

        r.m[0]  = f / aspect;
        r.m[5]  = f;
        r.m[10] = (zFar + zNear) / (zNear - zFar);
        r.m[11] = -1.0f;
        r.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);

        return r;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up)
    {
        Vec3 f = normalize(center - eye);
        Vec3 s = normalize(cross(f, up));
        Vec3 u = cross(s, f);

        Mat4 r;

        r.m[0] = s.x;
        r.m[1] = u.x;
        r.m[2] = -f.x;
        r.m[3] = 0.0f;

        r.m[4] = s.y;
        r.m[5] = u.y;
        r.m[6] = -f.y;
        r.m[7] = 0.0f;

        r.m[8]  = s.z;
        r.m[9]  = u.z;
        r.m[10] = -f.z;
        r.m[11] = 0.0f;

        r.m[12] = -dot(s, eye);
        r.m[13] = -dot(u, eye);
        r.m[14] =  dot(f, eye);
        r.m[15] = 1.0f;

        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r = Mat4::zero();

    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            r.m[col * 4 + row] = sum;
        }
    }

    return r;
}

//------------------------------------------------------------------------------
// Shaders.
//------------------------------------------------------------------------------

static const char* LINE_VS = R"GLSL(
#version 410 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;

uniform mat4 uProj;
uniform mat4 uView;

out vec4 vColor;

void main()
{
    gl_Position = uProj * uView * vec4(aPos, 1.0);
    vColor = aColor;
}
)GLSL";

static const char* LINE_FS = R"GLSL(
#version 410 core

in vec4 vColor;
out vec4 frag;

void main()
{
    frag = vec4(vColor.rgb, vColor.a);
}
)GLSL";

static const char* POINT_VS = R"GLSL(
#version 410 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in float aSize;

uniform mat4 uProj;
uniform mat4 uView;
uniform float uPointScale;

out vec4 vColor;

void main()
{
    vec4 clip = uProj * uView * vec4(aPos, 1.0);
    gl_Position = clip;

    float w = max(clip.w, 0.0001);
    gl_PointSize = clamp(aSize * uPointScale / w, 1.0, 512.0);

    vColor = aColor;
}
)GLSL";

static const char* POINT_FS = R"GLSL(
#version 410 core

in vec4 vColor;
out vec4 frag;

void main()
{
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);

    if (r2 > 1.0)
        discard;

    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 n = normalize(vec3(p.x, -p.y, z));

    vec3 L = normalize(vec3(0.4, 0.6, 1.0));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);

    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), 48.0);
    float rim  = pow(1.0 - z, 2.0);

    vec3 col = vColor.rgb * (0.30 + 0.75 * diff)
             + vec3(1.0) * spec * 0.65
             + vColor.rgb * rim * 0.85;

    float a = vColor.a * (0.65 + 0.35 * z);

    frag = vec4(col, a);
}
)GLSL";

//------------------------------------------------------------------------------
// App / geometry types.
//------------------------------------------------------------------------------

enum Mode
{
    MODE_SHOWROOM = 0,
    MODE_ASSEMBLE_A,
    MODE_ASSEMBLE_B
};

struct Edge
{
    int a;
    int b;
    int stage;
};

struct App
{
    Mode mode = MODE_ASSEMBLE_A;

    int showroomStage = 5;

    double time = 0.0;
    bool playing = true;
    float speed = 1.0f;

    bool guide = true;
    bool autoRotate = true;

    float yaw = 0.9f;
    float pitch = 0.32f;
    float dist = 8.6f;

    bool dragging = false;
    double lastX = 0.0;
    double lastY = 0.0;

    int fbW = 1280;
    int fbH = 720;
};

App app;
GLFWwindow* gWindow = nullptr;

Vec3 gTarget[NODE_COUNT];
Vec3 gRaw[NODE_COUNT];
Vec4 gNodeColor[NODE_COUNT];
float gNodeSize[NODE_COUNT];
int gNodeGroup[NODE_COUNT];

vector<Edge> gEdges;

float gNodeDelay[NODE_COUNT];
float gNodeDur[NODE_COUNT];
Vec3 gNodeStart[NODE_COUNT];

vector<float> gEdgeDelay;
vector<float> gEdgeDur;

GLuint lineProg = 0;
GLuint pointProg = 0;

GLuint lineVAO = 0;
GLuint lineVBO = 0;

GLuint pointVAO = 0;
GLuint pointVBO = 0;

GLint lineProjLoc = -1;
GLint lineViewLoc = -1;

GLint pointProjLoc = -1;
GLint pointViewLoc = -1;
GLint pointScaleLoc = -1;

//------------------------------------------------------------------------------
// Small helpers.
//------------------------------------------------------------------------------

inline int nodeStage(int i)
{
    switch (gNodeGroup[i])
    {
        case 0: return 0; // seed
        case 1: return 1; // first pyramid
        case 2: return 2; // second pyramid
        default: return 4; // side points
    }
}

inline float baseEdgeAlpha(int stage)
{
    if (stage == 1 || stage == 2) return 0.90f;
    if (stage == 3) return 0.65f;
    if (stage == 4) return 0.50f;
    return 0.16f;
}

inline void pushVertex(vector<float>& v, const Vec3& p, const Vec4& c)
{
    v.push_back(p.x);
    v.push_back(p.y);
    v.push_back(p.z);
    v.push_back(c.x);
    v.push_back(c.y);
    v.push_back(c.z);
    v.push_back(c.w);
}

inline void pushPointVertex(vector<float>& v, const Vec3& p, const Vec4& c, float size)
{
    v.push_back(p.x);
    v.push_back(p.y);
    v.push_back(p.z);
    v.push_back(c.x);
    v.push_back(c.y);
    v.push_back(c.z);
    v.push_back(c.w);
    v.push_back(size);
}

void appendCircle(vector<float>& lines, const Vec3& u, const Vec3& v, float radius, float alpha)
{
    const int segments = 80;
    Vec4 c(0.50f, 0.72f, 1.0f, alpha);

    for (int i = 0; i < segments; ++i)
    {
        float a0 = 2.0f * PI * float(i) / float(segments);
        float a1 = 2.0f * PI * float(i + 1) / float(segments);

        Vec3 p0 = (u * cosf(a0) + v * sinf(a0)) * radius;
        Vec3 p1 = (u * cosf(a1) + v * sinf(a1)) * radius;

        pushVertex(lines, p0, c);
        pushVertex(lines, p1, c);
    }
}

//------------------------------------------------------------------------------
// Geometry construction.
//
// Node layout:
//   0       center seed
//   1..4    first tetrahedron / pyramid
//   5..8    second tetrahedron / pyramid
//   9..14   six spherical side points, forming an octahedral/polyhedral cage
//
// All outer points are placed on the same sphere.
//------------------------------------------------------------------------------

int classifyEdge(int i, int j)
{
    bool iCube = (i >= 1 && i <= 8);
    bool jCube = (j >= 1 && j <= 8);

    bool iAxis = (i >= 9);
    bool jAxis = (j >= 9);

    // Center to side points: part of the polyhedral cage.
    if (i == 0 && jAxis) return 4;

    // Center to tetra vertices: final full web.
    if (i == 0 && jCube) return 5;

    // Cube/tetra vertex connections.
    if (iCube && jCube)
    {
        int gi = gNodeGroup[i];
        int gj = gNodeGroup[j];

        if (gi == 1 && gj == 1) return 1;
        if (gi == 2 && gj == 2) return 2;

        // The two tetrahedra are inscribed in a cube.
        // Cube edges connect vertices differing in exactly one coordinate sign.
        int diff = 0;
        if (fabsf(gRaw[i].x - gRaw[j].x) > 0.1f) ++diff;
        if (fabsf(gRaw[i].y - gRaw[j].y) > 0.1f) ++diff;
        if (fabsf(gRaw[i].z - gRaw[j].z) > 0.1f) ++diff;

        if (diff == 1) return 3;

        // Body diagonals / remaining cross links.
        return 5;
    }

    // Axis/side point to axis/side point.
    if (iAxis && jAxis)
    {
        float d = dot(gRaw[i], gRaw[j]);

        // Perpendicular axes make the octahedral cage.
        if (fabsf(d) < 0.01f) return 4;

        // Opposite axes are final web connections.
        return 5;
    }

    // Axis/side point to cube/tetra point.
    if ((iAxis && jCube) || (iCube && jAxis))
    {
        int ai = iAxis ? i : j;
        int ci = iCube ? i : j;

        float d = dot(normalize(gRaw[ai]), normalize(gRaw[ci]));

        // Face center connects to its four nearby cube vertices.
        if (d > 0.5f) return 4;

        return 5;
    }

    return 5;
}

void buildGeometry()
{
    // Seed / center.
    gTarget[0] = Vec3(0.0f, 0.0f, 0.0f);
    gRaw[0] = Vec3(0.0f, 0.0f, 0.0f);
    gNodeColor[0] = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    gNodeSize[0] = 0.30f;
    gNodeGroup[0] = 0;

    // Two interlocking tetrahedra as cube vertices.
    // First tetra: product of signs +1.
    // Second tetra: product of signs -1.
    const int cubeSigns[8][3] =
    {
        {  1,  1,  1 },
        {  1, -1, -1 },
        { -1,  1, -1 },
        { -1, -1,  1 },

        { -1, -1, -1 },
        { -1,  1,  1 },
        {  1, -1,  1 },
        {  1,  1, -1 }
    };

    for (int i = 0; i < 8; ++i)
    {
        int idx = 1 + i;

Vec3 raw{
    static_cast<float>(cubeSigns[i][0]),
    static_cast<float>(cubeSigns[i][1]),
    static_cast<float>(cubeSigns[i][2])
};

        gRaw[idx] = raw;
        gTarget[idx] = normalize(raw) * WORLD_RADIUS;

        bool firstPyramid = (i < 4);

        gNodeGroup[idx] = firstPyramid ? 1 : 2;

        if (firstPyramid)
            gNodeColor[idx] = Vec4(1.0f, 0.78f, 0.25f, 1.0f);
        else
            gNodeColor[idx] = Vec4(0.25f, 0.78f, 1.0f, 1.0f);

        gNodeSize[idx] = 0.20f;
    }

    // Six extra side points, on the same sphere.
    const int axisSigns[6][3] =
    {
        {  1,  0,  0 },
        { -1,  0,  0 },
        {  0,  1,  0 },
        {  0, -1,  0 },
        {  0,  0,  1 },
        {  0,  0, -1 }
    };

    for (int i = 0; i < 6; ++i)
    {
        int idx = 9 + i;

Vec3 raw{
    static_cast<float>(axisSigns[i][0]),
    static_cast<float>(axisSigns[i][1]),
    static_cast<float>(axisSigns[i][2])
};

        gRaw[idx] = raw;
        gTarget[idx] = raw * WORLD_RADIUS;

        gNodeGroup[idx] = 3;
        gNodeColor[idx] = Vec4(0.35f, 1.0f, 0.55f, 1.0f);
        gNodeSize[idx] = 0.17f;
    }

    // Complete graph over all nodes, classified into construction stages.
    gEdges.clear();

    for (int i = 0; i < NODE_COUNT; ++i)
    {
        for (int j = i + 1; j < NODE_COUNT; ++j)
        {
            Edge e;
            e.a = i;
            e.b = j;
            e.stage = classifyEdge(i, j);
            gEdges.push_back(e);
        }
    }

    gEdgeDelay.assign(gEdges.size(), 0.0f);
    gEdgeDur.assign(gEdges.size(), 1.0f);
}

//------------------------------------------------------------------------------
// Assembly timelines.
//------------------------------------------------------------------------------

void buildTimeline(Mode mode)
{
    gEdgeDelay.assign(gEdges.size(), 0.0f);
    gEdgeDur.assign(gEdges.size(), 1.0f);

    for (int i = 0; i < NODE_COUNT; ++i)
    {
        gNodeDelay[i] = 0.0f;
        gNodeDur[i] = 1.0f;
        gNodeStart[i] = Vec3(0.0f, 0.0f, 0.0f);
    }

    if (mode == MODE_SHOWROOM)
        return;

    mt19937 rng(987654);
    uniform_real_distribution<float> uni(0.0f, 1.0f);

    if (mode == MODE_ASSEMBLE_A)
    {
        // Hierarchical assembly:
        // seed -> first pyramid -> second pyramid -> cube links -> side cage -> full web.

        gNodeDelay[0] = 0.0f;
        gNodeDur[0] = 1.0f;

        for (int i = 1; i <= 4; ++i)
        {
            gNodeDelay[i] = 0.6f + float(i - 1) * 0.35f;
            gNodeDur[i] = 1.5f;
        }

        for (int i = 5; i <= 8; ++i)
        {
            gNodeDelay[i] = 3.4f + float(i - 5) * 0.35f;
            gNodeDur[i] = 1.5f;
        }

        for (int i = 9; i <= 14; ++i)
        {
            gNodeDelay[i] = 8.8f + float(i - 9) * 0.22f;
            gNodeDur[i] = 1.2f;
        }

        vector<int> cnt(6, 0);

        for (size_t k = 0; k < gEdges.size(); ++k)
        {
            const Edge& e = gEdges[k];

            float arrivalA = gNodeDelay[e.a] + gNodeDur[e.a];
            float arrivalB = gNodeDelay[e.b] + gNodeDur[e.b];
            float arrival = max(arrivalA, arrivalB);

            float delay = arrival;
            float dur = 0.8f;

            switch (e.stage)
            {
                case 1:
                    delay = arrival + cnt[1] * 0.18f;
                    dur = 0.8f;
                    break;

                case 2:
                    delay = arrival + cnt[2] * 0.18f;
                    dur = 0.8f;
                    break;

                case 3:
                    delay = 6.4f + cnt[3] * 0.09f;
                    dur = 0.85f;
                    break;

                case 4:
                    delay = arrival + cnt[4] * 0.05f;
                    dur = 0.7f;
                    break;

                default:
                    delay = 14.5f + cnt[5] * 0.025f;
                    dur = 1.4f;
                    break;
            }

            gEdgeDelay[k] = delay;
            gEdgeDur[k] = dur;

            ++cnt[e.stage];
        }
    }
    else if (mode == MODE_ASSEMBLE_B)
    {
        // More parallel / sphere-sweep assembly.
        // Both pyramids arrive together from rotated positions on the sphere.
        // Local shorter links tend to form before long-range web lines.

        gNodeDelay[0] = 0.2f;
        gNodeDur[0] = 0.9f;

        for (int i = 1; i <= 8; ++i)
        {
            gNodeDelay[i] = 0.5f + float(i - 1) * 0.08f;
            gNodeDur[i] = 1.8f;

            Vec3 dir = normalize(gTarget[i]);

            float ang = PI + (uni(rng) - 0.5f) * 2.0f;
            Vec3 start = rotateAround(dir, Vec3(0.0f, 1.0f, 0.0f), ang);
            start = rotateAround(start, Vec3(1.0f, 0.0f, 0.0f), (uni(rng) - 0.5f) * 1.2f);

            gNodeStart[i] = normalize(start);
        }

        for (int i = 9; i <= 14; ++i)
        {
            gNodeDelay[i] = 4.4f + float(i - 9) * 0.12f;
            gNodeDur[i] = 1.1f;

            Vec3 dir = normalize(gTarget[i]);

            Vec3 upLike = (fabs(dot(dir, Vec3(0.0f, 1.0f, 0.0f))) > 0.9f)
                        ? Vec3(1.0f, 0.0f, 0.0f)
                        : Vec3(0.0f, 1.0f, 0.0f);

            Vec3 tangent = normalize(cross(dir, upLike));
            if (length(tangent) < 0.01f)
                tangent = Vec3(1.0f, 0.0f, 0.0f);

            gNodeStart[i] = normalize(rotateAround(tangent, dir, uni(rng) * 2.0f * PI));
        }

        vector<int> cnt(6, 0);

        for (size_t k = 0; k < gEdges.size(); ++k)
        {
            const Edge& e = gEdges[k];

            float arrivalA = gNodeDelay[e.a] + gNodeDur[e.a];
            float arrivalB = gNodeDelay[e.b] + gNodeDur[e.b];
            float arrival = max(arrivalA, arrivalB);

            float len = length(gTarget[e.a] - gTarget[e.b]);

            float delay = arrival;
            float dur = 0.8f;

            switch (e.stage)
            {
                case 1:
                    delay = arrival + len * 0.25f + cnt[1] * 0.05f;
                    dur = 0.8f;
                    break;

                case 2:
                    delay = arrival + len * 0.25f + cnt[2] * 0.05f;
                    dur = 0.8f;
                    break;

                case 3:
                    delay = arrival + 0.85f + len * 0.2f + cnt[3] * 0.03f;
                    dur = 0.8f;
                    break;

                case 4:
                    delay = arrival + 0.4f + len * 0.35f + cnt[4] * 0.02f;
                    dur = 0.7f;
                    break;

                default:
                    delay = arrival + 5.6f + len * 0.25f + cnt[5] * 0.012f;
                    dur = 1.2f;
                    break;
            }

            gEdgeDelay[k] = delay;
            gEdgeDur[k] = dur;

            ++cnt[e.stage];
        }
    }
}

//------------------------------------------------------------------------------
// Per-frame node state.
//------------------------------------------------------------------------------

void getNodeRenderState(int i, Vec3& pos, Vec4& col, float& size)
{
    float t = float(app.time);

    col = gNodeColor[i];
    size = gNodeSize[i];

    if (app.mode == MODE_SHOWROOM)
    {
        int ns = nodeStage(i);

        if (app.showroomStage < ns)
        {
            pos = gTarget[i];
            col.w = 0.0f;
            size = 0.0f;
            return;
        }

        pos = gTarget[i];

        if (length(gTarget[i]) > 0.01f)
        {
            float breathe = 1.0f + 0.015f * sinf(t * 1.7f + float(i) * 0.7f);
            pos = pos * breathe;
        }

        col.w = 1.0f;
        size = gNodeSize[i] * (1.0f + 0.06f * sinf(t * 2.3f + float(i) * 1.3f));

        return;
    }

    float dur = max(0.001f, gNodeDur[i]);
    float p = clampf((t - gNodeDelay[i]) / dur, 0.0f, 1.0f);

    if (p <= 0.0f)
    {
        if (app.mode == MODE_ASSEMBLE_B && i != 0)
            pos = gNodeStart[i] * WORLD_RADIUS;
        else
            pos = Vec3(0.0f, 0.0f, 0.0f);

        col.w = 0.0f;
        size = 0.0f;
        return;
    }

    float alpha = smoothstep(0.0f, 0.18f, p);

    if (app.mode == MODE_ASSEMBLE_A)
    {
        if (i == 0)
        {
            pos = Vec3(0.0f, 0.0f, 0.0f);
            size = gNodeSize[i] * easeOutBack(p);
        }
        else
        {
            float e = easeOutCubic(p);
            Vec3 dir = normalize(gTarget[i]);

            float ang = (1.0f - e) * (2.8f + 0.33f * float(i));
            Vec3 spun = rotateAround(dir, Vec3(0.0f, 1.0f, 0.0f), ang);

            float radius = WORLD_RADIUS * e;
            pos = spun * radius;

            size = gNodeSize[i] * easeOutBack(p);
        }
    }
    else // MODE_ASSEMBLE_B
    {
        if (i == 0)
        {
            pos = Vec3(0.0f, 0.0f, 0.0f);
            size = gNodeSize[i] * easeOutBack(p);
        }
        else
        {
            float e = easeInOutCubic(p);

            Vec3 dir = slerp(gNodeStart[i], normalize(gTarget[i]), e);
            float radius = WORLD_RADIUS * (0.9f + 0.1f * e);

            pos = dir * radius;
            size = gNodeSize[i] * easeOutBack(p);
        }
    }

    col.w = alpha;
}

//------------------------------------------------------------------------------
// UI / mode management.
//------------------------------------------------------------------------------

void updateTitle()
{
    if (!gWindow)
        return;

    static const char* stageNames[] =
    {
        "Seed",
        "First Pyramid",
        "Two Pyramids",
        "Cube Cross Links",
        "Side-Point Polyhedron",
        "Spherical Metatron"
    };

    ostringstream ss;
    ss << "Metatron's Cube | ";

    if (app.mode == MODE_SHOWROOM)
    {
        int s = clampf(float(app.showroomStage), 0.0f, 5.0f);
        ss << "Showroom " << (app.showroomStage + 1) << "/6: "
           << stageNames[app.showroomStage];
    }
    else if (app.mode == MODE_ASSEMBLE_A)
    {
        ss << "Self Assembly A: Hierarchical Pyramids";
    }
    else
    {
        ss << "Self Assembly B: Sphere Sweep / Local First";
    }

    if (!app.playing)
        ss << " | paused";

    glfwSetWindowTitle(gWindow, ss.str().c_str());
}

void printHelp()
{
    cout <<
        "\n"
        "Metatron-style assembly demo\n"
        "\n"
        "Keys:\n"
        "  1..6   showroom stages\n"
        "  A      assembly mode A\n"
        "  B      assembly mode B\n"
        "  R      restart current mode\n"
        "  Space  pause/unpause\n"
        "  G      toggle guide sphere rings\n"
        "  F      toggle auto camera orbit\n"
        "  H      help\n"
        "\n"
        "Mouse:\n"
        "  drag   orbit camera\n"
        "  wheel  zoom\n"
        "\n";
}

void setMode(Mode m)
{
    app.mode = m;
    app.time = 0.0;
    app.playing = true;

    buildTimeline(m);
    updateTitle();
}

//------------------------------------------------------------------------------
// GLFW callbacks.
//------------------------------------------------------------------------------

void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;

    if (action != GLFW_PRESS)
        return;

    if (key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_SPACE)
    {
        app.playing = !app.playing;
        updateTitle();
    }
    else if (key == GLFW_KEY_A)
    {
        setMode(MODE_ASSEMBLE_A);
    }
    else if (key == GLFW_KEY_B)
    {
        setMode(MODE_ASSEMBLE_B);
    }
    else if (key == GLFW_KEY_R)
    {
        app.time = 0.0;
        buildTimeline(app.mode);
        updateTitle();
    }
    else if (key == GLFW_KEY_G)
    {
        app.guide = !app.guide;
    }
    else if (key == GLFW_KEY_F)
    {
        app.autoRotate = !app.autoRotate;
    }
    else if (key == GLFW_KEY_H)
    {
        printHelp();
    }
    else if (key == GLFW_KEY_RIGHT)
    {
        if (app.mode == MODE_SHOWROOM)
        {
            app.showroomStage = min(5, app.showroomStage + 1);
            updateTitle();
        }
        else
        {
            setMode(app.mode == MODE_ASSEMBLE_A ? MODE_ASSEMBLE_B : MODE_ASSEMBLE_A);
        }
    }
    else if (key == GLFW_KEY_LEFT)
    {
        if (app.mode == MODE_SHOWROOM)
        {
            app.showroomStage = max(0, app.showroomStage - 1);
            updateTitle();
        }
        else
        {
            setMode(app.mode == MODE_ASSEMBLE_A ? MODE_ASSEMBLE_B : MODE_ASSEMBLE_A);
        }
    }
    else if (key >= GLFW_KEY_1 && key <= GLFW_KEY_6)
    {
        app.mode = MODE_SHOWROOM;
        app.showroomStage = key - GLFW_KEY_1;
        app.time = 0.0;

        buildTimeline(app.mode);
        updateTitle();
    }
}

void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
    (void)mods;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            app.dragging = true;
            glfwGetCursorPos(w, &app.lastX, &app.lastY);
        }
        else
        {
            app.dragging = false;
        }
    }
}

void cursorPosCallback(GLFWwindow* w, double x, double y)
{
    if (app.dragging)
    {
        double dx = x - app.lastX;
        double dy = y - app.lastY;

        app.yaw += float(dx) * 0.005f;
        app.pitch += float(dy) * 0.005f;

        app.pitch = clampf(app.pitch, -1.45f, 1.45f);

        // Manual interaction overrides auto-orbit.
        app.autoRotate = false;
    }

    app.lastX = x;
    app.lastY = y;

    (void)w;
}

void scrollCallback(GLFWwindow* w, double xoffset, double yoffset)
{
    (void)w;
    (void)xoffset;

    app.dist *= expf(float(-yoffset) * 0.08f);
    app.dist = clampf(app.dist, 3.5f, 18.0f);
}

void framebufferSizeCallback(GLFWwindow* w, int width, int height)
{
    (void)w;

    app.fbW = width;
    app.fbH = height;
}

//------------------------------------------------------------------------------
// OpenGL initialization.
//------------------------------------------------------------------------------

GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);

    if (!ok)
    {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);

        vector<char> log(max(1, len));
        glGetShaderInfoLog(s, GLsizei(log.size()), nullptr, log.data());

        cerr << "Shader compile error:\n" << log.data() << "\n";

        return 0;
    }

    return s;
}

GLuint createProgram(const char* vsSrc, const char* fsSrc)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    if (!vs || !fs)
        return 0;

    GLuint p = glCreateProgram();

    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);

    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);

    if (!ok)
    {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);

        vector<char> log(max(1, len));
        glGetProgramInfoLog(p, GLsizei(log.size()), nullptr, log.data());

        cerr << "Program link error:\n" << log.data() << "\n";

        glDeleteProgram(p);
        p = 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return p;
}

bool initGL()
{
    glewExperimental = GL_TRUE;

    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        cerr << "GLEW init failed: " << glewGetErrorString(err) << "\n";
        return false;
    }

    lineProg = createProgram(LINE_VS, LINE_FS);
    pointProg = createProgram(POINT_VS, POINT_FS);

    if (!lineProg || !pointProg)
        return false;

    lineProjLoc = glGetUniformLocation(lineProg, "uProj");
    lineViewLoc = glGetUniformLocation(lineProg, "uView");

    pointProjLoc = glGetUniformLocation(pointProg, "uProj");
    pointViewLoc = glGetUniformLocation(pointProg, "uView");
    pointScaleLoc = glGetUniformLocation(pointProg, "uPointScale");

    // Line VAO/VBO: vec3 pos, vec4 color.
    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (const void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (const void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Point VAO/VBO: vec3 pos, vec4 color, float size.
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);

    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (const void*)(7 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);

    return true;
}

//------------------------------------------------------------------------------
// Rendering.
//------------------------------------------------------------------------------

void render()
{
    int w = app.fbW;
    int h = app.fbH;

    if (w <= 0 || h <= 0)
        return;

    glViewport(0, 0, w, h);

    glClearColor(0.012f, 0.018f, 0.045f, 1.0f);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glDisable(GL_CULL_FACE);

    glLineWidth(1.4f);

    float fov = 50.0f * PI / 180.0f;
    float aspect = float(w) / float(h);

    Mat4 proj = Mat4::perspective(fov, aspect, 0.1f, 100.0f);

    Vec3 eye(
        cosf(app.pitch) * sinf(app.yaw) * app.dist,
        sinf(app.pitch) * app.dist,
        cosf(app.pitch) * cosf(app.yaw) * app.dist
    );

    Mat4 view = Mat4::lookAt(eye, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));

    // Compute node states.
    Vec3 nodePos[NODE_COUNT];
    Vec4 nodeCol[NODE_COUNT];
    float nodeSize[NODE_COUNT];

    for (int i = 0; i < NODE_COUNT; ++i)
        getNodeRenderState(i, nodePos[i], nodeCol[i], nodeSize[i]);

    vector<float> lineData;
    lineData.reserve(12000);

    float t = float(app.time);

    // Optional guide sphere.
    if (app.guide)
    {
        float guideAlpha = 0.0f;

        if (app.mode == MODE_SHOWROOM)
        {
            if (app.showroomStage >= 4)
                guideAlpha = 0.05f;
        }
        else
        {
            guideAlpha = 0.05f * smoothstep(8.0f, 11.0f, t);
        }

        if (guideAlpha > 0.001f)
        {
            float r = WORLD_RADIUS * 1.18f;

            appendCircle(lineData, Vec3(1, 0, 0), Vec3(0, 1, 0), r, guideAlpha);
            appendCircle(lineData, Vec3(1, 0, 0), Vec3(0, 0, 1), r, guideAlpha);
            appendCircle(lineData, Vec3(0, 1, 0), Vec3(0, 0, 1), r, guideAlpha);
        }
    }

    // Draw edges.
    for (size_t k = 0; k < gEdges.size(); ++k)
    {
        const Edge& e = gEdges[k];

        float alpha = 0.0f;
        float progress = 1.0f;

        if (app.mode == MODE_SHOWROOM)
        {
            if (e.stage > app.showroomStage)
                continue;

            alpha = baseEdgeAlpha(e.stage);

            if (app.showroomStage == 5)
            {
                if (e.stage == 5)
                    alpha = 0.10f;
                else
                    alpha = baseEdgeAlpha(e.stage) * 0.75f;
            }
        }
        else
        {
            float dur = max(0.001f, gEdgeDur[k]);
            float p = clampf((t - gEdgeDelay[k]) / dur, 0.0f, 1.0f);

            if (p <= 0.0f)
                continue;

            progress = easeOutCubic(p);
            alpha = baseEdgeAlpha(e.stage) * smoothstep(0.0f, 0.12f, p);

            if (e.stage == 5)
                alpha *= 0.85f + 0.15f * sinf(t * 2.1f + float(k) * 0.17f);
        }

        int ia = e.a;
        int ib = e.b;

        // Grow lines from the node that arrived earlier.
        if (app.mode != MODE_SHOWROOM)
        {
            float arrA = gNodeDelay[ia] + gNodeDur[ia];
            float arrB = gNodeDelay[ib] + gNodeDur[ib];

            if (arrB < arrA)
                swap(ia, ib);
        }

        float aa = alpha * nodeCol[ia].w;
        float bb = alpha * nodeCol[ib].w;

        if (aa <= 0.001f && bb <= 0.001f)
            continue;

        Vec3 start = nodePos[ia];
        Vec3 end = mix(start, nodePos[ib], progress);

        if (progress < 1.0f)
            bb = min(1.0f, bb * 1.6f);

        pushVertex(lineData, start, Vec4(nodeCol[ia].x, nodeCol[ia].y, nodeCol[ia].z, aa));
        pushVertex(lineData, end,   Vec4(nodeCol[ib].x, nodeCol[ib].y, nodeCol[ib].z, bb));
    }

    // Draw nodes.
    vector<float> pointData;
    pointData.reserve(NODE_COUNT * 8);

    for (int i = 0; i < NODE_COUNT; ++i)
    {
        if (nodeCol[i].w <= 0.001f || nodeSize[i] <= 0.001f)
            continue;

        pushPointVertex(pointData, nodePos[i], nodeCol[i], nodeSize[i]);
    }

    // Lines.
    glUseProgram(lineProg);

    glUniformMatrix4fv(lineProjLoc, 1, GL_FALSE, proj.m);
    glUniformMatrix4fv(lineViewLoc, 1, GL_FALSE, view.m);

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        GLsizeiptr(lineData.size() * sizeof(float)),
        lineData.empty() ? nullptr : lineData.data(),
        GL_DYNAMIC_DRAW
    );

    if (!lineData.empty())
        glDrawArrays(GL_LINES, 0, GLsizei(lineData.size() / 7));

    // Points / spheres.
    glUseProgram(pointProg);

    glUniformMatrix4fv(pointProjLoc, 1, GL_FALSE, proj.m);
    glUniformMatrix4fv(pointViewLoc, 1, GL_FALSE, view.m);

    float pointScale = float(h) / (2.0f * tanf(fov * 0.5f));
    glUniform1f(pointScaleLoc, pointScale);

    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        GLsizeiptr(pointData.size() * sizeof(float)),
        pointData.empty() ? nullptr : pointData.data(),
        GL_DYNAMIC_DRAW
    );

    if (!pointData.empty())
        glDrawArrays(GL_POINTS, 0, GLsizei(pointData.size() / 8));

    glBindVertexArray(0);
    glUseProgram(0);
}

//------------------------------------------------------------------------------
// Main.
//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    if (!glfwInit())
    {
        cerr << "Failed to initialize GLFW.\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1280, 800, "Metatron's Cube", nullptr, nullptr);

    if (!window)
    {
        cerr << "Failed to create OpenGL 4.1 window.\n";
        glfwTerminate();
        return -1;
    }

    gWindow = window;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    glfwGetFramebufferSize(window, &app.fbW, &app.fbH);

    if (!initGL())
    {
        glfwTerminate();
        return -1;
    }

    buildGeometry();
    buildTimeline(app.mode);

    updateTitle();
    printHelp();

    double last = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        double now = glfwGetTime();
        double dt = now - last;
        last = now;

        if (dt > 0.1)
            dt = 0.1;

        if (app.playing)
            app.time += dt * double(app.speed);

        if (app.autoRotate && app.playing)
            app.yaw += float(dt) * 0.07f;

        // Loop assembly modes for a continuous demo.
        if (app.mode != MODE_SHOWROOM && app.time > 26.0)
        {
            app.time = 0.0;
            buildTimeline(app.mode);
        }

        render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}