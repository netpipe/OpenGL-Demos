// ============================================================================
//  TESSERACT // OBSERVER LIFE SIM  —  single-file OpenGL demo
// ----------------------------------------------------------------------------
//  Build:
//    Linux:   g++ -std=c++17 -O2 tess_life.cpp -o tess_life -lglfw -lGLEW -lGL
//    macOS:   g++ -std=c++17 tess_life.cpp -o tess_life -lglfw -lGLEW \
//                 -framework OpenGL
//    Windows: cl /O2 /EHsc tess_life.cpp /link glfw3.lib glew32.lib opengl32.lib
//
//  Controls:
//    WASD / arrows .... move          mouse .......... look
//    SHIFT ............ run           SPACE .......... archive a memory now
//    T ................ auto-walk AI  N .............. regenerate the room
//    R ................ reset sim     F .............. fullscreen
//    ESC .............. quit
//
//  Concept: the observer is always "here". Moving archives the current body
//  as a frozen scene-graph snapshot (a memory diorama) and a new observer
//  generation steps into its place. The room itself is procedurally generated.
//  No GLM — all math is hand-rolled below.
// ============================================================================

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>
#include <deque>
#include <string>

// ------------------------------------------------------------------ constants
static const float PI  = 3.14159265358979f;
static const float TAU = 6.28318530717958f;

static const int   MAX_MEMORIES = 28;         // living memory cap
static const float MEM_FADE     = 1.4f;       // seconds to dissolve a memory
static const float EYE_HEIGHT   = 1.7f;
static const float ROOM_HALF    = 13.0f;
static const float STEP_DIST    = 2.6f;       // walk distance between clones
static const float TESS_P       = 3.4f;       // 4D perspective distance
static const float TESS_SCALE   = 1.1f;
//static const Vec3_; // fwd placeholder never used; real Vec3 below

// ================================================================== math ====
struct Vec2 { float x, y; };

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(float s)       const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(float s)       const { return Vec3(x/s, y/s, z/s); }
    Vec3 operator-()              const { return Vec3(-x, -y, -z); } // Unary minus
    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; } // Subtraction assignment
};
struct Vec4 {
    float x, y, z, w;
    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    Vec4(const Vec3& v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {} // Construct from Vec3 + w
};
inline Vec3 operator*(float s, const Vec3& v) { return v * s; }
inline float  dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3   cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline float  length(const Vec3& v)  { return sqrtf(dot(v, v)); }
inline Vec3   normalize(const Vec3& v) { float l = length(v); return l > 1e-8f ? v/l : Vec3(); }
inline float  clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
inline float  clamp01(float v) { return clampf(v, 0.f, 1.f); }
inline float  lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline Vec3   mixv(const Vec3& a, const Vec3& b, float t) {
    return Vec3(lerpf(a.x,b.x,t), lerpf(a.y,b.y,t), lerpf(a.z,b.z,t));
}
inline Vec4   mixv(const Vec4& a, const Vec4& b, float t) { // Add this overload
    return Vec4(lerpf(a.x,b.x,t), lerpf(a.y,b.y,t), lerpf(a.z,b.z,t), lerpf(a.w,b.w,t));
}
inline float  lerpAngle(float a, float b, float t) {
    float d = fmodf(b - a + PI*3.0f, TAU) - PI;
    return a + d * t;
}

// Column-major 4x4 matrix (OpenGL convention). m[col*4+row].
struct Mat4 {
    float m[16];
    Mat4() { memset(m, 0, sizeof(m)); }
    static Mat4 identity() { Mat4 r; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
    static Mat4 translate(float x, float y, float z) {
        Mat4 r = identity(); r.m[12]=x; r.m[13]=y; r.m[14]=z; return r;
    }
    static Mat4 translate(const Vec3& v) { return translate(v.x, v.y, v.z); }
    static Mat4 scale(float x, float y, float z) {
        Mat4 r; r.m[0]=x; r.m[5]=y; r.m[10]=z; r.m[15]=1; return r;
    }
    static Mat4 scale(const Vec3& v) { return scale(v.x, v.y, v.z); }
    static Mat4 rotX(float a) {
        Mat4 r = identity(); float c = cosf(a), s = sinf(a);
        r.m[5]=c; r.m[6]=s; r.m[9]=-s; r.m[10]=c; return r;
    }
    static Mat4 rotY(float a) {
        Mat4 r = identity(); float c = cosf(a), s = sinf(a);
        r.m[0]=c; r.m[2]=-s; r.m[8]=s; r.m[10]=c; return r;
    }
    static Mat4 perspective(float fovDeg, float aspect, float zn, float zf) {
        Mat4 r; float f = 1.0f / tanf(fovDeg * PI / 360.0f);
        r.m[0]=f/aspect; r.m[5]=f;
        r.m[10]=(zf+zn)/(zn-zf); r.m[11]=-1.0f;
        r.m[14]=2.0f*zf*zn/(zn-zf); return r;
    }
    static Mat4 ortho(float l, float r_, float b, float t, float n, float f) {
        Mat4 r;
        r.m[0]=2.0f/(r_-l); r.m[5]=2.0f/(t-b); r.m[10]=-2.0f/(f-n);
        r.m[12]=-(r_+l)/(r_-l); r.m[13]=-(t+b)/(t-b); r.m[14]=-(f+n)/(f-n);
        r.m[15]=1.0f; return r;
    }
    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int c = 0; c < 4; c++)
            for (int rw = 0; rw < 4; rw++) {
                float s = 0;
                for (int k = 0; k < 4; k++) s += m[k*4+rw] * b.m[c*4+k];
                r.m[c*4+rw] = s;
            }
        return r;
    }
};

// ------------------------------------------------------------------- RNG ----
static uint64_t rngState = 0x9e3779b97f4a7c15ULL;
static float frand() {
    rngState ^= rngState >> 12; rngState ^= rngState << 25; rngState ^= rngState >> 27;
    return ((rngState * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

// ================================================================ shaders ===
static const char* MESH_VS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uProj, uView, uModel;
out vec3 vWorldPos;
out vec3 vNormal;
void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vWorldPos = wp.xyz;
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uProj * uView * wp;
}
)";

static const char* MESH_FS = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
uniform vec3  uCamPos;
uniform vec4  uColor;
uniform vec3  uEmissive;
uniform float uShin;
uniform float uSpec;
uniform int   uMatMode;     // 0 standard, 1 grid floor
uniform float uGhost;       // 0..1 memory ghosting
uniform float uFade;
uniform float uTime;
uniform vec3  uFogColor;
uniform float uFogStart, uFogEnd;
uniform float uExposure;
uniform int   uLightCount;
uniform vec3  uLightPos[4];
uniform vec3  uLightColor[4];
out vec4 outColor;

float gridMask(vec2 p, float s) {
    vec2 q = p / s;
    vec2 g = abs(fract(q - 0.5) - 0.5) / fwidth(q);
    return 1.0 - min(min(g.x, g.y), 1.0);
}
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 albedo = uColor.rgb;
    float alpha = uColor.a * uFade;
    vec3 emissive = uEmissive;

    if (uMatMode == 1) {                        // procedural grid floor
        float minor = gridMask(vWorldPos.xz, 1.0);
        float major = gridMask(vWorldPos.xz, 5.0);
        vec3 lineCol = vec3(0.14,0.30,0.42) * minor * 0.5
                     + vec3(0.28,0.60,0.85) * major * 0.8;
        float coreGlow = exp(-dot(vWorldPos.xz, vWorldPos.xz) * 0.055);
        albedo  += lineCol * 0.30;
        emissive += lineCol * (0.35 + coreGlow * 0.75);
    }

    vec3 skyC = vec3(0.16, 0.19, 0.26);         // hemispheric ambient
    vec3 gndC = vec3(0.05, 0.05, 0.06);
    vec3 col = albedo * mix(gndC, skyC, clamp(N.y*0.5+0.5, 0.0, 1.0)) * 1.6;

    for (int i = 0; i < uLightCount; i++) {     // blinn-phong point lights
        vec3  Ld  = uLightPos[i] - vWorldPos;
        float d   = length(Ld);
        vec3  L   = Ld / max(d, 1e-4);
        float att = 1.0 / (1.0 + 0.14*d + 0.05*d*d);
        float ndl = max(dot(N, L), 0.0);
        vec3  H   = normalize(L + V);
        float sp  = pow(max(dot(N, H), 0.0), uShin) * uSpec;
        col += (albedo * ndl + sp) * uLightColor[i] * att;
    }
    col += emissive;

    if (uGhost > 0.001) {                       // desaturate toward memory blue
        float luma = dot(col, vec3(0.299, 0.587, 0.114));
        col = mix(col, vec3(luma) * vec3(0.72, 0.84, 1.05), uGhost * 0.9);
        alpha = mix(alpha, min(alpha, 0.55), uGhost);
    }

    float dist = length(uCamPos - vWorldPos);   // fog
    float fog  = clamp((dist - uFogStart) / (uFogEnd - uFogStart), 0.0, 1.0);
    fog = fog*fog*(3.0 - 2.0*fog);
    col = mix(col, uFogColor, fog);

    col = 1.0 - exp(-col * uExposure);          // filmic-ish tonemap
    col = pow(col, vec3(0.92));
    outColor = vec4(col, alpha);
}
)";

static const char* LINE_VS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aCol;
uniform mat4 uProj, uView, uModel;
out vec3 vCol;
void main() {
    vCol = aCol;
    gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);
}
)";
static const char* LINE_FS = R"(
#version 330 core
in vec3 vCol;
uniform float uFade;
out vec4 outColor;
void main() { outColor = vec4(vCol * uFade, uFade); }
)";

static const char* DUST_VS = R"(
#version 330 core
layout(location=0) in vec3 aSeed;
uniform mat4 uProj, uView;
uniform float uTime;
uniform vec3 uExtent;
out float vAlpha;
void main() {
    float t = uTime;
    vec3 p;
    p.x = mod(aSeed.x*91.7 + t*(0.08 + 0.12*aSeed.y), uExtent.x*2.0) - uExtent.x;
    p.z = mod(aSeed.z*57.3 - t*(0.05 + 0.09*aSeed.x), uExtent.z*2.0) - uExtent.z;
    p.y = 0.2 + aSeed.y*(uExtent.y - 0.4) + sin(t*(0.3 + aSeed.z) + aSeed.x*40.0)*0.3;
    vec4 mv = uView * vec4(p, 1.0);
    gl_Position = uProj * mv;
    float sz = 1.5 + aSeed.z*2.5;
    gl_PointSize = clamp(sz * 26.0 / max(0.5, -mv.z), 1.0, 7.0);
    vAlpha = 0.10 + 0.16*aSeed.y;
}
)";
static const char* DUST_FS = R"(
#version 330 core
in float vAlpha;
out vec4 outColor;
void main() {
    vec2 q = gl_PointCoord*2.0 - 1.0;
    float d = dot(q, q);
    if (d > 1.0) discard;
    float a = (1.0 - d)*(1.0 - d)*vAlpha;
    outColor = vec4(vec3(0.9, 0.85, 0.7)*a, a);
}
)";

static const char* TEXT_VS = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUv;
layout(location=2) in vec4 aCol;
uniform mat4 uProj;
out vec2 vUv; out vec4 vCol;
void main() { vUv = aUv; vCol = aCol; gl_Position = uProj * vec4(aPos, 0.0, 1.0); }
)";
static const char* TEXT_FS = R"(
#version 330 core
in vec2 vUv; in vec4 vCol;
uniform sampler2D uTex;
out vec4 outColor;
void main() { outColor = vec4(vCol.rgb, vCol.a * texture(uTex, vUv).r); }
)";

static const char* FLAT_VS = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uProj;
out vec4 vCol;
void main() { vCol = aCol; gl_Position = uProj * vec4(aPos, 0.0, 1.0); }
)";
static const char* FLAT_FS = R"(
#version 330 core
in vec4 vCol;
out vec4 outColor;
void main() { outColor = vCol; }
)";

// =========================================================== shader utils ===
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log);
        fprintf(stderr, "shader compile error:\n%s\n", log);
    }
    return s;
}
static GLuint linkProgram(const char* vsSrc, const char* fsSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetProgramInfoLog(p, 1024, nullptr, log);
        fprintf(stderr, "program link error:\n%s\n", log);
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}
static GLint U(GLuint p, const char* n) { return glGetUniformLocation(p, n); }

// ================================================================ meshes ====
struct Mesh { GLuint vao = 0, vbo = 0, ebo = 0; int count = 0; };
enum { M_BOX = 0, M_SPHERE, M_CYL, M_DISC, M_RING, M_COUNT };
static Mesh meshes[M_COUNT];

struct MeshBuilder {
    std::vector<float> v;
    std::vector<uint32_t> idx;
    void vert(Vec3 p, Vec3 n) {
        float f[8] = { p.x,p.y,p.z, n.x,n.y,n.z, 0,0 };
        v.insert(v.end(), f, f + 8);
    }
    void quad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        uint32_t q[6] = { a,b,c, a,c,d };
        idx.insert(idx.end(), q, q + 6);
    }
    void tri(uint32_t a, uint32_t b, uint32_t c) {
        idx.push_back(a); idx.push_back(b); idx.push_back(c);
    }
};
static Mesh upload(MeshBuilder& b) {
    Mesh m;
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo); glGenBuffers(1, &m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, b.v.size()*sizeof(float), b.v.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, b.idx.size()*sizeof(uint32_t), b.idx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, (void*)12);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, (void*)24);
    glBindVertexArray(0);
    m.count = (int)b.idx.size();
    return m;
}
static Mesh makeBox() {
    MeshBuilder b; float s = 0.5f;
    auto face = [&](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 n) {
        uint32_t i = (uint32_t)(b.v.size() / 8);
        b.vert(p0,n); b.vert(p1,n); b.vert(p2,n); b.vert(p3,n);
        b.quad(i, i+1, i+2, i+3);
    };
    face({-s,-s, s},{ s,-s, s},{ s, s, s},{-s, s, s},{0,0,1});
    face({ s,-s,-s},{-s,-s,-s},{-s, s,-s},{ s, s,-s},{0,0,-1});
    face({-s, s, s},{ s, s, s},{ s, s,-s},{-s, s,-s},{0,1,0});
    face({-s,-s,-s},{ s,-s,-s},{ s,-s, s},{-s,-s, s},{0,-1,0});
    face({ s,-s, s},{ s,-s,-s},{ s, s,-s},{ s, s, s},{1,0,0});
    face({-s,-s,-s},{-s,-s, s},{-s, s, s},{-s, s,-s},{-1,0,0});
    return upload(b);
}
static Mesh makeSphere() {
    MeshBuilder b; int LAT = 16, LON = 24;
    for (int i = 0; i <= LAT; i++) {
        float th = i / (float)LAT * PI;
        for (int j = 0; j <= LON; j++) {
            float ph = j / (float)LON * TAU;
            Vec3 n(sinf(th)*cosf(ph), cosf(th), sinf(th)*sinf(ph));
            b.vert(n * 0.5f, n);
        }
    }
    for (int i = 0; i < LAT; i++)
        for (int j = 0; j < LON; j++) {
            uint32_t ij = i*(LON+1)+j, ij1 = ij+1;
            uint32_t i1j = (i+1)*(LON+1)+j, i1j1 = i1j+1;
            b.tri(ij, i1j1, i1j);
            b.tri(ij, ij1, i1j1);
        }
    return upload(b);
}
static Mesh makeCylinder() {
    MeshBuilder b; int SEG = 24;
    for (int k = 0; k <= SEG; k++) {
        float a = k / (float)SEG * TAU, x = cosf(a), z = sinf(a);
        b.vert({x*0.5f,-0.5f,z*0.5f}, {x,0,z});
        b.vert({x*0.5f, 0.5f,z*0.5f}, {x,0,z});
    }
    for (int k = 0; k < SEG; k++) {
        uint32_t bk = 2*k, tk = bk+1, bk1 = bk+2, tk1 = bk+3;
        b.tri(bk, tk, bk1);
        b.tri(bk, tk, tk1);
    }
    uint32_t cb = (uint32_t)(b.v.size()/8); b.vert({0,-0.5f,0},{0,-1,0});
    uint32_t rb = cb + 1;
    for (int k = 0; k <= SEG; k++) {
        float a = k / (float)SEG * TAU;
        b.vert({cosf(a)*0.5f,-0.5f,sinf(a)*0.5f},{0,-1,0});
    }
    for (int k = 0; k < SEG; k++) b.tri(cb, rb+k, rb+k+1);
    uint32_t ct = (uint32_t)(b.v.size()/8); b.vert({0,0.5f,0},{0,1,0});
    uint32_t rt = ct + 1;
    for (int k = 0; k <= SEG; k++) {
        float a = k / (float)SEG * TAU;
        b.vert({cosf(a)*0.5f,0.5f,sinf(a)*0.5f},{0,1,0});
    }
    for (int k = 0; k < SEG; k++) b.tri(ct, rt+k+1, rt+k);
    return upload(b);
}
static Mesh makeDisc() {
    MeshBuilder b; int SEG = 36;
    b.vert({0,0,0},{0,1,0});
    for (int k = 0; k <= SEG; k++) {
        float a = k / (float)SEG * TAU;
        b.vert({cosf(a),0,sinf(a)},{0,1,0});
    }
    for (int k = 0; k < SEG; k++) b.tri(0, k+2, k+1);
    return upload(b);
}
static Mesh makeRing() {
    MeshBuilder b; int SEG = 40; float r0 = 0.72f, r1 = 1.0f;
    for (int k = 0; k <= SEG; k++) {
        float a = k / (float)SEG * TAU, x = cosf(a), z = sinf(a);
        b.vert({x*r0,0,z*r0},{0,1,0});
        b.vert({x*r1,0,z*r1},{0,1,0});
    }
    for (int k = 0; k < SEG; k++) {
        uint32_t ik = 2*k, ok = ik+1, ik1 = ik+2, ok1 = ik+3;
        b.tri(ik, ik1, ok1);
        b.tri(ik, ok1, ok);
    }
    return upload(b);
}

// ============================================================ scene graph ===
struct Node {
    Vec3 pos, scl;
    float yaw = 0;
    int mesh = -1;
    Vec4 color;
    Vec3 emissive;
    float shin = 30, spec = 0.15f;
    int matMode = 0;
    float ghost = 0;
    std::vector<Node*> kids;
    Node() : scl(1,1,1), color(1,1,1,1) {}
    ~Node() { for (Node* k : kids) delete k; }
    Mat4 local() const {
        return Mat4::translate(pos) * Mat4::rotY(yaw) * Mat4::scale(scl);
    }
};
static Node* addNode(Node* parent, int mesh, Vec3 pos, float yaw, Vec3 scl, Vec4 color,
                     Vec3 emissive = Vec3(), float shin = 30, float spec = 0.15f,
                     int matMode = 0, float ghost = 0) {
    Node* n = new Node();
    n->mesh = mesh; n->pos = pos; n->yaw = yaw; n->scl = scl;
    n->color = color; n->emissive = emissive; n->shin = shin; n->spec = spec;
    n->matMode = matMode; n->ghost = ghost;
    parent->kids.push_back(n);
    return n;
}

// ============================================================== tesseract ===
static Vec4 tessBase[16];
static int  tessEdges[32][2];
static int  tessEdgeCount = 0;
static void tessInit() {
    for (int i = 0; i < 16; i++)
        tessBase[i] = Vec4((i&1)?0.9f:-0.9f, (i&2)?0.9f:-0.9f,
                           (i&4)?0.9f:-0.9f, (i&8)?0.9f:-0.9f);
    for (int i = 0; i < 16; i++)
        for (int j = i+1; j < 16; j++) {
            int d = i ^ j;
            if (d && !(d & (d-1))) { tessEdges[tessEdgeCount][0]=i; tessEdges[tessEdgeCount][1]=j; tessEdgeCount++; }
        }
}
struct TessAngles { float xw, yz, zw; };
static inline void rotPlane(float& u, float& v, float a) {
    float c = cosf(a), s = sinf(a);
    float nu = c*u - s*v, nv = s*u + c*v;
    u = nu; v = nv;
}
static void tessEval(const TessAngles& A, Vec3 outV[16], float outW[16]) {
    for (int i = 0; i < 16; i++) {
        Vec4 q = tessBase[i];
        rotPlane(q.x, q.w, A.xw);
        rotPlane(q.y, q.z, A.yz);
        rotPlane(q.z, q.w, A.zw);
        float k = TESS_P / (TESS_P - q.w);
        outV[i] = Vec3(q.x*k, q.y*k, q.z*k);
        outW[i] = q.w;
    }
}
static Vec3 tessColor(float w) {
    float t = clamp01(w * (1.0f/1.8f) * 0.5f + 0.5f);
    Vec3 c = mixv(Vec3(0.25f,0.85f,1.0f), Vec3(1.0f,0.42f,0.28f), t);
    return c * (0.55f + 0.6f*t);
}

// ============================================================ bitmap font ===
struct FontGlyph { char ch; unsigned char rows[7]; };
static const FontGlyph FONT_TABLE[] = {
    {' ', {0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
    {'A', {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}},
    {'B', {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
    {'C', {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}},
    {'D', {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
    {'E', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}},
    {'F', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
    {'G', {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}},
    {'H', {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
    {'I', {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}},
    {'K', {0x11,0x12,0x14,0x18,0x14,0x12,0x11}},
    {'L', {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
    {'M', {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}},
    {'N', {0x11,0x19,0x15,0x13,0x11,0x11,0x11}},
    {'O', {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}},
    {'P', {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
    {'Q', {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}},
    {'R', {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
    {'S', {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}},
    {'T', {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
    {'U', {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}},
    {'V', {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
    {'W', {0x11,0x11,0x11,0x15,0x15,0x15,0x0A}},
    {'Y', {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}},
    {'0', {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}},
    {'1', {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
    {'2', {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}},
    {'3', {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}},
    {'4', {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}},
    {'5', {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
    {'6', {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}},
    {'7', {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
    {'8', {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}},
    {'9', {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
    {':', {0x00,0x04,0x04,0x00,0x04,0x04,0x00}},
    {'.', {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}},
    {'-', {0x00,0x00,0x00,0x0E,0x00,0x00,0x00}},
    {'/', {0x01,0x01,0x02,0x04,0x08,0x10,0x10}},
    {'#', {0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A}},
};
static const int FONT_COUNT = (int)(sizeof(FONT_TABLE)/sizeof(FONT_TABLE[0]));
static GLuint fontTex = 0;
static int fontAtlasW = 1;

static void buildFont() {
    int W = FONT_COUNT * 6, H = 8;
    fontAtlasW = W;
    std::vector<unsigned char> px(W*H, 0);
    for (int g = 0; g < FONT_COUNT; g++)
        for (int r = 0; r < 7; r++)
            for (int c = 0; c < 5; c++)
                if (FONT_TABLE[g].rows[r] & (1 << (4 - c)))
                    px[r*W + g*6 + c] = 255;
    glGenTextures(1, &fontTex);
    glBindTexture(GL_TEXTURE_2D, fontTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W, H, 0, GL_RED, GL_UNSIGNED_BYTE, px.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}
static int glyphIndex(char c) {
    for (int i = 0; i < FONT_COUNT; i++) if (FONT_TABLE[i].ch == c) return i;
    return -1;
}

// ================================================================= globals ==
static GLFWwindow* win = nullptr;
static int fbW = 1280, fbH = 800;

struct { GLuint prog; GLint proj,view,model,camPos,color,emissive,shin,spec,matMode,
         ghost,fade,time,fogColor,fogStart,fogEnd,exposure,lightCount,lightPos,lightColor; } UM;
struct { GLuint prog; GLint proj,view,model,fade; } UL;
struct { GLuint prog; GLint proj,tex; } UT;
struct { GLuint prog; GLint proj; } UF;
struct { GLuint prog; GLint proj,view,time,extent; } UD;

static GLuint lineVAO = 0, lineVBO = 0;
static GLuint dustVAO = 0, dustVBO = 0;
static const int DUST_N = 160;

static GLuint textVBO = 0;  static std::vector<float> textVerts;
static GLuint flatVBO = 0;  static std::vector<float> flatVerts;

// simulation state
static float simTime = 0;
static int   gen = 1;
static bool  autoMode = true;
static float distAcc = 0;
static float lastMemoryAt = -10;
static uint32_t roomSeed = 1;
static Node* roomRoot = nullptr;
static Node* pulseRing = nullptr;
static Vec3 autoTarget(6, 0, 0);

struct Camera { Vec3 pos; float yaw = 0, pitch = -0.06f; };
static Camera cam;
static float bobT = 0, bobAmt = 0;
static double lastMX = 0, lastMY = 0; static bool hasLastM = false;
static int fps = 60;

struct Memory {
    Node* root = nullptr;
    TessAngles ang;
    Vec3 tessPos; float tessScale = 0.34f;
    Vec3 eye; float yaw = 0;
    float born = 0, die = -1;
    int genId = 0;
    Node* visor = nullptr;
    Node* candle = nullptr;
};
static std::deque<Memory*> memories;
struct StepRing { Vec3 pos; float t0; };
static std::vector<StepRing> stepRings;

static const Vec3 TESS_CENTER(0, 2.3f, 0);

// ============================================================ room builder ==
static void generateRoom(uint32_t seed) {
    if (roomRoot) { delete roomRoot; pulseRing = nullptr; }
    roomSeed = seed;
    rngState = ((uint64_t)seed << 1) | 1ULL;
    roomRoot = new Node();
    float H = ROOM_HALF, W = 2*H + 1.2f;
    Vec4 wallCol(0.09f, 0.095f, 0.12f, 1);

    addNode(roomRoot, M_BOX, Vec3(0,-0.16f,0), 0, Vec3(W,0.32f,W),
            Vec4(0.075f,0.08f,0.10f,1), Vec3(), 70, 0.35f, 1);          // grid floor
    addNode(roomRoot, M_BOX, Vec3(0,2.2f, H+0.15f), 0, Vec3(W,4.4f,0.3f), wallCol);
    addNode(roomRoot, M_BOX, Vec3(0,2.2f,-(H+0.15f)), 0, Vec3(W,4.4f,0.3f), wallCol);
    addNode(roomRoot, M_BOX, Vec3( H+0.15f,2.2f,0), 0, Vec3(0.3f,4.4f,W), wallCol);
    addNode(roomRoot, M_BOX, Vec3(-(H+0.15f),2.2f,0), 0, Vec3(0.3f,4.4f,W), wallCol);
    addNode(roomRoot, M_BOX, Vec3(0,4.56f,0), 0, Vec3(W,0.3f,W), Vec4(0.045f,0.05f,0.065f,1));

    int np = 6 + (int)(frand()*3);                                      // pillars
    for (int i = 0; i < np; i++) {
        float a = (i/(float)np)*TAU + frand()*0.5f;
        float r = H - 2.6f + frand()*1.2f;
        Vec3 p(cosf(a)*r, 0, sinf(a)*r);
        addNode(roomRoot, M_CYL, p+Vec3(0,2.2f,0), 0, Vec3(0.55f,4.4f,0.55f),
                Vec4(0.115f,0.12f,0.15f,1), Vec3(), 40, 0.15f);
        addNode(roomRoot, M_BOX, p+Vec3(0,0.12f,0), 0, Vec3(0.95f,0.24f,0.95f),
                Vec4(0.13f,0.13f,0.16f,1));
    }
    for (int i = 0; i < 6; i++) {                                       // ceiling strips
        float x = (frand()*2-1)*(H-3), z = (frand()*2-1)*(H-3);
        float yaw = frand() < 0.5f ? 0 : PI/2;
        Vec3 em(1.35f,1.25f,1.05f); em = em * (0.7f + frand()*0.5f);
        addNode(roomRoot, M_BOX, Vec3(x,4.38f,z), yaw, Vec3(2.6f,0.07f,0.2f),
                Vec4(0.05f,0.05f,0.05f,1), em);
    }
    for (int i = 0; i < 10; i++) {                                      // wall sconces
        int wall = (int)(frand()*4);
        float t = (frand()*2-1)*(H-1.5f), hgt = 2.3f + frand()*1.2f;
        Vec3 p;
        if      (wall == 0) p = Vec3( t,      hgt,  (H-0.05f));
        else if (wall == 1) p = Vec3( t,      hgt, -(H-0.05f));
        else if (wall == 2) p = Vec3( (H-0.05f), hgt,  t);
        else                p = Vec3(-(H-0.05f), hgt,  t);
        Vec3 em = frand() < 0.5f ? Vec3(1.0f,0.62f,0.35f) : Vec3(0.45f,0.75f,1.0f);
        addNode(roomRoot, M_BOX, p, 0, Vec3(0.12f,0.5f,0.06f),
                Vec4(0.04f,0.04f,0.05f,1), em * 1.4f);
    }
    addNode(roomRoot, M_CYL, Vec3(0,0.26f,0), 0, Vec3(2.6f,0.52f,2.6f),  // pedestal
            Vec4(0.13f,0.13f,0.17f,1), Vec3(), 60, 0.4f);
    pulseRing = addNode(roomRoot, M_RING, Vec3(0,0.54f,0), 0, Vec3(1.9f,1,1.9f),
            Vec4(0.03f,0.06f,0.09f,1), Vec3(0.2f,0.85f,1.1f), 30, 0.2f);
}

// ================================================================ memories ==
static TessAngles liveAngles() {
    return { simTime*0.43f, simTime*0.29f + sinf(simTime*0.21f)*0.5f, simTime*0.17f };
}
static void spawnMemory() {
    Memory* m = new Memory();
    m->born = simTime; m->die = -1; m->genId = gen++;
    m->ang = liveAngles(); m->eye = cam.pos; m->yaw = cam.yaw;
    Vec3 fwd(sinf(cam.yaw), 0, -cosf(cam.yaw));
    Vec3 P(cam.pos.x, 0, cam.pos.z);
    Node* root = new Node();
    m->root = root;
    // platform + glow rim
    addNode(root, M_DISC, P+Vec3(0,0.02f,0), 0, Vec3(1.05f,1,1.05f),
            Vec4(0.09f,0.10f,0.14f,1), Vec3(0.02f,0.04f,0.08f), 40, 0.2f, 0, 0.7f);
    addNode(root, M_RING, P+Vec3(0,0.035f,0), 0, Vec3(1.05f,1,1.05f),
            Vec4(0.06f,0.12f,0.2f,1), Vec3(0.12f,0.44f,0.76f), 30, 0.2f, 0, 0.75f);
    // pedestal + candle holding the frozen mini-tesseract
    Vec3 pd = P + fwd*0.62f;
    addNode(root, M_CYL, pd+Vec3(0,0.28f,0), 0, Vec3(0.5f,0.56f,0.5f),
            Vec4(0.12f,0.12f,0.17f,1), Vec3(), 50, 0.3f, 0, 0.75f);
    m->candle = addNode(root, M_SPHERE, pd+Vec3(0,0.66f,0), 0, Vec3(0.16f,0.16f,0.16f),
            Vec4(1,0.85f,0.6f,0.95f), Vec3(1.5f,0.95f,0.5f), 30, 0, 0, 0.5f);
    // ghost of the previous observer body
    addNode(root, M_CYL, P+Vec3(0,0.70f,0), cam.yaw, Vec3(0.34f,1.30f,0.34f),
            Vec4(0.55f,0.7f,0.95f,0.30f), Vec3(), 60, 0.5f, 0, 1.0f);
    addNode(root, M_SPHERE, P+Vec3(0,1.52f,0), 0, Vec3(0.34f,0.34f,0.34f),
            Vec4(0.55f,0.7f,0.95f,0.30f), Vec3(), 60, 0.5f, 0, 1.0f);
    m->visor = addNode(root, M_SPHERE, P+fwd*0.14f+Vec3(0,1.55f,0), 0,
            Vec3(0.13f,0.13f,0.13f), Vec4(0.9f,0.97f,1.0f,0.9f),
            Vec3(0.6f,1.0f,1.4f), 30, 0, 0, 0.4f);
    m->tessPos = pd + Vec3(0,1.42f,0); m->tessScale = 0.34f;
    memories.push_back(m);
    lastMemoryAt = simTime;
    stepRings.push_back({ P, simTime });
    // cap: begin dissolving oldest living memories
    int alive = 0;
    for (auto q : memories) if (q->die < 0) alive++;
    for (auto q : memories) {
        if (alive <= MAX_MEMORIES) break;
        if (q->die < 0) { q->die = simTime; alive--; }
    }
}
static float memFade(const Memory* m) {
    float fi = clamp01((simTime - m->born) / 0.5f);
    if (m->die > 0) fi *= clamp01(1.0f - (simTime - m->die) / MEM_FADE);
    return fi;
}
static void updateMemories() {
    for (auto it = memories.begin(); it != memories.end(); ) {
        Memory* m = *it;
        if (m->die > 0 && simTime - m->die > MEM_FADE) {
            delete m->root; delete m;
            it = memories.erase(it);
            continue;
        }
        float fi = memFade(m);
        float sc = 0.7f + 0.3f*fi;
        m->root->scl = Vec3(sc, sc, sc);
        if (m->visor) {
            float hb = 0.6f + 0.5f*sinf(simTime*2.6f + m->genId*1.7f);
            m->visor->emissive = Vec3(0.55f,0.9f,1.3f)*hb;
        }
        if (m->candle) {
            float fl = 0.75f + 0.25f*sinf(simTime*9 + m->genId*3.1f)
                             * sinf(simTime*5.7f + m->genId);
            m->candle->emissive = Vec3(1.5f,0.95f,0.5f)*fl;
        }
        ++it;
    }
}
static void clearMemories() {
    for (auto m : memories) { delete m->root; delete m; }
    memories.clear();
    stepRings.clear();
}

// ================================================================ movement ==
static Vec3 eyePos() { return cam.pos + Vec3(0, sinf(bobT)*0.05f*bobAmt, 0); }
static void moveCam(Vec3 delta) {
    Vec3 np = cam.pos + delta; np.y = EYE_HEIGHT;
    float lim = ROOM_HALF - 0.7f;
    np.x = clampf(np.x, -lim, lim);
    np.z = clampf(np.z, -lim, lim);
    distAcc += length(np - cam.pos);
    cam.pos = np;
    if (distAcc >= STEP_DIST) { spawnMemory(); distAcc = 0; }
}
static void pickAutoTarget() {
    float r = (frand() < 0.35f) ? 2.0f + frand()*1.5f
                                : 4.5f + frand()*(ROOM_HALF - 5.5f);
    float a = frand()*TAU;
    autoTarget = Vec3(cosf(a)*r, 0, sinf(a)*r);
}
static void updateAuto(float dt) {
    Vec3 p(cam.pos.x, 0, cam.pos.z);
    Vec3 d = autoTarget - p;
    float dist = length(d);
    if (dist < 0.7f) pickAutoTarget();
    if (dist < 1e-4f) return;
    Vec3 dir = d / dist;
    moveCam(dir * (2.3f * dt));
    float desired = atan2f(dir.x, -dir.z);
    cam.yaw = lerpAngle(cam.yaw, desired, 1.0f - expf(-3.5f*dt));
    Vec3 tc = TESS_CENTER - eyePos();
    float hd = sqrtf(tc.x*tc.x + tc.z*tc.z);
    float dp = atan2f(tc.y, hd > 0.01f ? hd : 0.01f);
    cam.pitch = lerpf(cam.pitch, dp*0.5f, 1.0f - expf(-2.0f*dt));
}
static void updatePlayer(float dt) {
    bool kF = glfwGetKey(win, GLFW_KEY_W) || glfwGetKey(win, GLFW_KEY_UP);
    bool kB = glfwGetKey(win, GLFW_KEY_S) || glfwGetKey(win, GLFW_KEY_DOWN);
    bool kL = glfwGetKey(win, GLFW_KEY_A) || glfwGetKey(win, GLFW_KEY_LEFT);
    bool kR = glfwGetKey(win, GLFW_KEY_D) || glfwGetKey(win, GLFW_KEY_RIGHT);
    if (kF || kB || kL || kR) autoMode = false;
    if (autoMode) { updateAuto(dt); return; }
    Vec3 fwd(sinf(cam.yaw), 0, -cosf(cam.yaw));
    Vec3 rgt(cosf(cam.yaw), 0,  sinf(cam.yaw));
    Vec3 mv;
    if (kF) mv += fwd; if (kB) mv -= fwd;
    if (kR) mv += rgt; if (kL) mv -= rgt;
    float sp = glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) ? 4.6f : 3.0f;
    float moved = 0;
    if (length(mv) > 1e-4f) {
        mv = normalize(mv);
        moveCam(mv * (sp * dt));
        moved = sp * dt;
    }
    bobAmt = lerpf(bobAmt, moved > 0 ? 1.0f : 0.0f, 1.0f - expf(-8.0f*dt));
    bobT += dt * 9.0f * bobAmt;
}

// ================================================================ callbacks =
static void fbSizeCb(GLFWwindow*, int w, int h) { fbW = w; fbH = h; }
static void keyCb(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    switch (key) {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); break;
    case GLFW_KEY_SPACE:  spawnMemory(); break;
    case GLFW_KEY_T:      autoMode = !autoMode; if (autoMode) pickAutoTarget(); break;
    case GLFW_KEY_R:
        clearMemories(); gen = 1; distAcc = 0;
        cam.pos = Vec3(0, EYE_HEIGHT, 9.5f); cam.yaw = 0; cam.pitch = -0.06f;
        autoMode = true; pickAutoTarget(); lastMemoryAt = -10;
        break;
    case GLFW_KEY_N: generateRoom(roomSeed * 1664525u + 1013904223u); break;
    case GLFW_KEY_F: {
        static bool fs = false; static int wx, wy, ww, wh;
        if (!fs) {
            glfwGetWindowPos(win, &wx, &wy); glfwGetWindowSize(win, &ww, &wh);
            GLFWmonitor* mon = glfwGetPrimaryMonitor();
            const GLFWvidmode* vm = glfwGetVideoMode(mon);
            glfwSetWindowMonitor(win, mon, 0, 0, vm->width, vm->height, vm->refreshRate);
        } else {
            glfwSetWindowMonitor(win, nullptr, wx, wy, ww, wh, 0);
        }
        fs = !fs;
        break;
    }
    }
}
static void cursorCb(GLFWwindow*, double x, double y) {
    if (!hasLastM) { lastMX = x; lastMY = y; hasLastM = true; return; }
    float dx = (float)(x - lastMX), dy = (float)(y - lastMY);
    lastMX = x; lastMY = y;
    if (autoMode && fabsf(dx) + fabsf(dy) > 2.0f) autoMode = false;
    cam.yaw   += dx * 0.0022f;
    cam.pitch -= dy * 0.0022f;
    cam.pitch = clampf(cam.pitch, -1.45f, 1.45f);
}

// ================================================================ rendering =
static void setMeshDraw(const Mesh& m, const Mat4& model, Vec4 color, Vec3 emissive,
                        float shin, float spec, int matMode, float ghost, float fade) {
    glUniformMatrix4fv(UM.model, 1, GL_FALSE, model.m);
    color.w *= fade;
    glUniform4fv(UM.color, 1, &color.x);
    glUniform3fv(UM.emissive, 1, &emissive.x);
    glUniform1f(UM.shin, shin);
    glUniform1f(UM.spec, spec);
    glUniform1i(UM.matMode, matMode);
    glUniform1f(UM.ghost, ghost);
    glUniform1f(UM.fade, fade);
    glBindVertexArray(m.vao);
    glDrawElements(GL_TRIANGLES, m.count, GL_UNSIGNED_INT, nullptr);
}
static void drawMeshOnce(int meshId, const Mat4& model, Vec4 color, Vec3 emissive,
                         float shin, float spec, int matMode, float ghost, float fade) {
    setMeshDraw(meshes[meshId], model, color, emissive, shin, spec, matMode, ghost, fade);
}
static void renderGraph(Node* n, const Mat4& parent, float fadeMul, int pass) {
    if (!n) return;
    Mat4 w = parent * n->local();
    bool tr = n->ghost > 0.001f || n->color.w < 0.99f || fadeMul < 0.99f;
    if (n->mesh >= 0 && ((pass == 1) == tr))
        setMeshDraw(meshes[n->mesh], w, n->color, n->emissive,
                    n->shin, n->spec, n->matMode, n->ghost, fadeMul);
    for (Node* k : n->kids) renderGraph(k, w, fadeMul, pass);
}
static void drawTessLines(const TessAngles& A, Vec3 center, float s, float fade, float width) {
    Vec3 v[16]; float w[16];
    tessEval(A, v, w);
    float buf[32*2*6]; int p = 0;
    for (int e = 0; e < tessEdgeCount; e++)
        for (int k = 0; k < 2; k++) {
            int i = tessEdges[e][k];
            buf[p++]=v[i].x; buf[p++]=v[i].y; buf[p++]=v[i].z;
            Vec3 c = tessColor(w[i]);
            buf[p++]=c.x; buf[p++]=c.y; buf[p++]=c.z;
        }
    Mat4 model = Mat4::translate(center) * Mat4::scale(s, s, s);
    glUniformMatrix4fv(UL.model, 1, GL_FALSE, model.m);
    glUniform1f(UL.fade, fade);
    glLineWidth(width);
    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(buf), buf, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, tessEdgeCount * 2);
    glLineWidth(1.0f);
}
static void drawDust(const Mat4& proj, const Mat4& view) {
    glUseProgram(UD.prog);
    glUniformMatrix4fv(UD.proj, 1, GL_FALSE, proj.m);
    glUniformMatrix4fv(UD.view, 1, GL_FALSE, view.m);
    glUniform1f(UD.time, simTime);
    Vec3 ext(ROOM_HALF, 4.3f, ROOM_HALF);
    glUniform3fv(UD.extent, 1, &ext.x);
    glBindVertexArray(dustVAO);
    glDrawArrays(GL_POINTS, 0, DUST_N);
}

// ------------------------------------------------------------------- HUD ---
static void textQuad(float x, float y, float w, float h,
                     float u0, float v0, float u1, float v1, Vec4 c) {
    float q[6][8] = {
        {x,   y,   u0,v0, c.x,c.y,c.z,c.w},
        {x+w, y,   u1,v0, c.x,c.y,c.z,c.w},
        {x+w, y+h, u1,v1, c.x,c.y,c.z,c.w},
        {x,   y,   u0,v0, c.x,c.y,c.z,c.w},
        {x+w, y+h, u1,v1, c.x,c.y,c.z,c.w},
        {x,   y+h, u0,v1, c.x,c.y,c.z,c.w},
    };
    for (int i = 0; i < 6; i++) textVerts.insert(textVerts.end(), q[i], q[i]+8);
}
static void drawText(float x, float y, float s, Vec4 col, const char* str) {
    for (const char* c = str; *c; c++) {
        if (*c == ' ') { x += 6*s; continue; }
        int gi = glyphIndex(*c);
        if (gi < 0) { x += 6*s; continue; }
        float u0 = (gi*6) / (float)fontAtlasW;
        float u1 = (gi*6 + 5) / (float)fontAtlasW;
        textQuad(x, y, 5*s, 7*s, u0, 0.0f, u1, 7.0f/8.0f, col);
        x += 6*s;
    }
}
static float textWidth(const char* s, float sc) { return (float)strlen(s) * 6 * sc; }
static void textFlush(const Mat4& ortho) {
    if (textVerts.empty()) return;
    glUseProgram(UT.prog);
    glUniformMatrix4fv(UT.proj, 1, GL_FALSE, ortho.m);
    glUniform1i(UT.tex, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fontTex);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, textVerts.size()*sizeof(float), textVerts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, (void*)8);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, (void*)16);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(textVerts.size()/8));
    glDisableVertexAttribArray(0); glDisableVertexAttribArray(1); glDisableVertexAttribArray(2);
    textVerts.clear();
}
static void flatTri(float x0,float y0,float x1,float y1,float x2,float y2, Vec4 c) {
    float q[3][6] = { {x0,y0,c.x,c.y,c.z,c.w}, {x1,y1,c.x,c.y,c.z,c.w}, {x2,y2,c.x,c.y,c.z,c.w} };
    for (int i = 0; i < 3; i++) flatVerts.insert(flatVerts.end(), q[i], q[i]+6);
}
static void flatRect(float x, float y, float w, float h, Vec4 c) {
    flatTri(x,   y,   x+w, y,   x+w, y+h, c);
    flatTri(x,   y,   x+w, y+h, x,   y+h, c);
}
static void flatFlush(const Mat4& ortho) {
    if (flatVerts.empty()) return;
    glUseProgram(UF.prog);
    glUniformMatrix4fv(UF.proj, 1, GL_FALSE, ortho.m);
    glBindBuffer(GL_ARRAY_BUFFER, flatVBO);
    glBufferData(GL_ARRAY_BUFFER, flatVerts.size()*sizeof(float), flatVerts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, (void*)8);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(flatVerts.size()/6));
    glDisableVertexAttribArray(0); glDisableVertexAttribArray(1);
    flatVerts.clear();
}
static void drawHUD() {
    Mat4 ortho = Mat4::ortho(0, (float)fbW, (float)fbH, 0, -1, 1);
    char buf[160];
    Vec4 cyan(0.40f,0.90f,1.0f,0.95f), lab(0.72f,0.78f,0.85f,0.9f);
    float y = 16;
    drawText(16, y, 2, cyan, "TESSERACT LIFE SIM"); y += 28;
    snprintf(buf, sizeof(buf), "OBSERVER %04d   STEPS %04d", gen, gen-1);
    drawText(16, y, 2, lab, buf); y += 20;
    int alive = 0; for (auto m : memories) if (m->die < 0) alive++;
    int mm = (int)simTime / 60, ss = (int)simTime % 60, dsec = (int)(simTime*10) % 10;
    snprintf(buf, sizeof(buf), "MEMORIES %02d/%02d   TIME %02d:%02d.%d",
             alive, MAX_MEMORIES, mm, ss, dsec);
    drawText(16, y, 2, lab, buf); y += 20;
    snprintf(buf, sizeof(buf), "MODE %s   SEED %u   FPS %d",
             autoMode ? "AUTO" : "MANUAL", roomSeed, fps);
    drawText(16, y, 2, lab, buf);
    // controls hint
    snprintf(buf, sizeof(buf),
        "WASD MOVE  MOUSE LOOK  SHIFT RUN  SPACE MEMORY  T AUTO  N ROOM  R RESET  F FULL  ESC QUIT");
    drawText((fbW - textWidth(buf, 2))*0.5f, (float)fbH - 28, 2, Vec4(0.5f,0.56f,0.64f,0.75f), buf);
    // compass distance to core
    Vec3 eye = eyePos();
    Vec3 toCore(-eye.x, 0, -eye.z);
    float coreDist = length(toCore);
    snprintf(buf, sizeof(buf), "CORE %d.%dM", (int)coreDist, ((int)(coreDist*10))%10);
    drawText((fbW - textWidth(buf, 2))*0.5f, (float)fbH - 74, 2, lab, buf);
    // memory archived event
    float e = simTime - lastMemoryAt;
    if (e < 1.8f) {
        float a = clamp01(1.8f - e);
        snprintf(buf, sizeof(buf), "MEMORY %04d ARCHIVED", gen - 1);
        drawText((fbW - textWidth(buf, 3))*0.5f, fbH*0.30f, 3,
                 Vec4(0.5f, 0.95f, 1.0f, a*0.9f), buf);
    }
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    textFlush(ortho);
    // crosshair
    float cx = fbW*0.5f, cy = fbH*0.5f;
    Vec4 ch(0.9f, 0.95f, 1.0f, 0.85f);
    flatRect(cx-1.5f, cy-1.5f, 3, 3, ch);
    flatRect(cx-0.5f, cy-12, 1, 6, ch); flatRect(cx-0.5f, cy+6, 1, 6, ch);
    flatRect(cx-12, cy-0.5f, 6, 1, ch); flatRect(cx+6, cy-0.5f, 6, 1, ch);
    // bearing arrow to core
    Vec3 fwd(sinf(cam.yaw), 0, -cosf(cam.yaw)), rgt(cosf(cam.yaw), 0, sinf(cam.yaw));
    float dn = dot(toCore, fwd), dr = dot(toCore, rgt);
    float bear = atan2f(dr, dn);
    float facing = clamp01(1.0f - fabsf(bear)/PI);
    Vec4 ac = mixv(Vec4(0.5f,0.55f,0.6f,0.8f), Vec4(0.3f,0.95f,1.0f,0.95f), facing);
    float cb = cosf(bear), sb = sinf(bear);
    float ax = fbW*0.5f, ay = fbH - 100.0f;
    float px[3] = { 0, 8, -8 }, py[3] = { -13, 9, 9 };
    float rx[3], ry[3];
    for (int i = 0; i < 3; i++) { rx[i] = ax + px[i]*cb - py[i]*sb; ry[i] = ay + px[i]*sb + py[i]*cb; }
    flatTri(rx[0],ry[0], rx[1],ry[1], rx[2],ry[2], ac);
    flatFlush(ortho);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

// ------------------------------------------------------------------ frame --
static void render() {
    glViewport(0, 0, fbW, fbH);
    Vec3 fogCol(0.030f, 0.035f, 0.052f);
    glClearColor(fogCol.x, fogCol.y, fogCol.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Vec3 eye = eyePos();
    Mat4 proj = Mat4::perspective(62.0f, fbW / (float)fbH, 0.1f, 60.0f);
    Mat4 view = Mat4::rotX(-cam.pitch) * Mat4::rotY(cam.yaw) * Mat4::translate(-eye);

    // lights (index 3 is the observer's lantern)
    Vec3 lPos[4] = { Vec3(0,4.2f,0), Vec3(8.6f,3.0f,-8.6f), Vec3(-8.6f,2.6f,8.6f), eye };
    Vec3 lCol[4] = { Vec3(3.4f,3.1f,2.65f), Vec3(0.7f,1.5f,2.0f),
                     Vec3(1.7f,0.85f,0.55f), Vec3(0.95f,0.75f,0.5f) };

    glUseProgram(UM.prog);
    glUniformMatrix4fv(UM.proj, 1, GL_FALSE, proj.m);
    glUniformMatrix4fv(UM.view, 1, GL_FALSE, view.m);
    glUniform3fv(UM.camPos, 1, &eye.x);
    glUniform3fv(UM.fogColor, 1, &fogCol.x);
    glUniform1f(UM.fogStart, 8.0f); glUniform1f(UM.fogEnd, 34.0f);
    glUniform1f(UM.time, simTime);
    float flash = expf(-(simTime - lastMemoryAt) * 7.0f);
    glUniform1f(UM.exposure, 1.7f * (1.0f + 0.3f*flash));
    glUniform1i(UM.lightCount, 4);
    glUniform3fv(UM.lightPos, 4, &lPos[0].x);
    glUniform3fv(UM.lightColor, 4, &lCol[0].x);

    Mat4 IDENT = Mat4::identity();

    // ---- pass 0: opaque
    glDisable(GL_BLEND); glDepthMask(GL_TRUE);
    if (pulseRing)
        pulseRing->emissive = Vec3(0.2f,0.85f,1.1f) * (0.75f + 0.45f*sinf(simTime*1.6f));
    renderGraph(roomRoot, IDENT, 1.0f, 0);
    for (auto m : memories) renderGraph(m->root, IDENT, memFade(m), 0);
    // vertex orbs of the living tesseract
    Vec3 ov[16]; float ow[16];
    TessAngles live = liveAngles();
    tessEval(live, ov, ow);
    for (int i = 0; i < 16; i++) {
        float t = clamp01(ow[i]*(1/1.8f)*0.5f + 0.5f);
        Vec3 c = tessColor(ow[i]);
        float r = (0.05f + 0.055f*t) * TESS_SCALE;
        drawMeshOnce(M_SPHERE,
            Mat4::translate(TESS_CENTER + ov[i]*TESS_SCALE) * Mat4::scale(2*r, 2*r, 2*r),
            Vec4(c, 1), c*1.4f, 50, 0.6f, 0, 0, 1);
    }

    // ---- pass 1: transparent
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    renderGraph(roomRoot, IDENT, 1.0f, 1);
    for (auto m : memories) renderGraph(m->root, IDENT, memFade(m), 1);
    // presence ring under the current observer
    drawMeshOnce(M_RING,
        Mat4::translate(Vec3(eye.x, 0.04f, eye.z)) * Mat4::scale(0.85f, 1, 0.85f),
        Vec4(0.04f,0.09f,0.13f,0.55f), Vec3(0.2f,0.85f,1.1f)*(0.4f+0.15f*sinf(simTime*2.2f)),
        30, 0.2f, 0, 0, 1);
    // expanding step pulses
    for (size_t i = 0; i < stepRings.size(); ) {
        float p = (simTime - stepRings[i].t0) / 0.9f;
        if (p >= 1.0f) { stepRings.erase(stepRings.begin() + i); continue; }
        float s = 0.4f + 3.4f*p;
        drawMeshOnce(M_RING,
            Mat4::translate(stepRings[i].pos + Vec3(0,0.05f,0)) * Mat4::scale(s,1,s),
            Vec4(0.04f,0.1f,0.15f,(1-p)*0.6f), Vec3(0.3f,1.1f,1.5f)*(1-p)*1.3f,
            30, 0.2f, 0, 0, 1);
        i++;
    }

    // ---- additive: tesseract edges + dust
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glUseProgram(UL.prog);
    glUniformMatrix4fv(UL.proj, 1, GL_FALSE, proj.m);
    glUniformMatrix4fv(UL.view, 1, GL_FALSE, view.m);
    drawTessLines(live, TESS_CENTER, TESS_SCALE, 0.35f, 5.0f);  // soft glow
    drawTessLines(live, TESS_CENTER, TESS_SCALE, 1.0f, 2.0f);
    for (auto m : memories)
        drawTessLines(m->ang, m->tessPos, m->tessScale, memFade(m)*0.85f, 1.5f);
    drawDust(proj, view);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    drawHUD();
}

// ==================================================================== main ==
int main() {
    if (!glfwInit()) { fprintf(stderr, "glfw init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    win = glfwCreateWindow(1280, 800, "TESSERACT // OBSERVER LIFE SIM", nullptr, nullptr);
    if (!win) { fprintf(stderr, "window creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSetFramebufferSizeCallback(win, fbSizeCb);
    glfwSetKeyCallback(win, keyCb);
    glfwSetCursorPosCallback(win, cursorCb);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwGetFramebufferSize(win, &fbW, &fbH);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "glew init failed\n"); return 1; }
    glGetError(); // swallow dummy error some drivers raise with core profiles

    UM.prog = linkProgram(MESH_VS, MESH_FS);
    UM.proj=U(UM.prog,"uProj"); UM.view=U(UM.prog,"uView"); UM.model=U(UM.prog,"uModel");
    UM.camPos=U(UM.prog,"uCamPos"); UM.color=U(UM.prog,"uColor");
    UM.emissive=U(UM.prog,"uEmissive"); UM.shin=U(UM.prog,"uShin"); UM.spec=U(UM.prog,"uSpec");
    UM.matMode=U(UM.prog,"uMatMode"); UM.ghost=U(UM.prog,"uGhost"); UM.fade=U(UM.prog,"uFade");
    UM.time=U(UM.prog,"uTime"); UM.fogColor=U(UM.prog,"uFogColor");
    UM.fogStart=U(UM.prog,"uFogStart"); UM.fogEnd=U(UM.prog,"uFogEnd");
    UM.exposure=U(UM.prog,"uExposure"); UM.lightCount=U(UM.prog,"uLightCount");
    UM.lightPos=U(UM.prog,"uLightPos"); UM.lightColor=U(UM.prog,"uLightColor");
    UL.prog = linkProgram(LINE_VS, LINE_FS);
    UL.proj=U(UL.prog,"uProj"); UL.view=U(UL.prog,"uView");
    UL.model=U(UL.prog,"uModel"); UL.fade=U(UL.prog,"uFade");
    UT.prog = linkProgram(TEXT_VS, TEXT_FS);
    UT.proj=U(UT.prog,"uProj"); UT.tex=U(UT.prog,"uTex");
    UF.prog = linkProgram(FLAT_VS, FLAT_FS);
    UF.proj=U(UF.prog,"uProj");
    UD.prog = linkProgram(DUST_VS, DUST_FS);
    UD.proj=U(UD.prog,"uProj"); UD.view=U(UD.prog,"uView");
    UD.time=U(UD.prog,"uTime"); UD.extent=U(UD.prog,"uExtent");

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);

    meshes[M_BOX]    = makeBox();
    meshes[M_SPHERE] = makeSphere();
    meshes[M_CYL]    = makeCylinder();
    meshes[M_DISC]   = makeDisc();
    meshes[M_RING]   = makeRing();
    tessInit();
    buildFont();

    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);
    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, 32*2*6*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void*)12);
    glBindVertexArray(0);

    glGenVertexArrays(1, &dustVAO);
    glGenBuffers(1, &dustVBO);
    std::vector<float> seeds;
    for (int i = 0; i < DUST_N*3; i++) seeds.push_back(frand());
    glBindVertexArray(dustVAO);
    glBindBuffer(GL_ARRAY_BUFFER, dustVBO);
    glBufferData(GL_ARRAY_BUFFER, seeds.size()*sizeof(float), seeds.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void*)0);
    glBindVertexArray(0);

    glGenBuffers(1, &textVBO);
    glGenBuffers(1, &flatVBO);

    generateRoom((uint32_t)time(nullptr) & 0xffffffu);
    cam.pos = Vec3(0, EYE_HEIGHT, 9.5f);
    cam.yaw = 0; cam.pitch = -0.06f;
    pickAutoTarget();

    double lastT = glfwGetTime(), titleT = 0, fpsT = lastT;
    int frames = 0;
    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dt = (float)(now - lastT);
        lastT = now;
        if (dt > 0.05f) dt = 0.05f;
        simTime += dt;

        updatePlayer(dt);
        updateMemories();
        render();

        frames++;
        if (now - fpsT > 0.5) { fps = (int)(frames / (now - fpsT)); frames = 0; fpsT = now; }
        if (now - titleT > 0.5) {
            char t[160];
            int alive = 0; for (auto m : memories) if (m->die < 0) alive++;
            snprintf(t, sizeof(t), "TESSERACT LIFE SIM - OBSERVER %04d  MEMORIES %02d  %d FPS",
                     gen, alive, fps);
            glfwSetWindowTitle(win, t);
            titleT = now;
        }
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    clearMemories();
    if (roomRoot) delete roomRoot;
    glfwTerminate();
    return 0;
}