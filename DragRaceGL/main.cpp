/* ============================================================================
   DRAG RACER — single-file OpenGL 4.1 drag racing game
   Modes : (1) Player vs Computer, (2) Two players on one keyboard
   Deps  : GLEW + GLFW only. No GLM (hand-rolled math). No audio.

   Build
   -----
   Linux   : g++ drag_racer.cpp -std=c++11 -O2 -o drag_racer -lGLEW -lglfw -lGL
   macOS   : clang++ drag_racer.cpp -std=c++11 -O2 -o drag_racer -lGLEW -lglfw -framework OpenGL
   Windows : cl /EHsc /O2 drag_racer.cpp /link glew32.lib glfw3.lib opengl32.lib

   Controls
   --------
   P1 : W = throttle, S = brake, E or Left-Shift = shift up, Q = shift down
   P2 : Up = throttle, Down = brake, . or Right-Shift = shift up, , = shift down
   1 = race the CPU, 2 = two players, R = restart, ESC = menu

   Rules : launch exactly on green (throttle before green = red light foul).
   Keep the revs near the top of the band and shift when "SHIFT!" flashes.
   First legal car to the 1/4 mile (402.336 m) line wins.
   ========================================================================== */

#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>

/* ============================================================================
   1. TINY MATH (column-major Mat4, OpenGL convention)
   ========================================================================== */
struct Vec3 { float x, y, z; };
static Vec3 v3(float x, float y, float z) { Vec3 v = {x, y, z}; return v; }
static Vec3 operator+(Vec3 a, Vec3 b){ return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static Vec3 operator-(Vec3 a, Vec3 b){ return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static Vec3 operator*(Vec3 a, float s){ return v3(a.x*s, a.y*s, a.z*s); }
static float dot(Vec3 a, Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3 cross(Vec3 a, Vec3 b){
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static Vec3 norm3(Vec3 a){ float l = sqrtf(dot(a,a)); if (l < 1e-8f) l = 1; return a*(1.0f/l); }

struct Mat4 { float m[16]; };

static Mat4 mIdentity(){ Mat4 r; memset(&r,0,sizeof(r)); r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }

static Mat4 mMul(const Mat4& a, const Mat4& b){
    Mat4 r;
    for (int c = 0; c < 4; c++)
        for (int rw = 0; rw < 4; rw++){
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k*4+rw] * b.m[c*4+k];
            r.m[c*4+rw] = s;
        }
    return r;
}
static Mat4 mTranslate(float x, float y, float z){
    Mat4 r = mIdentity(); r.m[12]=x; r.m[13]=y; r.m[14]=z; return r;
}
static Mat4 mRotX(float a){
    float c = cosf(a), s = sinf(a); Mat4 r = mIdentity();
    r.m[5]=c; r.m[6]=s; r.m[9]=-s; r.m[10]=c; return r;
}
static Mat4 mPerspective(float fovDeg, float aspect, float zn, float zf){
    float f = 1.0f / tanf(fovDeg * 3.14159265f / 360.0f);
    Mat4 r; memset(&r,0,sizeof(r));
    r.m[0]=f/aspect; r.m[5]=f; r.m[10]=(zf+zn)/(zn-zf); r.m[11]=-1; r.m[14]=2*zf*zn/(zn-zf);
    return r;
}
static Mat4 mOrtho(float l, float r_, float b, float t){
    Mat4 r; memset(&r,0,sizeof(r));
    r.m[0]=2/(r_-l); r.m[5]=2/(t-b); r.m[10]=-1;
    r.m[12]=-(r_+l)/(r_-l); r.m[13]=-(t+b)/(t-b); r.m[15]=1;
    return r;
}
static Mat4 mLookAt(Vec3 e, Vec3 c, Vec3 up){
    Vec3 f = norm3(c - e), s = norm3(cross(f, up)), u = cross(s, f);
    Mat4 r = mIdentity();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8]=s.z;
    r.m[1]=u.x; r.m[5]=u.y; r.m[9]=u.z;
    r.m[2]=-f.x; r.m[6]=-f.y; r.m[10]=-f.z;
    r.m[12]=-dot(s,e); r.m[13]=-dot(u,e); r.m[14]=dot(f,e);
    return r;
}

/* ============================================================================
   2. SHADERS
   ========================================================================== */
static const char* VS3 =
"#version 410 core\n"
"layout(location=0) in vec3 aPos;\n"
"layout(location=1) in vec3 aNormal;\n"
"layout(location=2) in vec3 aColor;\n"
"uniform mat4 uProj, uView, uModel;\n"
"out vec3 vN; out vec3 vC; out vec3 vW;\n"
"void main(){\n"
"  vec4 w = uModel * vec4(aPos,1.0);\n"
"  vW = w.xyz; vN = mat3(uModel)*aNormal; vC = aColor;\n"
"  gl_Position = uProj * uView * w;\n"
"}\n";

static const char* FS3 =
"#version 410 core\n"
"in vec3 vN; in vec3 vC; in vec3 vW;\n"
"uniform vec3 uCam; uniform vec3 uLight; uniform vec3 uFog;\n"
"out vec4 frag;\n"
"void main(){\n"
"  vec3 n = normalize(vN);\n"
"  float dif = max(dot(n, uLight), 0.0);\n"
"  vec3 col = vC * (0.38 + 0.72*dif);\n"
"  vec3 V = normalize(uCam - vW);\n"
"  vec3 H = normalize(uLight + V);\n"
"  col += vec3(1.0) * pow(max(dot(n,H),0.0), 40.0) * 0.18;\n"
"  float d = length(uCam - vW);\n"
"  float fog = clamp((d-70.0)/230.0, 0.0, 1.0);\n"
"  frag = vec4(mix(col, uFog, fog), 1.0);\n"
"}\n";

static const char* VS2 =
"#version 410 core\n"
"layout(location=0) in vec2 aPos;\n"
"layout(location=1) in vec2 aUV;\n"
"layout(location=2) in vec4 aColor;\n"
"uniform mat4 uOrtho;\n"
"out vec2 vUV; out vec4 vColor;\n"
"void main(){ vUV=aUV; vColor=aColor; gl_Position = uOrtho * vec4(aPos,0.0,1.0); }\n";

static const char* FS2 =
"#version 410 core\n"
"in vec2 vUV; in vec4 vColor;\n"
"uniform sampler2D uTex;\n"
"out vec4 frag;\n"
"void main(){ frag = vec4(vColor.rgb, vColor.a * texture(uTex, vUV).r); }\n";

static GLuint compileShader(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok){
        char log[1024]; glGetShaderInfoLog(s, 1024, NULL, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
    }
    return s;
}
static GLuint makeProgram(const char* vs, const char* fs){
    GLuint p = glCreateProgram();
    GLuint a = compileShader(GL_VERTEX_SHADER, vs);
    GLuint b = compileShader(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, a); glAttachShader(p, b); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok){
        char log[1024]; glGetProgramInfoLog(p, 1024, NULL, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
    }
    glDeleteShader(a); glDeleteShader(b);
    return p;
}

/* ============================================================================
   3. EMBEDDED 5x7 BITMAP FONT (atlas generated at startup)
   ========================================================================== */
struct Glyph { char ch; unsigned char d[7]; };
static const Glyph GLYPHS[] = {
 {' ', {0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
 {'!', {0x04,0x04,0x04,0x04,0x04,0x00,0x04}},
 {'(', {0x02,0x04,0x08,0x08,0x08,0x04,0x02}},
 {')', {0x08,0x04,0x02,0x02,0x02,0x04,0x08}},
 {'-', {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}},
 {'.', {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}},
 {'/', {0x01,0x01,0x02,0x04,0x08,0x10,0x10}},
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
 {':', {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}},
 {'?', {0x0E,0x11,0x01,0x06,0x04,0x00,0x04}},
 {'A', {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}},
 {'B', {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
 {'C', {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}},
 {'D', {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
 {'E', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}},
 {'F', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
 {'G', {0x0E,0x11,0x10,0x17,0x11,0x11,0x0E}},
 {'H', {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
 {'I', {0x1F,0x04,0x04,0x04,0x04,0x04,0x1F}},
 {'J', {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
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
 {'X', {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
 {'Y', {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}},
 {'Z', {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
};

static GLuint  gAtlasTex = 0;
static int     gCharCell[128];
static const float WHITE_U = (15*8+4)/128.0f;   // solid-white cell (index 127)
static const float WHITE_V = ( 7*8+4)/128.0f;

static void initFont(){
    for (int i = 0; i < 128; i++) gCharCell[i] = -1;
    static unsigned char pix[128*128];
    memset(pix, 0, sizeof(pix));
    int n = (int)(sizeof(GLYPHS)/sizeof(GLYPHS[0]));
    for (int i = 0; i < n; i++){
        gCharCell[(unsigned char)GLYPHS[i].ch] = i;
        int cx = (i % 16) * 8, cy = (i / 16) * 8;
        for (int r = 0; r < 7; r++)
            for (int j = 0; j < 5; j++)
                if (GLYPHS[i].d[r] & (1 << (4-j)))
                    pix[(cy + 6 - r) * 128 + cx + 1 + j] = 255;
    }
    // cell 127 = solid white (used for flat rectangles)
    for (int y = 56; y < 64; y++)
        for (int x = 120; x < 128; x++) pix[y*128+x] = 255;

    glGenTextures(1, &gAtlasTex);
    glBindTexture(GL_TEXTURE_2D, gAtlasTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 128, 128, 0, GL_RED, GL_UNSIGNED_BYTE, pix);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

/* ============================================================================
   4. MESH HELPERS
   ========================================================================== */
struct Vert { float p[3]; float n[3]; float c[3]; };
struct Mesh { GLuint vao; int n; };

static void tri(std::vector<Vert>& v, Vec3 a, Vec3 b, Vec3 c, Vec3 n, Vec3 col){
    const Vec3* pts[3] = {&a, &b, &c};
    for (int i = 0; i < 3; i++){
        Vert t;
        t.p[0]=pts[i]->x; t.p[1]=pts[i]->y; t.p[2]=pts[i]->z;
        t.n[0]=n.x; t.n[1]=n.y; t.n[2]=n.z;
        t.c[0]=col.x; t.c[1]=col.y; t.c[2]=col.z;
        v.push_back(t);
    }
}
static void quad(std::vector<Vert>& v, Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 n, Vec3 col){
    tri(v, a, b, c, n, col);
    tri(v, a, c, d, n, col);
}
static void flatQuad(std::vector<Vert>& v, float x0, float z0, float x1, float z1, float y, Vec3 col){
    quad(v, v3(x0,y,z0), v3(x0,y,z1), v3(x1,y,z1), v3(x1,y,z0), v3(0,1,0), col);
}
static void addBox(std::vector<Vert>& v, Vec3 p, Vec3 s, Vec3 col){
    float x0=p.x-s.x/2, x1=p.x+s.x/2, y0=p.y-s.y/2, y1=p.y+s.y/2, z0=p.z-s.z/2, z1=p.z+s.z/2;
    quad(v, v3(x0,y0,z1), v3(x1,y0,z1), v3(x1,y1,z1), v3(x0,y1,z1), v3( 0,0, 1), col);
    quad(v, v3(x1,y0,z0), v3(x0,y0,z0), v3(x0,y1,z0), v3(x1,y1,z0), v3( 0,0,-1), col);
    quad(v, v3(x1,y0,z1), v3(x1,y0,z0), v3(x1,y1,z0), v3(x1,y1,z1), v3( 1,0, 0), col);
    quad(v, v3(x0,y0,z0), v3(x0,y0,z1), v3(x0,y1,z1), v3(x0,y1,z0), v3(-1,0, 0), col);
    quad(v, v3(x0,y1,z0), v3(x0,y1,z1), v3(x1,y1,z1), v3(x1,y1,z0), v3( 0, 1,0), col);
    quad(v, v3(x0,y0,z1), v3(x0,y0,z0), v3(x1,y0,z0), v3(x1,y0,z1), v3( 0,-1,0), col);
}
static void addCylinderX(std::vector<Vert>& v, Vec3 c, float r, float hl, int seg, Vec3 col){
    for (int i = 0; i < seg; i++){
        float t0 = i*6.2831853f/seg, t1 = (i+1)*6.2831853f/seg, tm = (t0+t1)*0.5f;
        float c0=cosf(t0), s0=sinf(t0), c1=cosf(t1), s1=sinf(t1);
        Vec3 n = v3(0, cosf(tm), sinf(tm));
        quad(v, v3(c.x-hl, c.y+c0*r, c.z+s0*r), v3(c.x-hl, c.y+c1*r, c.z+s1*r),
                v3(c.x+hl, c.y+c1*r, c.z+s1*r), v3(c.x+hl, c.y+c0*r, c.z+s0*r), n, col);
        tri(v, v3(c.x+hl,c.y,c.z), v3(c.x+hl, c.y+c0*r, c.z+s0*r), v3(c.x+hl, c.y+c1*r, c.z+s1*r), v3( 1,0,0), col);
        tri(v, v3(c.x-hl,c.y,c.z), v3(c.x-hl, c.y+c1*r, c.z+s1*r), v3(c.x-hl, c.y+c0*r, c.z+s0*r), v3(-1,0,0), col);
    }
}
static void addConeY(std::vector<Vert>& v, Vec3 b, float r, float h, int seg, Vec3 col){
    Vec3 apex = v3(b.x, b.y+h, b.z);
    float len = sqrtf(h*h + r*r);
    for (int i = 0; i < seg; i++){
        float t0 = i*6.2831853f/seg, t1 = (i+1)*6.2831853f/seg, tm = (t0+t1)*0.5f;
        Vec3 p0 = v3(b.x+cosf(t0)*r, b.y, b.z+sinf(t0)*r);
        Vec3 p1 = v3(b.x+cosf(t1)*r, b.y, b.z+sinf(t1)*r);
        Vec3 n = v3(cosf(tm)*h/len, r/len, sinf(tm)*h/len);
        tri(v, p0, apex, p1, n, col);
        tri(v, b, p0, p1, v3(0,-1,0), col);
    }
}
static Mesh uploadMesh(const std::vector<Vert>& v){
    Mesh m;
    glGenVertexArrays(1, &m.vao);
    glBindVertexArray(m.vao);
    GLuint vbo; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size()*sizeof(Vert)), v.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 36, (void*)12);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 36, (void*)24);
    glBindVertexArray(0);
    m.n = (int)v.size();
    return m;
}
static void drawMesh(const Mesh& m){
    glBindVertexArray(m.vao);
    glDrawArrays(GL_TRIANGLES, 0, m.n);
}

/* ============================================================================
   5. GAME CONSTANTS + STATE
   ========================================================================== */
static const double TRACK_LEN = 402.336;      // quarter mile in meters
static const double CAR_LEN   = 4.4;
static const double GEAR_TOP[5] = { 20, 36, 54, 72, 96 };
static const double GEAR_ACC[5] = { 10.5, 8.6, 7.0, 5.6, 4.4 };
static const double DRAG_C  = 0.0003;
static const double ROLL_C  = 0.15;
static const double SHIFT_CUT = 0.18;
static const double TREE_A1 = 1.0, TREE_A2 = 1.5, TREE_A3 = 2.0, TREE_GREEN = 2.5;

enum State { ST_MENU, ST_COUNT, ST_RACE, ST_END };

struct Car {
    double z, speed;
    int    gear;
    double shiftTimer, rpm;
    bool   finished, reacted;
    double et, reaction, trap, topSpeed;
    float  laneX, wheelAngle;
    double aiReact, aiShift;
};

static Car    cars[2];
static State  state = ST_MENU;
static int    mode  = 0;          // 1 = vs CPU, 2 = two players
static double cdStart = 0, greenTime = 0;
static int    foul = -1;          // -1 none, 0/1 car index, 2 = both
static Vec3   camPos = v3(0, 5, -16);
static float  camFov = 60;

static bool keyCur[512], keyPrev[512];
static bool pressed(int k){ return k >= 0 && k < 512 && keyCur[k] && !keyPrev[k]; }

static float rnd01(){ return (float)rand() / (float)RAND_MAX; }

static void startRace(int m){
    mode = m;
    for (int i = 0; i < 2; i++){
        cars[i].z = -2.2; cars[i].speed = 0; cars[i].gear = 1;
        cars[i].shiftTimer = 0; cars[i].rpm = 1000;
        cars[i].finished = false; cars[i].reacted = false;
        cars[i].et = cars[i].reaction = cars[i].trap = cars[i].topSpeed = 0;
        cars[i].laneX = (i == 0) ? -2.25f : 2.25f;
        cars[i].wheelAngle = 0;
        cars[i].aiReact = 0.20 + 0.25 * rnd01();
        cars[i].aiShift = 0.94 + 0.04 * rnd01();
    }
    foul = -1;
    cdStart = glfwGetTime();
    state = ST_COUNT;
}

/* ============================================================================
   6. PHYSICS + AI
   ========================================================================== */
static void updateCar(Car& c, bool th, bool br, bool sUp, bool sDn,
                      double dt, double now, bool racing){
    if (c.finished){ th = false; sUp = sDn = false; }

    if (c.shiftTimer > 0){
        c.shiftTimer -= dt;
    } else if (sUp && c.gear < 5){
        c.gear++; c.shiftTimer = SHIFT_CUT;
    } else if (sDn && c.gear > 1){
        c.gear--; c.shiftTimer = SHIFT_CUT;
    }

    double ratio = c.speed / GEAR_TOP[c.gear - 1];
    c.rpm = 1000.0 + 6500.0 * ratio;

    double a = 0;
    if (th && c.shiftTimer <= 0 && !c.finished){
        double band = 0.5 + 0.5 * ratio;          // power band: stronger near redline
        if (ratio >= 1.0) band = 0.15;            // rev limiter
        double traction = (c.speed < 6.0) ? (0.70 + c.speed * 0.05) : 1.0;
        a += GEAR_ACC[c.gear - 1] * band * traction;
    }
    a -= DRAG_C * c.speed * c.speed + ROLL_C;
    if (br && !c.finished) a -= 11.0;
    if (c.finished) a -= 2.5;                     // coast down after the line
    if (a < 0 && c.speed == 0) a = 0;

    c.speed += a * dt; if (c.speed < 0) c.speed = 0;
    c.z += c.speed * dt;
    c.wheelAngle += (float)(c.speed * dt / 0.34);

    if (racing && !c.finished){
        if (!c.reacted && th && now >= greenTime){
            c.reacted = true; c.reaction = now - greenTime;
        }
        if (c.speed > c.topSpeed) c.topSpeed = c.speed;
        if (c.z + CAR_LEN * 0.5 >= TRACK_LEN){
            c.finished = true;
            c.et = now - greenTime;
            c.trap = c.speed;
        }
    }
}

static void aiControl(Car& c, double tGreen, bool& th, bool& sUp){
    th  = tGreen >= c.aiReact;
    sUp = false;
    if (th && c.gear < 5 && c.shiftTimer <= 0){
        double r = c.speed / GEAR_TOP[c.gear - 1];
        if (r >= c.aiShift){
            sUp = true;
            c.aiShift = 0.93 + 0.05 * rnd01();
        }
    }
}

static int judge(){
    if (foul >= 0) return (foul == 2) ? -1 : 1 - foul;
    if (cars[0].finished && cars[1].finished){
        if (fabs(cars[0].et - cars[1].et) < 1e-4) return -2;   // dead heat
        return cars[0].et < cars[1].et ? 0 : 1;
    }
    if (cars[0].finished) return 0;
    if (cars[1].finished) return 1;
    return -1;
}
static const char* carName(int i){
    return i == 0 ? "PLAYER 1" : (mode == 1 ? "CPU" : "PLAYER 2");
}

static void update(double dt, double now){
    // global keys
    if (pressed(GLFW_KEY_1)) startRace(1);
    if (pressed(GLFW_KEY_2)) startRace(2);
    if (pressed(GLFW_KEY_ESCAPE)) state = ST_MENU;
    if (mode != 0 && state != ST_MENU && pressed(GLFW_KEY_R)) startRace(mode);

    if (state == ST_MENU) return;

    if (state == ST_COUNT){
        bool f1 = keyCur[GLFW_KEY_W];
        bool f2 = (mode == 2) && keyCur[GLFW_KEY_UP];
        if (f1 || f2){
            foul = (f1 && f2) ? 2 : (f1 ? 0 : 1);
            state = ST_END;
        } else if (now - cdStart >= TREE_GREEN){
            greenTime = cdStart + TREE_GREEN;
            state = ST_RACE;
        }
        return;
    }

    // inputs
    bool th[2] = {false, false}, br[2] = {false, false};
    bool sUp[2] = {false, false}, sDn[2] = {false, false};

    th[0]  = keyCur[GLFW_KEY_W];
    br[0]  = keyCur[GLFW_KEY_S];
    sUp[0] = pressed(GLFW_KEY_E) || pressed(GLFW_KEY_LEFT_SHIFT);
    sDn[0] = pressed(GLFW_KEY_Q);

    if (mode == 2){
        th[1]  = keyCur[GLFW_KEY_UP];
        br[1]  = keyCur[GLFW_KEY_DOWN];
        sUp[1] = pressed(GLFW_KEY_PERIOD) || pressed(GLFW_KEY_RIGHT_SHIFT);
        sDn[1] = pressed(GLFW_KEY_COMMA);
    } else if (state == ST_RACE){
        aiControl(cars[1], now - greenTime, th[1], sUp[1]);
    }

    bool racing = (state == ST_RACE);
    updateCar(cars[0], th[0], br[0], sUp[0], sDn[0], dt, now, racing);
    updateCar(cars[1], th[1], br[1], sUp[1], sDn[1], dt, now, racing);

    if (state == ST_RACE){
        bool f0 = cars[0].finished, f1 = cars[1].finished;
        if (f0 && f1) state = ST_END;
        else if (f0 && now >= greenTime + cars[0].et + 3.0) state = ST_END;
        else if (f1 && now >= greenTime + cars[1].et + 3.0) state = ST_END;
    }
}

/* ============================================================================
   7. SCENE GEOMETRY
   ========================================================================== */
static Mesh sceneMesh, carMesh[2], wheelMesh;
static GLuint dynVAO, dynVBO;
static std::vector<Vert> dynVerts;

static float hash11(float n){ return fmodf(fabsf(sinf(n) * 43758.5453f), 1.0f); }

static void buildScene(){
    std::vector<Vert> v; v.reserve(200000);
    Vec3 asphalt1 = v3(0.30f,0.30f,0.32f), asphalt2 = v3(0.27f,0.27f,0.30f);
    Vec3 grass1 = v3(0.16f,0.31f,0.15f), grass2 = v3(0.15f,0.28f,0.14f);
    Vec3 white = v3(0.85f,0.85f,0.85f);

    // grass strips
    for (float z = -40; z < (float)TRACK_LEN + 160; z += 12)
        flatQuad(v, -220, z, 220, z + 12, 0.0f, ((int)(z/12) & 1) ? grass1 : grass2);
    // road
    for (float z = -25; z < (float)TRACK_LEN + 100; z += 10)
        flatQuad(v, -4.6f, z, 4.6f, z + 10, 0.02f, ((int)(z/10) & 1) ? asphalt1 : asphalt2);
    // edge lines
    for (float z = -5; z < (float)TRACK_LEN + 30; z += 10){
        flatQuad(v,  4.30f, z,  4.42f, z + 10, 0.03f, white);
        flatQuad(v, -4.42f, z, -4.30f, z + 10, 0.03f, white);
    }
    // center dashes
    for (float z = -20; z < (float)TRACK_LEN + 40; z += 7)
        flatQuad(v, -0.08f, z, 0.08f, z + 3, 0.03f, white);
    // start line
    flatQuad(v, -4.6f, 0, 4.6f, 0.35f, 0.035f, white);
    // finish line (checkers)
    for (int i = 0; i < 26; i++)
        for (int j = 0; j < 2; j++){
            float x0 = -4.55f + i * 0.35f, z0 = (float)TRACK_LEN + j * 0.35f;
            flatQuad(v, x0, z0, x0 + 0.35f, z0 + 0.35f, 0.035f,
                     ((i + j) & 1) ? v3(0.92f,0.92f,0.92f) : v3(0.12f,0.12f,0.12f));
        }
    // red/white barriers
    for (float z = -20; z < (float)TRACK_LEN + 60; z += 8){
        Vec3 wc = ((int)(z/8) & 1) ? v3(0.85f,0.22f,0.18f) : v3(0.9f,0.9f,0.9f);
        addBox(v, v3(-6.2f, 0.425f, z+4), v3(0.35f,0.85f,8), wc);
        addBox(v, v3( 6.2f, 0.425f, z+4), v3(0.35f,0.85f,8), wc);
    }
    // scenery trees
    for (float z = 6; z < (float)TRACK_LEN + 80; z += 16)
        for (int s = -1; s <= 1; s += 2){
            float r = hash11(z * 3.7f + (float)s * 17.0f);
            float x = s * (10.0f + r * 30.0f);
            addBox(v, v3(x, 0.55f + r*0.5f, z + r*6), v3(0.3f, 1.1f + r, 0.3f), v3(0.35f,0.24f,0.15f));
            addConeY(v, v3(x, 1.0f + r, z + r*6), 1.0f + r, 2.2f + 1.6f*r, 10,
                     v3(0.13f + 0.10f*r, 0.32f + 0.16f*r, 0.12f));
        }
    // grandstands near finish
    for (int s = -1; s <= 1; s += 2){
        float gx = s * 13.5f;
        addBox(v, v3(gx, 3.2f, (float)TRACK_LEN-20), v3(5, 6.4f, 46), v3(0.42f,0.45f,0.55f));
        addBox(v, v3(gx, 6.6f, (float)TRACK_LEN-20), v3(6, 0.4f, 48), v3(0.22f,0.23f,0.28f));
        float ix = s * (13.5f - 2.6f);
        addBox(v, v3(ix, 1.6f, (float)TRACK_LEN-20), v3(0.3f,1.0f,44), v3(0.7f,0.3f,0.3f));
        addBox(v, v3(ix, 3.4f, (float)TRACK_LEN-20), v3(0.3f,1.0f,44), v3(0.3f,0.4f,0.75f));
        addBox(v, v3(ix, 5.2f, (float)TRACK_LEN-20), v3(0.3f,1.0f,44), v3(0.8f,0.75f,0.3f));
    }
    // christmas-tree poles + backboards
    for (int s = -1; s <= 1; s += 2){
        float px = s * 7.2f;
        addBox(v, v3(px, 2.25f, -0.5f), v3(0.18f, 4.5f, 0.18f), v3(0.25f,0.25f,0.27f));
        addBox(v, v3(px, 2.7f, -0.64f), v3(0.55f, 3.2f, 0.1f), v3(0.1f,0.1f,0.12f));
    }
    sceneMesh = uploadMesh(v);
}

static void buildCars(){
    for (int i = 0; i < 2; i++){
        Vec3 col = (i == 0) ? v3(0.86f,0.12f,0.10f) : v3(0.12f,0.35f,0.90f);
        Vec3 dark = col * 0.55f;
        std::vector<Vert> v;
        addBox(v, v3(0,0.62f,0),        v3(1.8f,0.55f,4.4f),  col);
        addBox(v, v3(0,0.5f,2.05f),     v3(1.7f,0.3f,0.5f),   col);
        addBox(v, v3(0,1.12f,-0.2f),    v3(1.55f,0.5f,2.1f),  v3(0.07f,0.09f,0.12f));
        addBox(v, v3(0,0.91f,0.55f),    v3(0.5f,0.03f,3.1f),  dark);
        addBox(v, v3(0,1.08f,-2.05f),   v3(1.7f,0.07f,0.35f), dark);
        addBox(v, v3(-0.6f,0.95f,-2.05f), v3(0.09f,0.22f,0.09f), dark);
        addBox(v, v3( 0.6f,0.95f,-2.05f), v3(0.09f,0.22f,0.09f), dark);
        addBox(v, v3(0,0.42f, 2.22f),   v3(1.75f,0.22f,0.12f), v3(0.08f,0.08f,0.09f));
        addBox(v, v3(0,0.42f,-2.22f),   v3(1.75f,0.22f,0.12f), v3(0.08f,0.08f,0.09f));
        addBox(v, v3(-0.55f,0.68f, 2.26f), v3(0.35f,0.12f,0.06f), v3(0.95f,0.95f,0.8f));
        addBox(v, v3( 0.55f,0.68f, 2.26f), v3(0.35f,0.12f,0.06f), v3(0.95f,0.95f,0.8f));
        addBox(v, v3(-0.55f,0.70f,-2.26f), v3(0.4f,0.12f,0.06f),  v3(0.9f,0.1f,0.08f));
        addBox(v, v3( 0.55f,0.70f,-2.26f), v3(0.4f,0.12f,0.06f),  v3(0.9f,0.1f,0.08f));
        addBox(v, v3(-0.9f,0.4f,0),     v3(0.08f,0.16f,3.4f), dark);
        addBox(v, v3( 0.9f,0.4f,0),     v3(0.08f,0.16f,3.4f), dark);
        carMesh[i] = uploadMesh(v);
    }
    std::vector<Vert> wv;
    addCylinderX(wv, v3(0,0,0), 0.34f, 0.13f,  18, v3(0.09f,0.09f,0.10f));
    addCylinderX(wv, v3(0,0,0), 0.15f, 0.145f, 12, v3(0.55f,0.55f,0.60f));
    wheelMesh = uploadMesh(wv);

    glGenVertexArrays(1, &dynVAO);
    glBindVertexArray(dynVAO);
    glGenBuffers(1, &dynVBO);
    glBindBuffer(GL_ARRAY_BUFFER, dynVBO);
    glBufferData(GL_ARRAY_BUFFER, 256 * 1024, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 36, (void*)12);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 36, (void*)24);
    glBindVertexArray(0);
}

/* ============================================================================
   8. HUD (dynamic 2D batch: text + rects)
   ========================================================================== */
struct HVert { float x, y, u, v, r, g, b, a; };
static GLuint hudVAO, hudVBO;
static std::vector<HVert> hudVerts;

static void hudQuad(float x0, float y0, float x1, float y1,
                    float u0, float v0, float u1, float v1, const float* c){
    HVert a = {x0,y0,u0,v0,c[0],c[1],c[2],c[3]};
    HVert b = {x1,y0,u1,v0,c[0],c[1],c[2],c[3]};
    HVert c2= {x1,y1,u1,v1,c[0],c[1],c[2],c[3]};
    HVert d = {x0,y1,u0,v1,c[0],c[1],c[2],c[3]};
    hudVerts.push_back(a); hudVerts.push_back(b); hudVerts.push_back(c2);
    hudVerts.push_back(a); hudVerts.push_back(c2); hudVerts.push_back(d);
}
static void drawRect(float x, float y, float w, float h, const float* c){
    hudQuad(x, y, x + w, y + h, WHITE_U, WHITE_V, WHITE_U, WHITE_V, c);
}
static float textWidth(const char* s, float sz){ return (float)strlen(s) * sz * 0.75f; }
static float drawText(float x, float y, float sz, const float* c, const char* s){
    float pen = x;
    for (const char* p = s; *p; p++){
        char ch = *p;
        if (ch >= 'a' && ch <= 'z') ch -= 32;
        int cell = (ch > 0 && ch < 128) ? gCharCell[(int)ch] : -1;
        if (cell >= 0){
            float cx = (float)((cell % 16) * 8), cy = (float)((cell / 16) * 8);
            hudQuad(pen, y, pen + sz, y + sz,
                    cx/128.0f, cy/128.0f, (cx+8)/128.0f, (cy+8)/128.0f, c);
        }
        pen += sz * 0.75f;
    }
    return pen - x;
}
static void drawTextC(float cx, float y, float sz, const float* c, const char* s){
    drawText(cx - textWidth(s, sz) * 0.5f, y, sz, c, s);
}
static void flushHUD(){
    if (hudVerts.empty()) return;
    glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(hudVerts.size()*sizeof(HVert)),
                 hudVerts.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(hudVAO);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)hudVerts.size());
}

static const float C_WHITE[4] = {1,1,1,1}, C_GRAY[4] = {0.6f,0.6f,0.65f,1};
static const float CAR_COL[2][4] = { {0.95f,0.25f,0.2f,1}, {0.25f,0.5f,1,1} };

static void rpmBar(float x, float y, float w, float h, float frac, double now){
    drawRect(x - 2, y - 2, w + 4, h + 4, (const float[]){0,0,0,0.6f});
    drawRect(x, y, w, h, (const float[]){0.10f,0.10f,0.13f,0.95f});
    drawRect(x + w * 0.88f, y, w * 0.12f, h, (const float[]){0.45f,0.08f,0.08f,0.95f});
    if (frac > 1) frac = 1;
    const float* fc;
    if (frac < 0.7f)       fc = (const float[]){0.15f,0.85f,0.25f,1};
    else if (frac < 0.9f)  fc = (const float[]){0.95f,0.8f,0.1f,1};
    else                   fc = (const float[]){0.95f,0.2f,0.1f,1};
    if (frac > 0.985f && fmod(now * 6.0, 1.0) < 0.5f)
        fc = (const float[]){1,1,1,1};
    drawRect(x, y, w * frac, h, fc);
}

static void drawCarPanel(int i, float x, float y, float k, double now){
    Car& c = cars[i];
    float w = 320*k, h = 150*k;
    drawRect(x - 4*k, y - 4*k, w + 8*k, h + 8*k, (const float[]){0,0,0,0.35f});
    drawRect(x, y, w, h, (const float[]){0.05f,0.06f,0.09f,0.72f});
    float hdr[4] = {CAR_COL[i][0], CAR_COL[i][1], CAR_COL[i][2], 0.85f};
    drawRect(x, y + h - 30*k, w, 30*k, hdr);
    drawText(x + 10*k, y + h - 24*k, 16*k, C_WHITE, carName(i));

    char b[64];
    snprintf(b, sizeof(b), "%d MPH", (int)(c.speed * 2.23694));
    drawText(x + 12*k, y + 64*k, 30*k, C_WHITE, b);
    snprintf(b, sizeof(b), "GEAR %d", c.gear);
    drawText(x + 205*k, y + 70*k, 18*k, C_WHITE, b);

    float frac = (float)((c.rpm - 1000.0) / 6500.0);
    rpmBar(x + 12*k, y + 26*k, w - 24*k, 18*k, frac, now);
    snprintf(b, sizeof(b), "%d RPM", (int)c.rpm);
    drawText(x + 12*k, y + 5*k, 12*k, C_GRAY, b);

    if (!c.finished && frac > 0.97f && fmod(now * 4.0, 1.0) < 0.7f)
        drawTextC(x + w * 0.5f, y + h + 8*k, 20*k, (const float[]){1,0.3f,0.2f,1}, "SHIFT!");
}

static void drawHUD(double now, int fbW, int fbH){
    hudVerts.clear();
    float k = fbH / 720.0f;
    char b[96];

    if (state == ST_MENU){
        drawRect(0, 0, (float)fbW, (float)fbH, (const float[]){0,0,0,0.45f});
        drawTextC(fbW*0.5f, fbH*0.70f, 64*k, (const float[]){1,0.85f,0.15f,1}, "DRAG RACER");
        drawTextC(fbW*0.5f, fbH*0.64f, 18*k, C_GRAY, "QUARTER MILE SHOWDOWN");
        drawTextC(fbW*0.5f, fbH*0.52f, 24*k, C_WHITE, "PRESS 1 - RACE THE COMPUTER");
        drawTextC(fbW*0.5f, fbH*0.46f, 24*k, C_WHITE, "PRESS 2 - TWO PLAYERS");
        drawTextC(fbW*0.5f, fbH*0.34f, 14*k, C_GRAY, "P1: W GAS / S BRAKE / E OR L-SHIFT SHIFT UP / Q SHIFT DOWN");
        drawTextC(fbW*0.5f, fbH*0.30f, 14*k, C_GRAY, "P2: UP GAS / DOWN BRAKE / . OR R-SHIFT SHIFT UP / , SHIFT DOWN");
        drawTextC(fbW*0.5f, fbH*0.23f, 15*k, (const float[]){1,0.85f,0.15f,1},
                  "LAUNCH ON GREEN - THROTTLE BEFORE GREEN = FOUL!");
        drawTextC(fbW*0.5f, fbH*0.06f, 13*k, C_GRAY, "ESC QUITS FROM MENU");
        flushHUD();
        return;
    }

    // top-left mode label, top-right help
    drawText(14*k, fbH - 30*k, 15*k, C_GRAY, mode == 1 ? "VS COMPUTER" : "2 PLAYER");
    drawText(fbW - textWidth("R RESTART ESC MENU", 13*k) - 14*k, fbH - 28*k, 13*k, C_GRAY, "R RESTART ESC MENU");

    // ET clock + progress bar
    if (state == ST_RACE || state == ST_END){
        double t = 0;
        if (cars[0].finished) t = cars[0].et;
        else if (now > greenTime) t = now - greenTime;
        snprintf(b, sizeof(b), "ET %06.2f", t);
        drawTextC(fbW*0.5f, fbH - 32*k, 24*k, C_WHITE, b);

        float bw = 380*k, bx = fbW*0.5f - bw*0.5f, by = fbH - 52*k;
        drawRect(bx, by, bw, 6*k, (const float[]){0,0,0,0.5f});
        for (int i = 0; i < 2; i++){
            double fr = (cars[i].z + CAR_LEN*0.5) / TRACK_LEN;
            if (fr < 0) fr = 0; if (fr > 1) fr = 1;
            drawRect(bx + (float)fr * bw - 3*k, by - 3*k, 6*k, 12*k, CAR_COL[i]);
        }
    }

    drawCarPanel(0, 20*k, 20*k, k, now);
    drawCarPanel(1, fbW - 340*k, 20*k, k, now);

    if (state == ST_COUNT){
        double t = now - cdStart;
        if (t < TREE_A1)      drawTextC(fbW*0.5f, fbH*0.62f, 40*k, (const float[]){1,0.85f,0.15f,1}, "STAGE");
        drawTextC(fbW*0.5f, fbH*0.55f, 16*k, C_GRAY, "WAIT FOR GREEN - EARLY THROTTLE = RED LIGHT");
    }
    if (state == ST_RACE && now - greenTime < 0.7)
        drawTextC(fbW*0.5f, fbH*0.60f, 64*k, (const float[]){0.2f,1,0.3f,1}, "GO!");

    if (state == ST_END){
        float w = 560*k, h = 260*k, px = fbW*0.5f - w*0.5f, py = fbH*0.5f - h*0.5f;
        drawRect(px - 5*k, py - 5*k, w + 10*k, h + 10*k, (const float[]){0,0,0,0.5f});
        drawRect(px, py, w, h, (const float[]){0.04f,0.05f,0.09f,0.9f});
        int win = judge();
        if (win >= 0){
            snprintf(b, sizeof(b), "%s WINS!", carName(win));
            drawTextC(fbW*0.5f, py + h - 48*k, 30*k, CAR_COL[win], b);
        } else if (win == -1){
            drawTextC(fbW*0.5f, py + h - 48*k, 30*k, (const float[]){1,0.3f,0.2f,1}, "DOUBLE FOUL - NO WINNER");
        } else if (win == -2){
            drawTextC(fbW*0.5f, py + h - 48*k, 30*k, C_WHITE, "DEAD HEAT!");
        }
        for (int i = 0; i < 2; i++){
            float y = py + h - 100*k - i * 56*k;
            if (foul == i || foul == 2){
                snprintf(b, sizeof(b), "%s: FALSE START", carName(i));
                drawText(px + 30*k, y, 18*k, (const float[]){1,0.35f,0.3f,1}, b);
            } else if (cars[i].finished){
                snprintf(b, sizeof(b), "ET %.3f  TRAP %.1F MPH  RT %.3f",
                         cars[i].et, cars[i].trap * 2.23694, cars[i].reaction);
                drawText(px + 30*k, y, 18*k, CAR_COL[i], carName(i));
                drawText(px + 150*k, y, 18*k, C_WHITE, b);
            } else {
                snprintf(b, sizeof(b), "%s: RUNNING...", carName(i));
                drawText(px + 30*k, y, 18*k, CAR_COL[i], b);
            }
        }
        if (fmod(now * 2.0, 1.0) < 0.7f)
            drawTextC(fbW*0.5f, py + 16*k, 16*k, C_GRAY, "R RESTART   ESC MENU");
    }
    flushHUD();
}

/* ============================================================================
   9. RENDERING
   ========================================================================== */
static GLuint prog3, prog2;
static GLint u3_proj, u3_view, u3_model, u3_cam, u3_light, u3_fog;
static GLint u2_ortho;

static void buildDynLights(double now){
    dynVerts.clear();
    bool lit[5] = {false,false,false,false,false};
    if (state == ST_COUNT){
        double t = now - cdStart;
        lit[0] = t >= TREE_A1; lit[1] = t >= TREE_A2; lit[2] = t >= TREE_A3;
    } else if (state == ST_RACE){
        lit[3] = (now - greenTime) < 0.8;
    } else if (state == ST_END && foul >= 0){
        lit[4] = true;
    }
    Vec3 dimC[5] = { v3(0.30f,0.22f,0.08f), v3(0.30f,0.22f,0.08f), v3(0.30f,0.22f,0.08f),
                     v3(0.06f,0.22f,0.10f), v3(0.28f,0.07f,0.07f) };
    Vec3 litC[5] = { v3(1,0.75f,0.15f), v3(1,0.75f,0.15f), v3(1,0.75f,0.15f),
                     v3(0.25f,1,0.30f), v3(1,0.12f,0.10f) };
    float ys[5] = {3.9f, 3.3f, 2.7f, 2.1f, 1.5f};
    for (int s = -1; s <= 1; s += 2)
        for (int i = 0; i < 5; i++)
            addBox(dynVerts, v3(s*7.2f, ys[i], -0.5f), v3(0.34f,0.34f,0.34f),
                   lit[i] ? litC[i] : dimC[i]);

    glBindBuffer(GL_ARRAY_BUFFER, dynVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(dynVerts.size()*sizeof(Vert)),
                 dynVerts.data(), GL_DYNAMIC_DRAW);
}

static void render3D(double now, int fbW, int fbH){
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glViewport(0, 0, fbW, fbH);
    glClearColor(0.60f, 0.76f, 0.90f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(prog3);
    Mat4 proj = mPerspective(camFov, fbW / (float)fbH, 0.1f, 500.0f);
    Mat4 view = mLookAt(camPos,
                        (state == ST_MENU) ? v3(0,1,2)
                        : v3(0, 1.1f, (float)((cars[0].z + cars[1].z)*0.5 + 14.0)),
                        v3(0,1,0));
    glUniformMatrix4fv(u3_proj, 1, GL_FALSE, proj.m);
    glUniformMatrix4fv(u3_view, 1, GL_FALSE, view.m);
    glUniform3f(u3_cam, camPos.x, camPos.y, camPos.z);
    glUniform3f(u3_light, 0.43f, 0.86f, 0.28f);
    glUniform3f(u3_fog, 0.60f, 0.76f, 0.90f);

    Mat4 ident = mIdentity();
    glUniformMatrix4fv(u3_model, 1, GL_FALSE, ident.m);
    drawMesh(sceneMesh);

    buildDynLights(now);
    glBindVertexArray(dynVAO);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)dynVerts.size());

    for (int i = 0; i < 2; i++){
        Mat4 base = mTranslate(cars[i].laneX, 0, (float)cars[i].z);
        glUniformMatrix4fv(u3_model, 1, GL_FALSE, base.m);
        drawMesh(carMesh[i]);
        static const Vec3 WP[4] = { { 0.83f,0.34f, 1.45f}, {-0.83f,0.34f, 1.45f},
                                    { 0.83f,0.34f,-1.45f}, {-0.83f,0.34f,-1.45f} };
        for (int w = 0; w < 4; w++){
            Mat4 m = mMul(base, mMul(mTranslate(WP[w].x, WP[w].y, WP[w].z),
                                       mRotX(cars[i].wheelAngle)));
            glUniformMatrix4fv(u3_model, 1, GL_FALSE, m.m);
            drawMesh(wheelMesh);
        }
    }
}

static void updateCamera(double dt, double now){
    Vec3 want; float fovWant;
    if (state == ST_MENU){
        float a = (float)now * 0.3f;
        want = v3(sinf(a) * 15.0f, 5.5f, 2.0f + cosf(a) * 15.0f);
        fovWant = 55;
    } else {
        double midZ = (cars[0].z + cars[1].z) * 0.5;
        double avg  = (cars[0].speed + cars[1].speed) * 0.5;
        want = v3(sinf((float)now * 9.0f) * 0.05f * (float)(avg / 90.0),
                  (float)(3.2 + avg * 0.012),
                  (float)(midZ - (8.5 + avg * 0.045)));
        fovWant = 60.0f + (float)avg * 0.16f;
    }
    float t = 1.0f - expf(-(float)dt * 3.5f);
    camPos = camPos + (want - camPos) * t;
    camFov += (fovWant - camFov) * t;
}

/* ============================================================================
   10. INPUT + MAIN
   ========================================================================== */
static void keyCallback(GLFWwindow*, int key, int, int action, int){
    if (key >= 0 && key < 512) keyCur[key] = (action != GLFW_RELEASE);
}

int main(){
    srand((unsigned)time(NULL));
    if (!glfwInit()){ fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Drag Racer - OpenGL 4.1", NULL, NULL);
    if (!win){ fprintf(stderr, "Window creation failed (need OpenGL 4.1)\n"); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetKeyCallback(win, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK){ fprintf(stderr, "glewInit failed\n"); return 1; }
    glGetError(); // clear any init-time error
    printf("OpenGL: %s\n", (const char*)glGetString(GL_VERSION));

    prog3 = makeProgram(VS3, FS3);
    prog2 = makeProgram(VS2, FS2);
    u3_proj  = glGetUniformLocation(prog3, "uProj");
    u3_view  = glGetUniformLocation(prog3, "uView");
    u3_model = glGetUniformLocation(prog3, "uModel");
    u3_cam   = glGetUniformLocation(prog3, "uCam");
    u3_light = glGetUniformLocation(prog3, "uLight");
    u3_fog   = glGetUniformLocation(prog3, "uFog");
    u2_ortho = glGetUniformLocation(prog2, "uOrtho");

    initFont();
    buildScene();
    buildCars();

    glGenVertexArrays(1, &hudVAO);
    glBindVertexArray(hudVAO);
    glGenBuffers(1, &hudVBO);
    glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
    glBufferData(GL_ARRAY_BUFFER, 1024 * 1024, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, (void*)8);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, (void*)16);
    glBindVertexArray(0);

    glEnable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    double last = glfwGetTime();
    while (!glfwWindowShouldClose(win)){
        double now = glfwGetTime();
        double dt = now - last; last = now;
        if (dt > 0.05) dt = 0.05;

        glfwPollEvents();
        update(dt, now);
        updateCamera(dt, now);

        int fbW, fbH;
        glfwGetFramebufferSize(win, &fbW, &fbH);

        render3D(now, fbW, fbH);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glUseProgram(prog2);
        Mat4 o = mOrtho(0, (float)fbW, 0, (float)fbH);
        glUniformMatrix4fv(u2_ortho, 1, GL_FALSE, o.m);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, gAtlasTex);
        glUniform1i(glGetUniformLocation(prog2, "uTex"), 0);
        drawHUD(now, fbW, fbH);

        glfwSwapBuffers(win);
        memcpy(keyPrev, keyCur, sizeof(keyCur));
    }
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}