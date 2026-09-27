// ocp_octagon_rings.cpp
// OpenGL 4.1 core, GLFW + GLEW, no GLM.
//
// Octagonal OCP-style logo interpretation:
// - 3 concentric octagonal rings
// - inner ring (C): single side-gap open on the right in the logo pose
// - middle + outer rings (backwards-Q): both bands run straight past their
//   bottom-left corner and are cut by two shared vertical lines, which draws
//   the logo's long tapering stem, its square-ended bottom bar, and the
//   straight corridor between them (reads as a backwards Q / stylized P)
// - each ring only has 8 discrete rotations (45 degree steps)
// - doorway opens when all 3 gap orientations align
// - dot/person escapes through the aligned doorway
// - GUI buttons rotate rings
// - random rotation every 10 minutes
// - rings do not rotate while dot is in hallway or in rotating bands
// - dot cannot enter rotating bands unless door is open

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "ocp_logo_geom.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <array>
#include <algorithm>
#include <utility>

using namespace std;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
//
// Octagon constants, band radii and the backwards-Q notch lines all live in
// ocp_logo_geom.h, which the offline reference checker shares.

static const float DOT_RADIUS = 0.13f;
static const float PLAYER_SPEED = 2.2f;

// One 45-degree rotation takes about 0.35 seconds.
static const float ROT_SPEED = STEP / 0.35f;

static const double RANDOM_INTERVAL = 600.0; // 10 minutes

static const float WORLD_HALF_HEIGHT = 3.9f;
static const float WORLD_LIMIT = 6.0f;

// ---------------------------------------------------------------------------
// Tiny 2D math, no GLM
// ---------------------------------------------------------------------------
// Vec2 and its helpers come from ocp_logo_geom.h.

struct Vec4
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct Mat4
{
    // Column-major, compatible with OpenGL.
    float m[16] = {};
};

static Mat4 ortho2D(float left, float right, float bottom, float top, float zNear, float zFar)
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

// ---------------------------------------------------------------------------
// Shader sources
// ---------------------------------------------------------------------------

static const char* vertexSrc = R"glsl(
#version 410 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;

uniform mat4 uProj;

out vec4 vColor;

void main()
{
    vColor = aColor;
    gl_Position = uProj * vec4(aPos, 0.0, 1.0);
}
)glsl";

static const char* fragmentSrc = R"glsl(
#version 410 core

in vec4 vColor;
out vec4 fragColor;

void main()
{
    fragColor = vColor;
}
)glsl";

// ---------------------------------------------------------------------------
// Global shader program
// ---------------------------------------------------------------------------

static GLuint g_program = 0;
static GLint g_uProj = -1;

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);

    if (!ok)
    {
        char info[1024];
        glGetShaderInfoLog(shader, sizeof(info), nullptr, info);
        fprintf(stderr, "Shader compile error:\n%s\n", info);
    }

    return shader;
}

static GLuint linkProgram(GLuint vs, GLuint fs)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);

    if (!ok)
    {
        char info[1024];
        glGetProgramInfoLog(program, sizeof(info), nullptr, info);
        fprintf(stderr, "Program link error:\n%s\n", info);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

// ---------------------------------------------------------------------------
// Dynamic 2D mesh builder
// ---------------------------------------------------------------------------

struct DynamicMesh2D
{
    GLuint vao = 0;
    GLuint vbo = 0;
    vector<float> data; // x, y, r, g, b, a

    void init()
    {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));

        glBindVertexArray(0);

        data.reserve(20000);
    }

    void clear()
    {
        data.clear();
    }

    void addVertex(Vec2 p, Vec4 c)
    {
        data.push_back(p.x);
        data.push_back(p.y);
        data.push_back(c.x);
        data.push_back(c.y);
        data.push_back(c.z);
        data.push_back(c.w);
    }

    void addTriangle(Vec2 a, Vec2 b, Vec2 c, Vec4 color)
    {
        addVertex(a, color);
        addVertex(b, color);
        addVertex(c, color);
    }

    void addQuad(Vec2 a, Vec2 b, Vec2 c, Vec2 d, Vec4 color)
    {
        addTriangle(a, b, c, color);
        addTriangle(a, c, d, color);
    }

    void addCircle(float cx, float cy, float r, int seg, Vec4 color)
    {
        if (seg < 3) seg = 3;

        Vec2 center{cx, cy};

        for (int i = 0; i < seg; ++i)
        {
            float a0 = 2.0f * PI * (float)i / (float)seg;
            float a1 = 2.0f * PI * (float)(i + 1) / (float)seg;

            Vec2 p0{cx + cosf(a0) * r, cy + sinf(a0) * r};
            Vec2 p1{cx + cosf(a1) * r, cy + sinf(a1) * r};

            addTriangle(center, p0, p1, color);
        }
    }

    void addArc(float cx, float cy, float radius, float thickness, float start, float end, Vec4 color)
    {
        if (fabsf(end - start) < 1e-6f)
            return;

        int seg = max(2, (int)(fabsf(end - start) * 64.0f));

        float inner = radius - thickness * 0.5f;
        float outer = radius + thickness * 0.5f;

        if (inner < 0.0f) inner = 0.0f;

        for (int i = 0; i < seg; ++i)
        {
            float a0 = start + (end - start) * (float)i / (float)seg;
            float a1 = start + (end - start) * (float)(i + 1) / (float)seg;

            Vec2 ix0{cx + cosf(a0) * inner, cy + sinf(a0) * inner};
            Vec2 ox0{cx + cosf(a0) * outer, cy + sinf(a0) * outer};
            Vec2 ox1{cx + cosf(a1) * outer, cy + sinf(a1) * outer};
            Vec2 ix1{cx + cosf(a1) * inner, cy + sinf(a1) * inner};

            addQuad(ix0, ox0, ox1, ix1, color);
        }
    }

    void draw(const Mat4& proj)
    {
        if (data.empty())
            return;

        glUseProgram(g_program);
        glUniformMatrix4fv(g_uProj, 1, GL_FALSE, &proj.m[0]);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(data.size() * sizeof(float)), data.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(data.size() / 6));
        glBindVertexArray(0);
    }
};

// ---------------------------------------------------------------------------
// Ring / gameplay types
// ---------------------------------------------------------------------------

struct Ring
{
    float innerR = 0.0f;
    float outerR = 0.0f;

    // Gap orientation in 45-degree steps: 0..7
    // 0 = right, 1 = upper-right, 2 = top, 3 = upper-left,
    // 4 = left, 5 = lower-left, 6 = bottom, 7 = lower-right.
    // For C-style rings this is the missing side. For backwards-Q (pStyle)
    // rings it is the notch opening (diagonal tip + inset bottom);
    // the stem is the CCW-adjacent flat that meets the diagonal.
    int side = 0;

    // false = C opening (single missing side); true = backwards-Q stem/notch.
    bool pStyle = false;

    // Visual rotation angle in radians. Unwrapped for smooth animation.
    float angle = 0.0f;
    float targetAngle = 0.0f;

    bool animating = false;

    // UI color for buttons.
    Vec4 uiColor{};
};

enum Zone
{
    ZONE_INSIDE = 0,
    ZONE_BAND = 1,
    ZONE_HALLWAY = 2
};

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static GLFWwindow* window = nullptr;
static DynamicMesh2D g_mesh;

static array<Ring, 3> rings;

static bool keys[512] = {};

static double mouseXPixels = 0.0;
static double mouseYPixels = 0.0;

static float playerX = 0.0f;
static float playerY = 0.0f;

static bool anyAnimating = false;
static bool doorOpen = false;
static bool doorUsable = false;

static Zone playerZone = ZONE_INSIDE;

static double currentTime = 0.0;
static double nextRandom = 0.0;

static float rejectFlash = 0.0f;

// ---------------------------------------------------------------------------
// Ring initialization
// ---------------------------------------------------------------------------

static Ring makeRing(float innerR, float outerR, int side, bool pStyle, Vec4 uiColor)
{
    float a = (float)side * STEP;
    return Ring{innerR, outerR, side, pStyle, a, a, false, uiColor};
}

static void initRings()
{
    // Radii traced from the client logo; see ocp_logo_geom.h for the measured
    // pixel distances. Inner ring: C opening to the right (logo pose).
    rings[0] = makeRing(INNER_BAND_IN, INNER_BAND_OUT, 0, false,
                        Vec4{1.00f, 0.55f, 0.15f, 1.00f});

    // Middle + outer: shared backwards-Q break at lower-left (logo pose).
    rings[1] = makeRing(MID_BAND_IN, MID_BAND_OUT, 5, true,
                        Vec4{0.15f, 0.85f, 0.75f, 1.00f});
    rings[2] = makeRing(OUTER_BAND_IN, OUTER_BAND_OUT, 5, true,
                        Vec4{0.75f, 0.45f, 1.00f, 1.00f});
}

// ---------------------------------------------------------------------------
// Octagon wall geometry for visuals + collision
// ---------------------------------------------------------------------------

// Wall polygons and the circle/polygon tests come from ocp_logo_geom.h, so the
// shape the player collides with is the one the offline checkers measure.
static void ringLocalPolys(const Ring& r, vector<Poly>& polys)
{
    localSolidPolys(r.innerR, r.outerR, r.pStyle, polys);
}

static bool collidesWalls(float x, float y)
{
    Vec2 p{x, y};
    float rr = length(p);

    vector<Poly> polys;
    polys.reserve(8);

    for (int i = 0; i < 3; ++i)
    {
        const Ring& r = rings[i];

        // Conservative radial broad-phase. Clipping only removes material, so
        // a band still sits between its apothem and its circumradius.
        float minPossible = r.innerR * COS_HALF_STEP - DOT_RADIUS;
        float maxPossible = r.outerR + DOT_RADIUS;

        if (rr < minPossible || rr > maxPossible)
            continue;

        ringLocalPolys(r, polys);

        for (const Poly& lp : polys)
        {
            if (circleHitsPoly(p, DOT_RADIUS, rotatePoly(lp, (float)r.side * STEP)))
                return true;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------
// State updates
// ---------------------------------------------------------------------------

static float insideLimit()
{
    return rings[0].innerR - DOT_RADIUS * 1.25f;
}

static float hallwayLimit()
{
    return rings[2].outerR + DOT_RADIUS * 1.25f;
}

static void updateStates()
{
    anyAnimating = false;
    for (const Ring& r : rings)
        if (r.animating)
            anyAnimating = true;

    doorOpen =
        (rings[0].side == rings[1].side) &&
        (rings[1].side == rings[2].side);

    doorUsable = doorOpen && !anyAnimating;

    float rr = length(Vec2{playerX, playerY});

    float inLim = insideLimit();
    float hallLim = hallwayLimit();

    if (rr <= inLim)
        playerZone = ZONE_INSIDE;
    else if (rr >= hallLim)
        playerZone = ZONE_HALLWAY;
    else
        playerZone = ZONE_BAND;
}

static void updateRings(float dt)
{
    for (Ring& r : rings)
    {
        float diff = r.targetAngle - r.angle;

        if (fabsf(diff) > 1e-5f)
        {
            r.animating = true;

            float step = ROT_SPEED * dt;

            if (fabsf(diff) <= step)
            {
                r.angle = r.targetAngle;
                r.animating = false;
            }
            else
            {
                r.angle += (diff > 0.0f) ? step : -step;
            }
        }
        else
        {
            r.angle = r.targetAngle;
            r.animating = false;
        }

        // Keep unwrapped angle from growing forever.
        if (fabsf(r.angle) > 1000.0f)
        {
            float twoPi = 2.0f * PI;
            float shift = floorf(r.angle / twoPi) * twoPi;
            r.angle -= shift;
            r.targetAngle -= shift;
        }
    }
}

static bool rotateRing(int idx, int dir)
{
    if (idx < 0 || idx >= 3)
        return false;

    // Do not rotate while another ring is moving.
    if (anyAnimating)
        return false;

    // Only rotate while the person is safely inside the central container.
    // This also prevents rotation in hallway and prevents trapping in bands.
    if (playerZone != ZONE_INSIDE)
        return false;

    Ring& r = rings[idx];

    r.side = (r.side + dir + 8) % 8;
    r.targetAngle += (float)dir * STEP;
    r.animating = true;

    return true;
}

static bool attemptRandom(bool /*forced*/)
{
    if (anyAnimating)
        return false;

    if (playerZone != ZONE_INSIDE)
        return false;

    int idx = rand() % 3;
    int dir = (rand() % 2) ? 1 : -1;

    return rotateRing(idx, dir);
}

static void updateRandom(double now)
{
    if (now >= nextRandom)
    {
        if (attemptRandom(false))
            nextRandom = now + RANDOM_INTERVAL;
        else
            nextRandom = now + 1.0; // retry soon if blocked
    }
}

// ---------------------------------------------------------------------------
// Player movement / collision
// ---------------------------------------------------------------------------

static bool isValidPosition(float x, float y)
{
    Vec2 p{x, y};
    float rr = length(p);

    if (rr > WORLD_LIMIT)
        return false;

    if (collidesWalls(x, y))
        return false;

    float inLim = insideLimit();
    float hallLim = hallwayLimit();

    // The dot may not occupy the rotating band area unless the door is usable.
    if (!doorUsable && rr > inLim && rr < hallLim)
        return false;

    return true;
}

static void movePlayer(float dt)
{
    float vx = 0.0f;
    float vy = 0.0f;

    if (keys[GLFW_KEY_LEFT] || keys[GLFW_KEY_A]) vx -= 1.0f;
    if (keys[GLFW_KEY_RIGHT] || keys[GLFW_KEY_D]) vx += 1.0f;
    if (keys[GLFW_KEY_UP] || keys[GLFW_KEY_W]) vy += 1.0f;
    if (keys[GLFW_KEY_DOWN] || keys[GLFW_KEY_S]) vy -= 1.0f;

    float len = hypotf(vx, vy);
    if (len <= 1e-6f)
        return;

    vx /= len;
    vy /= len;

    float nx = playerX + vx * PLAYER_SPEED * dt;
    if (isValidPosition(nx, playerY))
        playerX = nx;

    float ny = playerY + vy * PLAYER_SPEED * dt;
    if (isValidPosition(playerX, ny))
        playerY = ny;
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

static void addRingVisual(DynamicMesh2D& m, const Ring& r, Vec4 color)
{
    vector<Poly> polys;
    ringLocalPolys(r, polys);

    // Local polys assume the opening at side 0; rotate the ring by r.angle.
    // The clipped stem and bar are convex, so a fan off vertex 0 is enough.
    for (const Poly& lp : polys)
    {
        Poly wp = rotatePoly(lp, r.angle);

        for (size_t i = 1; i + 1 < wp.size(); ++i)
            m.addTriangle(wp[0], wp[i], wp[i + 1], color);
    }
}

static void addDoorHighlight(DynamicMesh2D& m)
{
    if (!doorUsable)
        return;

    // Two hints, both in the aligned opening's frame: a wedge across the C
    // opening, then the straight corridor the backwards-Q notch leaves between
    // its stem and its bottom bar. That corridor runs parallel to side 7's flat
    // instead of straight out from the centre, so the way out is a dogleg.
    float mid = (float)rings[0].side * STEP;

    float c = cosf(mid);
    float s = sinf(mid);

    auto rot = [&](Vec2 p) -> Vec2
    {
        return Vec2{p.x * c - p.y * s, p.x * s + p.y * c};
    };

    Vec4 green{0.10f, 0.90f, 0.35f, 0.16f};

    float wedgeR = rings[0].outerR;

    m.addTriangle(
        rot(Vec2{0.0f, 0.0f}),
        rot(Vec2{wedgeR * cosf(-HALF_STEP), wedgeR * sinf(-HALF_STEP)}),
        rot(Vec2{wedgeR * cosf(HALF_STEP), wedgeR * sinf(HALF_STEP)}),
        green
    );

    // Offset out along side 7's normal, then run perpendicular to it.
    Vec2 n = sideNormal(7);
    Vec2 along{-n.y, n.x};

    float reach = rings[2].outerR + 0.25f;

    m.addQuad(
        rot(n * P_BAR_CLIP),
        rot(n * P_STEM_CLIP),
        rot(n * P_STEM_CLIP + along * reach),
        rot(n * P_BAR_CLIP + along * reach),
        green
    );
}

// ---------------------------------------------------------------------------
// GUI buttons
// ---------------------------------------------------------------------------

struct Button
{
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;

    int ring = 0;
    int dir = 0; // -1 CCW, +1 CW
};

static vector<Button> makeButtons(int fbW, int fbH)
{
    vector<Button> b;
    b.reserve(6);

    const float bw = 74.0f;
    const float bh = 58.0f;
    const float gap = 14.0f;
    const float groupGap = 34.0f;

    float total = 3.0f * (2.0f * bw + gap) + 2.0f * groupGap;
    float startX = ((float)fbW - total) * 0.5f;
    float y = (float)fbH - bh - 28.0f;

    for (int i = 0; i < 3; ++i)
    {
        float base = startX + (float)i * (2.0f * bw + gap + groupGap);

        b.push_back(Button{base, y, bw, bh, i, -1});
        b.push_back(Button{base + bw + gap, y, bw, bh, i, 1});
    }

    return b;
}

static void drawUI(int fbW, int fbH, double now)
{
    vector<Button> btns = makeButtons(fbW, fbH);

    for (const Button& btn : btns)
    {
        bool hover =
            mouseXPixels >= btn.x && mouseXPixels < btn.x + btn.w &&
            mouseYPixels >= btn.y && mouseYPixels < btn.y + btn.h;

        Vec4 bg{0.10f, 0.11f, 0.13f, hover ? 0.95f : 0.82f};

        g_mesh.addQuad(
            Vec2{btn.x, btn.y},
            Vec2{btn.x + btn.w, btn.y},
            Vec2{btn.x + btn.w, btn.y + btn.h},
            Vec2{btn.x, btn.y + btn.h},
            bg
        );

        Vec4 border = rings[btn.ring].uiColor;

        if (rejectFlash > 0.0f)
            border = Vec4{1.0f, 0.22f, 0.22f, 1.0f};

        float t = 3.0f;

        // Top border
        g_mesh.addQuad(
            Vec2{btn.x, btn.y},
            Vec2{btn.x + btn.w, btn.y},
            Vec2{btn.x + btn.w, btn.y + t},
            Vec2{btn.x, btn.y + t},
            border
        );

        // Bottom border
        g_mesh.addQuad(
            Vec2{btn.x, btn.y + btn.h - t},
            Vec2{btn.x + btn.w, btn.y + btn.h - t},
            Vec2{btn.x + btn.w, btn.y + btn.h},
            Vec2{btn.x, btn.y + btn.h},
            border
        );

        // Left border
        g_mesh.addQuad(
            Vec2{btn.x, btn.y},
            Vec2{btn.x + t, btn.y},
            Vec2{btn.x + t, btn.y + btn.h},
            Vec2{btn.x, btn.y + btn.h},
            border
        );

        // Right border
        g_mesh.addQuad(
            Vec2{btn.x + btn.w - t, btn.y},
            Vec2{btn.x + btn.w, btn.y},
            Vec2{btn.x + btn.w, btn.y + btn.h},
            Vec2{btn.x + btn.w - t, btn.y + btn.h},
            border
        );

        float cx = btn.x + btn.w * 0.5f;
        float cy = btn.y + btn.h * 0.5f + 6.0f;
        float s = 12.0f;

        Vec4 white{1.0f, 1.0f, 1.0f, 0.95f};

        if (btn.dir < 0)
        {
            // Counter-clockwise / left arrow.
            g_mesh.addTriangle(
                Vec2{cx + s, cy - s},
                Vec2{cx - s, cy},
                Vec2{cx + s, cy + s},
                white
            );
        }
        else
        {
            // Clockwise / right arrow.
            g_mesh.addTriangle(
                Vec2{cx - s, cy - s},
                Vec2{cx + s, cy},
                Vec2{cx - s, cy + s},
                white
            );
        }

        // Ring index indicator: 1 dot = inner, 2 dots = middle, 3 dots = outer.
        int count = btn.ring + 1;
        float dotY = btn.y + 13.0f;

        for (int n = 0; n < count; ++n)
        {
            float dx = cx + ((float)n - (float)(count - 1) * 0.5f) * 11.0f;
            g_mesh.addCircle(dx, dotY, 3.2f, 14, white);
        }
    }

    // Door status, top-left.
    {
        float cx = 48.0f;
        float cy = 48.0f;
        float r = 24.0f;

        Vec4 doorCol;
        if (doorUsable)
            doorCol = Vec4{0.15f, 0.95f, 0.35f, 1.0f};
        else
            doorCol = Vec4{1.00f, 0.22f, 0.22f, 1.0f};

        Vec4 fill = doorCol;
        fill.w = 0.18f;

        g_mesh.addCircle(cx, cy, r - 5.0f, 40, fill);
        g_mesh.addArc(cx, cy, r, 4.0f, 0.0f, 2.0f * PI, doorCol);
    }

    // Random rotation countdown, top-right.
    {
        float cx = (float)fbW - 56.0f;
        float cy = 56.0f;
        float r = 28.0f;

        double remaining = nextRandom - now;
        if (remaining < 0.0) remaining = 0.0;

        float progress = 1.0f - (float)(remaining / RANDOM_INTERVAL);
        if (progress < 0.0f) progress = 0.0f;
        if (progress > 1.0f) progress = 1.0f;

        Vec4 track{0.70f, 0.70f, 0.72f, 0.90f};
        g_mesh.addArc(cx, cy, r, 6.0f, 0.0f, 2.0f * PI, track);

        Vec4 progCol;
        if (playerZone == ZONE_INSIDE && !anyAnimating)
            progCol = Vec4{0.15f, 0.75f, 1.00f, 1.0f};
        else if (playerZone == ZONE_HALLWAY)
            progCol = Vec4{0.45f, 0.45f, 0.45f, 1.0f};
        else
            progCol = Vec4{1.00f, 0.75f, 0.15f, 1.0f};

        float start = -PI * 0.5f;
        g_mesh.addArc(cx, cy, r, 6.0f, start, start + 2.0f * PI * progress, progCol);
    }

    // Simple hallway indicator, top-center.
    if (playerZone == ZONE_HALLWAY)
    {
        Vec4 bar{0.15f, 0.90f, 0.35f, 0.25f};
        float cx = (float)fbW * 0.5f;

        g_mesh.addQuad(
            Vec2{cx - 90.0f, 18.0f},
            Vec2{cx + 90.0f, 18.0f},
            Vec2{cx + 90.0f, 34.0f},
            Vec2{cx - 90.0f, 34.0f},
            bar
        );
    }
}

static void render(double now)
{
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);

    if (fbW <= 0 || fbH <= 0)
        return;

    glViewport(0, 0, fbW, fbH);

    // Logo-like white background.
    glClearColor(0.96f, 0.96f, 0.97f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    float aspect = (float)fbW / (float)fbH;

    Mat4 worldProj = ortho2D(
        -WORLD_HALF_HEIGHT * aspect,
         WORLD_HALF_HEIGHT * aspect,
        -WORLD_HALF_HEIGHT,
         WORLD_HALF_HEIGHT,
        -1.0f,
         1.0f
    );

    g_mesh.clear();

    Vec4 ringColor{0.035f, 0.035f, 0.04f, 1.0f};

    for (int i = 0; i < 3; ++i)
        addRingVisual(g_mesh, rings[i], ringColor);

    addDoorHighlight(g_mesh);

    // Dot outline.
    Vec4 outline{0.04f, 0.04f, 0.05f, 0.95f};
    g_mesh.addCircle(playerX, playerY, DOT_RADIUS * 1.25f, 36, outline);

    // Dot color.
    Vec4 dotCol;
    if (playerZone == ZONE_HALLWAY)
        dotCol = Vec4{0.12f, 0.90f, 0.35f, 1.0f};
    else if (doorUsable)
        dotCol = Vec4{1.00f, 0.75f, 0.15f, 1.0f};
    else
        dotCol = Vec4{1.00f, 0.45f, 0.20f, 1.0f};

    g_mesh.addCircle(playerX, playerY, DOT_RADIUS, 36, dotCol);

    g_mesh.draw(worldProj);

    // UI in pixel coordinates.
    Mat4 uiProj = ortho2D(0.0f, (float)fbW, (float)fbH, 0.0f, -1.0f, 1.0f);

    g_mesh.clear();
    drawUI(fbW, fbH, now);
    g_mesh.draw(uiProj);
}

// ---------------------------------------------------------------------------
// GLFW callbacks
// ---------------------------------------------------------------------------

static void keyCallback(GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);

    if (key >= 0 && key < 512)
    {
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
            keys[key] = true;
        else if (action == GLFW_RELEASE)
            keys[key] = false;
    }

    if (action == GLFW_PRESS && key == GLFW_KEY_R)
    {
        if (attemptRandom(true))
            nextRandom = currentTime + RANDOM_INTERVAL;
        else
            rejectFlash = 0.35f;
    }
}

static void cursorCallback(GLFWwindow* w, double xpos, double ypos)
{
    int fbW = 0, fbH = 0;
    int winW = 0, winH = 0;

    glfwGetFramebufferSize(w, &fbW, &fbH);
    glfwGetWindowSize(w, &winW, &winH);

    if (winW <= 0 || winH <= 0)
        return;

    mouseXPixels = xpos * (double)fbW / (double)winW;
    mouseYPixels = ypos * (double)fbH / (double)winH;
}

static void mouseButtonCallback(GLFWwindow* w, int button, int action, int /*mods*/)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS)
        return;

    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(w, &fbW, &fbH);

    vector<Button> btns = makeButtons(fbW, fbH);

    for (const Button& b : btns)
    {
        bool hit =
            mouseXPixels >= b.x && mouseXPixels < b.x + b.w &&
            mouseYPixels >= b.y && mouseYPixels < b.y + b.h;

        if (hit)
        {
            if (!rotateRing(b.ring, b.dir))
                rejectFlash = 0.35f;

            break;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    window = glfwCreateWindow(1280, 800, "OCP Octagonal Ring Escape", nullptr, nullptr);
    if (!window)
    {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(glewErr));
        glfwTerminate();
        return 1;
    }

    while (glGetError() != GL_NO_ERROR) {}

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    g_program = linkProgram(vs, fs);
    g_uProj = glGetUniformLocation(g_program, "uProj");

    g_mesh.init();
    initRings();

    glEnable(GL_MULTISAMPLE);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    srand((unsigned)time(nullptr));

    double start = glfwGetTime();
    currentTime = start;
    nextRandom = start + RANDOM_INTERVAL;

    updateStates();

    double last = start;
    double accumulator = 0.0;
    const double physicsDt = 1.0 / 120.0;

    while (!glfwWindowShouldClose(window))
    {
        double now = glfwGetTime();
        double dt = now - last;
        last = now;

        if (dt > 0.1) dt = 0.1;

        currentTime = now;

        if (rejectFlash > 0.0f)
            rejectFlash -= (float)dt;

        updateRings((float)dt);
        updateStates();

        updateRandom(now);
        updateStates();

        accumulator += dt;
        while (accumulator >= physicsDt)
        {
            movePlayer((float)physicsDt);
            accumulator -= physicsDt;
        }

        updateStates();

        render(now);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
