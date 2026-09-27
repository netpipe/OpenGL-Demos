// ocp_logo_geom.h
// Octagonal OCP logo band geometry. Pure math, no GL / GLFW / windowing.
//
// Shared by the interactive demo (test.cpp) and the offline reference checker
// (tools/verify_logo.cpp) so both always describe the same shape. Earlier
// revisions kept a separate offline model, which is how f602603 shipped a
// notch that measured open offline and rendered sealed on screen.
//
// Every length below is a world unit with the outer band's circumradius pinned
// at 3.20, and every "(N px)" comment is the matching distance measured from
// the client logo (ocp-2.jpg, 195x195, centre 96.67, outer apothem 90.06 px).

#ifndef OCP_LOGO_GEOM_H
#define OCP_LOGO_GEOM_H

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Octagon constants
// ---------------------------------------------------------------------------

static const float PI = 3.14159265358979323846f;

// Octagon: 8 sides, 45 degrees per side.
static const float STEP = PI / 4.0f;
static const float HALF_STEP = STEP / 2.0f;

// cos(22.5): turns a circumradius into the octagon's apothem, i.e. the
// centre-to-flat distance that the logo scanlines actually measure.
static const float COS_HALF_STEP = 0.92387953f;

// ---------------------------------------------------------------------------
// Band radii (circumradius), traced from the logo
// ---------------------------------------------------------------------------

static const float INNER_BAND_IN = 0.7917f;  // 22.28 px
static const float INNER_BAND_OUT = 1.5087f; // 42.46 px
static const float MID_BAND_IN = 1.5900f;    // 44.75 px
static const float MID_BAND_OUT = 2.3064f;   // 64.91 px
static const float OUTER_BAND_IN = 2.3885f;  // 67.22 px
static const float OUTER_BAND_OUT = 3.2000f; // 90.06 px

// ---------------------------------------------------------------------------
// Backwards-Q notch
// ---------------------------------------------------------------------------
//
// The logo does not break the outer two bands by shortening an octagon side.
// Both bands are cut by two lines parallel to local side 7's flat, which stand
// vertical in the logo pose, and the strip between them is the notch corridor:
//
// - P_STEM_CLIP keeps local side 0 only where it is at least this far out along
//   side 7's normal. The band therefore runs straight past the octagon corner
//   and dies in a point on the line instead of turning, which is what draws the
//   logo's long bottom-left stem. It sits exactly on the middle band's inner
//   flat, so that band's inner edge reads as one unbroken straight line.
// - P_BAR_CLIP keeps local side 1 only where it is within this distance, giving
//   the bottom bar a square free end.
//
// Both are absolute distances, not per-side fractions, so the middle and outer
// bands share one stem line and one bar line the way the logo does.
static const float P_STEM_CLIP = MID_BAND_IN * COS_HALF_STEP; // 44.75 px
static const float P_BAR_CLIP = 0.6884f;                      // 20.97 px

// ---------------------------------------------------------------------------
// Tiny 2D math, no GLM
// ---------------------------------------------------------------------------

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

static inline Vec2 operator+(Vec2 a, Vec2 b)
{
    return Vec2{a.x + b.x, a.y + b.y};
}

static inline Vec2 operator-(Vec2 a, Vec2 b)
{
    return Vec2{a.x - b.x, a.y - b.y};
}

static inline Vec2 operator*(Vec2 a, float s)
{
    return Vec2{a.x * s, a.y * s};
}

static inline float dot2(Vec2 a, Vec2 b)
{
    return a.x * b.x + a.y * b.y;
}

static inline float cross2(Vec2 a, Vec2 b)
{
    return a.x * b.y - a.y * b.x;
}

static inline float length2(Vec2 a)
{
    return dot2(a, a);
}

static inline float length(Vec2 a)
{
    return sqrtf(length2(a));
}

// ---------------------------------------------------------------------------
// Convex wall polygons
// ---------------------------------------------------------------------------

typedef std::vector<Vec2> Poly;

// Outward normal of one octagon side.
static inline Vec2 sideNormal(int localSide)
{
    float a = (float)localSide * STEP;
    return Vec2{cosf(a), sinf(a)};
}

// One full octagon side as a trapezoid, wound CCW from the side's CCW-start
// vertex at the inner radius.
static Poly makeFlatSidePoly(float innerR, float outerR, int localSide)
{
    float a0 = (float)localSide * STEP - HALF_STEP;
    float a1 = (float)localSide * STEP + HALF_STEP;

    Poly p;
    p.reserve(4);
    p.push_back(Vec2{innerR * cosf(a0), innerR * sinf(a0)});
    p.push_back(Vec2{outerR * cosf(a0), outerR * sinf(a0)});
    p.push_back(Vec2{outerR * cosf(a1), outerR * sinf(a1)});
    p.push_back(Vec2{innerR * cosf(a1), innerR * sinf(a1)});
    return p;
}

// Sutherland-Hodgman: keep the part of a convex polygon where dot(p, n) <= d.
static Poly clipHalfPlane(const Poly& in, Vec2 n, float d)
{
    Poly out;
    out.reserve(in.size() + 2);

    int count = (int)in.size();

    for (int i = 0; i < count; ++i)
    {
        Vec2 a = in[i];
        Vec2 b = in[(i + 1) % count];

        float da = dot2(a, n) - d;
        float db = dot2(b, n) - d;

        if (da <= 0.0f)
            out.push_back(a);

        if ((da < 0.0f && db > 0.0f) || (da > 0.0f && db < 0.0f))
        {
            float t = da / (da - db);
            out.push_back(a + (b - a) * t);
        }
    }

    return out;
}

// Solid wall polygons in a band's LOCAL frame, where the opening faces local
// side 0. Callers rotate the result: the visuals by the animated angle, the
// collision by side * STEP.
//
// C band: drop local side 0 outright, leaving radial cut ends.
// Backwards-Q band: keep all eight sides but clip the stem (side 0) and the
// bottom bar (side 1) against the two notch lines described above.
static void localSolidPolys(float innerR, float outerR, bool pStyle, std::vector<Poly>& polys)
{
    polys.clear();

    if (!pStyle)
    {
        for (int localSide = 1; localSide < 8; ++localSide)
            polys.push_back(makeFlatSidePoly(innerR, outerR, localSide));

        return;
    }

    Vec2 n7 = sideNormal(7);

    // Stem: keep dot(p, n7) >= P_STEM_CLIP.
    Poly stem = clipHalfPlane(
        makeFlatSidePoly(innerR, outerR, 0),
        Vec2{-n7.x, -n7.y},
        -P_STEM_CLIP
    );

    if (stem.size() >= 3)
        polys.push_back(stem);

    // Bottom bar: keep dot(p, n7) <= P_BAR_CLIP.
    Poly bar = clipHalfPlane(
        makeFlatSidePoly(innerR, outerR, 1),
        n7,
        P_BAR_CLIP
    );

    if (bar.size() >= 3)
        polys.push_back(bar);

    for (int localSide = 2; localSide < 8; ++localSide)
        polys.push_back(makeFlatSidePoly(innerR, outerR, localSide));
}

// Rotate a local-frame polygon into world space.
static Poly rotatePoly(const Poly& p, float angle)
{
    float c = cosf(angle);
    float s = sinf(angle);

    Poly out;
    out.reserve(p.size());

    for (Vec2 v : p)
        out.push_back(Vec2{v.x * c - v.y * s, v.x * s + v.y * c});

    return out;
}

// ---------------------------------------------------------------------------
// Collision primitives
// ---------------------------------------------------------------------------

static bool pointInPoly(Vec2 p, const Poly& poly)
{
    bool hasPos = false;
    bool hasNeg = false;

    int n = (int)poly.size();

    for (int i = 0; i < n; ++i)
    {
        Vec2 a = poly[i];
        Vec2 b = poly[(i + 1) % n];

        float c = cross2(b - a, p - a);

        if (c > 1e-5f) hasPos = true;
        if (c < -1e-5f) hasNeg = true;

        if (hasPos && hasNeg)
            return false;
    }

    return true;
}

static float distPointSeg(Vec2 p, Vec2 a, Vec2 b)
{
    Vec2 ab = b - a;
    float len = length2(ab);

    if (len < 1e-8f)
        return length(p - a);

    float t = dot2(p - a, ab) / len;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    return length(p - (a + ab * t));
}

static bool circleHitsPoly(Vec2 c, float radius, const Poly& poly)
{
    if (pointInPoly(c, poly))
        return true;

    int n = (int)poly.size();

    for (int i = 0; i < n; ++i)
    {
        if (distPointSeg(c, poly[i], poly[(i + 1) % n]) < radius - 1e-5f)
            return true;
    }

    return false;
}

#endif // OCP_LOGO_GEOM_H
