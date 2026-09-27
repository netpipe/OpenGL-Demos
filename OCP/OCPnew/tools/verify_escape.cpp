// verify_escape.cpp
// Checks that the dot can still get out once the doorway lines up, and that it
// cannot when the doorway is shut.
//
// The logo's notch is not a radial slot: the corridor between the stem and the
// bottom bar runs parallel to local side 7's flat, so the way out is a dogleg
// through the C opening and then along the corridor. A grid flood fill is the
// honest way to confirm the dot actually fits through that turn.
//
// Build:  cl /EHsc /O2 verify_escape.cpp
// Run:    verify_escape

#include "../ocp_logo_geom.h"

#include <cstdio>
#include <deque>
#include <vector>

using namespace std;

static const float DOT_RADIUS = 0.13f; // must match test.cpp

struct Band
{
    float innerR;
    float outerR;
    int side;
    bool pStyle;
};

static vector<Poly> buildWalls(int innerSide, int qSide)
{
    Band bands[3] = {
        {INNER_BAND_IN, INNER_BAND_OUT, innerSide, false},
        {MID_BAND_IN, MID_BAND_OUT, qSide, true},
        {OUTER_BAND_IN, OUTER_BAND_OUT, qSide, true}
    };

    vector<Poly> walls;

    for (const Band& b : bands)
    {
        vector<Poly> local;
        localSolidPolys(b.innerR, b.outerR, b.pStyle, local);

        for (const Poly& lp : local)
            walls.push_back(rotatePoly(lp, (float)b.side * STEP));
    }

    return walls;
}

static bool blocked(Vec2 p, const vector<Poly>& walls)
{
    for (const Poly& w : walls)
        if (circleHitsPoly(p, DOT_RADIUS, w))
            return true;

    return false;
}

// Flood fill from the centre; report whether any reachable cell escapes past
// the outer band. Cell size well under DOT_RADIUS so the dot cannot tunnel.
static bool canEscape(const vector<Poly>& walls, float& reachedRadius)
{
    const float limit = OUTER_BAND_OUT + 0.45f;
    const float cell = 0.02f;
    const int half = (int)(limit / cell) + 2;
    const int dim = half * 2 + 1;

    vector<unsigned char> seen((size_t)dim * dim, 0);

    auto index = [&](int ix, int iy) -> size_t
    {
        return (size_t)(iy + half) * dim + (ix + half);
    };

    deque<pair<int, int>> queue;

    seen[index(0, 0)] = 1;
    queue.push_back(make_pair(0, 0));

    reachedRadius = 0.0f;
    bool escaped = false;

    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    while (!queue.empty())
    {
        pair<int, int> cur = queue.front();
        queue.pop_front();

        for (int k = 0; k < 4; ++k)
        {
            int nx = cur.first + dx[k];
            int ny = cur.second + dy[k];

            if (nx < -half || nx > half || ny < -half || ny > half)
                continue;

            if (seen[index(nx, ny)])
                continue;

            Vec2 p{(float)nx * cell, (float)ny * cell};
            float rr = length(p);

            if (rr > limit)
                continue;

            if (blocked(p, walls))
                continue;

            seen[index(nx, ny)] = 1;

            if (rr > reachedRadius)
                reachedRadius = rr;

            if (rr > OUTER_BAND_OUT + 0.25f)
                escaped = true;

            queue.push_back(make_pair(nx, ny));
        }
    }

    return escaped;
}

int main()
{
    printf("dot radius %.2f, corridor width %.3f (stem %.4f, bar %.4f)\n\n",
           DOT_RADIUS, P_STEM_CLIP - P_BAR_CLIP, P_STEM_CLIP, P_BAR_CLIP);

    int failures = 0;

    // Aligned: every band on the same side, which is what doorOpen tests.
    printf("aligned poses (doorOpen true) should escape:\n");
    for (int s = 0; s < 8; ++s)
    {
        float reached = 0.0f;
        bool escaped = canEscape(buildWalls(s, s), reached);

        printf("  side %d/%d/%d : %-7s reach %.2f\n",
               s, s, s, escaped ? "ESCAPE" : "sealed", reached);

        if (!escaped)
            ++failures;
    }

    // The logo pose and other mismatches must stay shut.
    printf("\nmisaligned poses (doorOpen false) should stay sealed:\n");
    for (int inner = 0; inner < 8; ++inner)
    {
        for (int q = 0; q < 8; ++q)
        {
            if (inner == q)
                continue;

            float reached = 0.0f;
            bool escaped = canEscape(buildWalls(inner, q), reached);

            if (escaped)
            {
                printf("  side %d/%d/%d : LEAK reach %.2f\n", inner, q, q, reached);
                ++failures;
            }
        }
    }

    printf("  %s\n", failures ? "leaks found" : "all 56 misaligned poses sealed");

    // The logo pose itself, called out because it is the shipped look.
    float reached = 0.0f;
    bool escaped = canEscape(buildWalls(0, 5), reached);
    printf("\nlogo pose 0/5/5 : %s (reach %.2f)\n",
           escaped ? "ESCAPE" : "sealed", reached);

    printf("\n%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
