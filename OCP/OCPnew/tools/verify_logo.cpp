// verify_logo.cpp
// Offline check that ocp_logo_geom.h really draws the client logo.
//
// Rasterises the logo pose (C band side 0, backwards-Q bands side 5) at the
// reference image's scale, then diffs it against a greyscale dump of
// ocp-2.jpg. Uses the same header as the demo, so a pass here means the on
// screen shape matches too.
//
// Build:  cl /EHsc /O2 /I.. verify_logo.cpp
// Run:    verify_logo ref.pgm [diff.ppm]

#include "../ocp_logo_geom.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std;

// Reference image frame: ocp-2.jpg, 195x195. The logo's centre measures 96.67
// in sample indices, and a sample sits at the middle of its pixel, so it lands
// at 97.17 in the continuous coordinates the rasteriser works in.
static const int IMG = 195;
static const float CENTRE = 97.17f;
static const float OUTER_APOTHEM_PX = 90.06f;
static const float SCALE = OUTER_APOTHEM_PX / (OUTER_BAND_OUT * COS_HALF_STEP);

// Supersampling per axis. 16 keeps coverage quantised finely enough that
// near-half-covered pixels along the flats are not decided by a tie.
static const int SUB = 16;

struct Band
{
    float innerR;
    float outerR;
    int side;
    bool pStyle;
};

static vector<unsigned char> loadPGM(const char* path, int& w, int& h)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }

    char magic[3] = {};
    int maxval = 0;
    if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxval) != 4 || string(magic) != "P5")
    {
        fprintf(stderr, "%s is not a binary PGM\n", path);
        exit(1);
    }
    fgetc(f);

    vector<unsigned char> px((size_t)w * h);
    if (fread(px.data(), 1, px.size(), f) != px.size())
    {
        fprintf(stderr, "%s is truncated\n", path);
        exit(1);
    }

    fclose(f);
    return px;
}

static void printRuns(const char* label, const vector<unsigned char>& mask, int w, int y)
{
    printf("  %-6s y=%3d :", label, y);

    int start = -1;
    for (int x = 0; x <= w; ++x)
    {
        bool ink = (x < w) && mask[(size_t)y * w + x] != 0;

        if (ink && start < 0)
            start = x;
        else if (!ink && start >= 0)
        {
            printf(" %d-%d", start, x - 1);
            start = -1;
        }
    }

    printf("\n");
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: verify_logo ref.pgm [diff.ppm]\n");
        return 1;
    }

    // Logo pose, matching initRings() in test.cpp.
    Band bands[3] = {
        {INNER_BAND_IN, INNER_BAND_OUT, 0, false},
        {MID_BAND_IN, MID_BAND_OUT, 5, true},
        {OUTER_BAND_IN, OUTER_BAND_OUT, 5, true}
    };

    // World-space walls for the whole logo.
    vector<Poly> walls;

    for (const Band& b : bands)
    {
        vector<Poly> local;
        localSolidPolys(b.innerR, b.outerR, b.pStyle, local);

        for (const Poly& lp : local)
            walls.push_back(rotatePoly(lp, (float)b.side * STEP));
    }

    printf("scale %.4f px/world, %d wall polygons\n", SCALE, (int)walls.size());

    // Rasterise with supersampling, then threshold at half coverage.
    vector<unsigned char> render((size_t)IMG * IMG, 0);

    for (int py = 0; py < IMG; ++py)
    {
        for (int px = 0; px < IMG; ++px)
        {
            int hits = 0;

            for (int sy = 0; sy < SUB; ++sy)
            {
                for (int sx = 0; sx < SUB; ++sx)
                {
                    float fx = (float)px + ((float)sx + 0.5f) / (float)SUB;
                    float fy = (float)py + ((float)sy + 0.5f) / (float)SUB;

                    // Image y grows downward, world y grows upward.
                    Vec2 w{(fx - CENTRE) / SCALE, -(fy - CENTRE) / SCALE};

                    for (const Poly& poly : walls)
                    {
                        if (pointInPoly(w, poly))
                        {
                            ++hits;
                            break;
                        }
                    }
                }
            }

            if (hits * 2 >= SUB * SUB)
                render[(size_t)py * IMG + px] = 255;
        }
    }

    // Reference mask.
    int rw = 0, rh = 0;
    vector<unsigned char> grey = loadPGM(argv[1], rw, rh);

    if (rw != IMG || rh != IMG)
    {
        fprintf(stderr, "reference is %dx%d, expected %dx%d\n", rw, rh, IMG, IMG);
        return 1;
    }

    vector<unsigned char> ref((size_t)IMG * IMG, 0);
    for (size_t i = 0; i < ref.size(); ++i)
        ref[i] = (grey[i] < 128) ? 255 : 0;

    // Diff.
    int refInk = 0;
    int renderInk = 0;
    int missing = 0; // logo has ink, we do not
    int extra = 0;   // we have ink, logo does not

    int minX = IMG, maxX = -1, minY = IMG, maxY = -1;

    for (int y = 0; y < IMG; ++y)
    {
        for (int x = 0; x < IMG; ++x)
        {
            bool r = ref[(size_t)y * IMG + x] != 0;
            bool o = render[(size_t)y * IMG + x] != 0;

            if (r) ++refInk;
            if (o) ++renderInk;

            if (r != o)
            {
                if (r) ++missing; else ++extra;

                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    int diff = missing + extra;

    printf("logo ink   %d px\n", refInk);
    printf("render ink %d px\n", renderInk);
    printf("mismatch   %d px (%.3f%% of image), missing %d, extra %d\n",
           diff, 100.0 * diff / (IMG * IMG), missing, extra);

    if (diff > 0)
        printf("mismatch bbox x[%d..%d] y[%d..%d]\n", minX, maxX, minY, maxY);

    // A pixel that disagrees only because an edge landed one pixel over is not
    // a shape difference: the same ink sits right next to it in the other
    // image. Whatever survives that slack is a feature one image has and the
    // other does not, which is what a wrong notch or stem would look like.
    int structural = 0;

    for (int y = 0; y < IMG; ++y)
    {
        for (int x = 0; x < IMG; ++x)
        {
            bool r = ref[(size_t)y * IMG + x] != 0;
            bool o = render[(size_t)y * IMG + x] != 0;

            if (r == o)
                continue;

            const vector<unsigned char>& other = r ? render : ref;

            bool nearby = false;

            for (int dy = -1; dy <= 1 && !nearby; ++dy)
            {
                for (int dx = -1; dx <= 1 && !nearby; ++dx)
                {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx < 0 || nx >= IMG || ny < 0 || ny >= IMG)
                        continue;

                    if (other[(size_t)ny * IMG + nx] != 0)
                        nearby = true;
                }
            }

            if (!nearby)
                ++structural;
        }
    }

    printf("structural mismatch (past one pixel of edge slack): %d px\n", structural);

    // Per-row mismatch histogram, worst rows first.
    vector<pair<int, int>> rowDiff;
    for (int y = 0; y < IMG; ++y)
    {
        int n = 0;
        for (int x = 0; x < IMG; ++x)
            if ((ref[(size_t)y * IMG + x] != 0) != (render[(size_t)y * IMG + x] != 0))
                ++n;

        if (n > 0)
            rowDiff.push_back(make_pair(n, y));
    }

    vector<pair<int, int>> worst = rowDiff;
    for (size_t i = 0; i < worst.size(); ++i)
        for (size_t j = i + 1; j < worst.size(); ++j)
            if (worst[j].first > worst[i].first)
                swap(worst[i], worst[j]);

    if (!worst.empty())
    {
        printf("rows with mismatches: %d (worst:", (int)rowDiff.size());

        for (size_t i = 0; i < worst.size() && i < 8; ++i)
            printf(" y%d:%d", worst[i].second, worst[i].first);

        printf(")\n");
    }

    // Scanline comparison across the notch, the stem and the bars.
    const int probes[] = {96, 115, 120, 130, 140, 150, 162, 170, 178, 186};

    printf("\nblack runs, logo vs render:\n");
    for (int p : probes)
    {
        printRuns("logo", ref, IMG, p);
        printRuns("render", render, IMG, p);
    }

    printf("\nblack runs on the worst mismatching rows:\n");
    for (size_t i = 0; i < worst.size() && i < 6; ++i)
    {
        printRuns("logo", ref, IMG, worst[i].second);
        printRuns("render", render, IMG, worst[i].second);
    }

    if (argc >= 3)
    {
        FILE* f = fopen(argv[2], "wb");
        if (f)
        {
            fprintf(f, "P6\n%d %d\n255\n", IMG, IMG);

            for (size_t i = 0; i < ref.size(); ++i)
            {
                bool r = ref[i] != 0;
                bool o = render[i] != 0;

                unsigned char rgb[3];

                if (r && o)       { rgb[0] = 40;  rgb[1] = 40;  rgb[2] = 40;  } // agreed ink
                else if (!r && !o) { rgb[0] = 255; rgb[1] = 255; rgb[2] = 255; } // agreed paper
                else if (r)        { rgb[0] = 230; rgb[1] = 30;  rgb[2] = 30;  } // logo only
                else               { rgb[0] = 30;  rgb[1] = 90;  rgb[2] = 230; } // render only

                fwrite(rgb, 1, 3, f);
            }

            fclose(f);
            printf("\nwrote %s (red = logo only, blue = render only)\n", argv[2]);
        }
    }

    if (argc >= 4)
    {
        FILE* f = fopen(argv[3], "wb");
        if (f)
        {
            fprintf(f, "P6\n%d %d\n255\n", IMG, IMG);

            for (size_t i = 0; i < render.size(); ++i)
            {
                unsigned char v = render[i] ? 0 : 255;
                unsigned char rgb[3] = {v, v, v};
                fwrite(rgb, 1, 3, f);
            }

            fclose(f);
            printf("wrote %s (rasterised geometry)\n", argv[3]);
        }
    }

    return 0;
}
