// ============================================================
// OPTIMIZED CPU PERCEPTUAL / FOVEATED RAY TRACER
// ============================================================
//
// CPU ONLY:
//
//   Ray tracing       -> CPU
//   Reconstruction    -> CPU
//   Accumulation      -> CPU
//   OpenGL            -> display only
//   GLFW              -> window/input
//   GLEW              -> OpenGL loading
//   libpng            -> screenshots
//
// Designed to remain considerably more responsive on
// relatively slow CPUs.
//
// ============================================================
//
// CONTROLS
//
//   W A S D          Move
//   SPACE            Move up
//   LEFT SHIFT       Move down
//   Hold RMB         Look
//   R                Reset accumulation
//   P                Save screenshot.png
//   ESC              Quit
//
// ============================================================
//
// IMPORTANT PERFORMANCE SETTINGS
//
// Internal render resolution is:
//
//      480 x 270
//
// Window resolution is:
//
//      960 x 540
//
// OpenGL scales the image to the window.
//
// This is intentional.
//
// 960x540 = 518,400 pixels
// 480x270 = 129,600 pixels
//
// That's 4x fewer pixels for the CPU.
//
// ============================================================
//
// Linux example:
//
// g++ test.cpp -std=c++17 -O3 \
//     -march=native \
//     -lglfw -lGLEW -lGL -lpng -pthread \
//     -o perceptual_rt
//
// ============================================================

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <thread>
#include <chrono>
#include <vector>

// ============================================================
// Resolution
// ============================================================

static constexpr int WINDOW_WIDTH  = 960;
static constexpr int WINDOW_HEIGHT = 540;

// CPU rendering resolution.
// Keep this low on a ~2 GHz CPU.

static constexpr int RENDER_WIDTH  = 480;
static constexpr int RENDER_HEIGHT = 270;


// ============================================================
// Constants
// ============================================================

static constexpr float PI =
    3.14159265358979323846f;

static constexpr float EPSILON =
    0.001f;

static constexpr double TARGET_FPS =
    15.0;

// Perceptual/foveated sampling. The center of the image gets
// more work; the far periphery is refined on alternating frames.
static constexpr float FOVEA_RADIUS = 0.24f;
static constexpr float MID_RADIUS   = 0.62f;

static constexpr double TARGET_FRAME_TIME =
    1.0 / TARGET_FPS;


// ============================================================
// Vec3
// ============================================================

struct Vec3
{
    float x;
    float y;
    float z;

    Vec3()
        : x(0), y(0), z(0)
    {
    }

    Vec3(float v)
        : x(v), y(v), z(v)
    {
    }

    Vec3(
        float X,
        float Y,
        float Z)
        : x(X), y(Y), z(Z)
    {
    }

    Vec3 operator+(const Vec3& b) const
    {
        return Vec3(
            x + b.x,
            y + b.y,
            z + b.z
        );
    }

    Vec3 operator-(const Vec3& b) const
    {
        return Vec3(
            x - b.x,
            y - b.y,
            z - b.z
        );
    }

    Vec3 operator-() const
    {
        return Vec3(
            -x,
            -y,
            -z
        );
    }

    Vec3 operator*(float b) const
    {
        return Vec3(
            x * b,
            y * b,
            z * b
        );
    }

    Vec3 operator/(float b) const
    {
        float inv = 1.0f / b;

        return Vec3(
            x * inv,
            y * inv,
            z * inv
        );
    }

    Vec3& operator+=(const Vec3& b)
    {
        x += b.x;
        y += b.y;
        z += b.z;

        return *this;
    }

    Vec3& operator-=(const Vec3& b)
    {
        x -= b.x;
        y -= b.y;
        z -= b.z;

        return *this;
    }

    Vec3& operator*=(float b)
    {
        x *= b;
        y *= b;
        z *= b;

        return *this;
    }

    Vec3& operator/=(float b)
    {
        x /= b;
        y /= b;
        z /= b;

        return *this;
    }
};


static Vec3 operator*(
    float a,
    const Vec3& b)
{
    return b * a;
}


// ============================================================
// Math
// ============================================================

static inline float dot(
    const Vec3& a,
    const Vec3& b)
{
    return
        a.x * b.x +
        a.y * b.y +
        a.z * b.z;
}


static inline Vec3 cross(
    const Vec3& a,
    const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}


static inline float lengthSq(
    const Vec3& v)
{
    return dot(v, v);
}


static inline float length(
    const Vec3& v)
{
    return std::sqrt(
        lengthSq(v)
    );
}


static inline Vec3 normalize(
    const Vec3& v)
{
    float l2 =
        lengthSq(v);

    if(l2 < 0.00000001f)
        return Vec3(0);

    float inv =
        1.0f / std::sqrt(l2);

    return v * inv;
}


static inline Vec3 multiply(
    const Vec3& a,
    const Vec3& b)
{
    return Vec3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}


static inline float clamp01(float x)
{
    return std::max(
        0.0f,
        std::min(
            1.0f,
            x
        )
    );
}


static inline Vec3 clamp01(
    const Vec3& v)
{
    return Vec3(
        clamp01(v.x),
        clamp01(v.y),
        clamp01(v.z)
    );
}


// ============================================================
// RNG
// ============================================================

struct RNG
{
    uint32_t state;

    explicit RNG(uint32_t s)
        : state(s)
    {
        if(state == 0)
            state = 1;
    }

    inline uint32_t nextUInt()
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        return state;
    }

    inline float next()
    {
        return
            (nextUInt() & 0x00ffffff) *
            (1.0f / 16777216.0f);
    }
};


// ============================================================
// Ray
// ============================================================

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};


// ============================================================
// Material
// ============================================================

struct Material
{
    Vec3 albedo;

    float roughness;

    // 0 = diffuse
    // 1 = glass/dielectric

    float transmission;

    // Index of refraction.
    // Glass is approximately 1.5.

    float ior;

    float emission;
};


// ============================================================
// Hit
// ============================================================

struct Hit
{
    float t;

    Vec3 position;
    Vec3 normal;

    Material material;

    bool hit;

    Hit()
        : t(std::numeric_limits<float>::max()),
          position(),
          normal(),
          material(),
          hit(false)
    {
    }
};


// ============================================================
// Sphere
// ============================================================

struct Sphere
{
    Vec3 center;

    float radius;

    Material material;


    inline bool intersect(
        const Ray& ray,
        Hit& hit) const
    {
        Vec3 oc =
            ray.origin -
            center;

        float b =
            dot(
                oc,
                ray.direction
            );

        float c =
            dot(oc, oc) -
            radius * radius;

        float h =
            b * b - c;

        if(h < 0.0f)
            return false;

        h = std::sqrt(h);

        float t =
            -b - h;

        if(t < EPSILON)
            t = -b + h;

        if(t < EPSILON)
            return false;

        if(t >= hit.t)
            return false;

        hit.t =
            t;

        hit.position =
            ray.origin +
            ray.direction * t;

        hit.normal =
            normalize(
                hit.position -
                center
            );

        hit.material =
            material;

        hit.hit =
            true;

        return true;
    }
};


// ============================================================
// Plane
// ============================================================

struct Plane
{
    Vec3 point;

    Vec3 normal;

    Material material;


    inline bool intersect(
        const Ray& ray,
        Hit& hit) const
    {
        float denom =
            dot(
                ray.direction,
                normal
            );

        if(std::fabs(denom) <
           0.00001f)
            return false;

        float t =
            dot(
                point - ray.origin,
                normal
            ) / denom;

        if(t < EPSILON)
            return false;

        if(t >= hit.t)
            return false;

        hit.t =
            t;

        hit.position =
            ray.origin +
            ray.direction * t;

        hit.normal =
            normal;

        hit.material =
            material;

        hit.hit =
            true;

        return true;
    }
};


// ============================================================
// Scene
// ============================================================

static std::vector<Sphere>
    spheres;

static Plane
    floorPlane;


// ============================================================
// Scene intersection
// ============================================================

static inline bool intersectScene(
    const Ray& ray,
    Hit& hit)
{
    for(const Sphere& s : spheres)
    {
        s.intersect(
            ray,
            hit
        );
    }

    floorPlane.intersect(
        ray,
        hit
    );

    return hit.hit;
}


// ============================================================
// Cosine hemisphere sample
// ============================================================

static inline Vec3 randomHemisphere(
    const Vec3& normal,
    RNG& rng)
{
    float u1 =
        rng.next();

    float u2 =
        rng.next();

    float r =
        std::sqrt(u1);

    float theta =
        2.0f * PI * u2;

    float x =
        r * std::cos(theta);

    float y =
        r * std::sin(theta);

    float z =
        std::sqrt(
            std::max(
                0.0f,
                1.0f - u1
            )
        );

    Vec3 tangent;

    if(std::fabs(normal.x) >
       0.1f)
    {
        tangent =
            normalize(
                cross(
                    Vec3(0, 1, 0),
                    normal
                )
            );
    }
    else
    {
        tangent =
            normalize(
                cross(
                    Vec3(1, 0, 0),
                    normal
                )
            );
    }

    Vec3 bitangent =
        cross(
            normal,
            tangent
        );

    return normalize(
        tangent * x +
        bitangent * y +
        normal * z
    );
}


// ============================================================
// Reflection
// ============================================================

static inline Vec3 reflectVec(
    const Vec3& incident,
    const Vec3& normal)
{
    return
        incident -
        normal *
        (2.0f *
         dot(incident, normal));
}


// ============================================================
// Refraction
// ============================================================

static bool refractVec(
    const Vec3& incident,
    Vec3 normal,
    float eta,
    Vec3& result)
{
    float cosi =
        std::clamp(
            -dot(incident, normal),
            -1.0f,
            1.0f
        );

    float etai = 1.0f;
    float etat = eta;

    if(cosi < 0.0f)
    {
        cosi = -cosi;

        std::swap(
            etai,
            etat
        );

        normal =
            -normal;
    }

    float etaRatio =
        etai / etat;

    float k =
        1.0f -
        etaRatio *
        etaRatio *
        (1.0f - cosi * cosi);

    if(k < 0.0f)
        return false;

    result =
        incident * etaRatio +
        normal *
        (
            etaRatio * cosi -
            std::sqrt(k)
        );

    return true;
}


// ============================================================
// Schlick Fresnel
// ============================================================

static inline float fresnel(
    const Vec3& incident,
    const Vec3& normal,
    float ior)
{
    float cosi =
        std::clamp(
            -dot(incident, normal),
            0.0f,
            1.0f
        );

    float r0 =
        (1.0f - ior) /
        (1.0f + ior);

    r0 *= r0;

    float oneMinus =
        1.0f - cosi;

    return
        r0 +
        (1.0f - r0) *
        oneMinus *
        oneMinus *
        oneMinus *
        oneMinus *
        oneMinus;
}


// ============================================================
// Direct shadow test
// ============================================================

static inline bool visibleToLight(
    const Vec3& position,
    const Vec3& normal,
    const Vec3& lightPosition)
{
    Vec3 d =
        lightPosition -
        position;

    float dist2 =
        lengthSq(d);

    float invDist =
        1.0f /
        std::sqrt(dist2);

    Vec3 direction =
        d * invDist;

    Ray shadow;

    shadow.origin =
        position +
        normal * 0.003f;

    shadow.direction =
        direction;

    Hit hit;

    if(intersectScene(
        shadow,
        hit))
    {
        return
            hit.t >
            (1.0f / invDist) -
            0.01f;
    }

    return true;
}


// ============================================================
// Sky
// ============================================================

static inline Vec3 skyColor(
    const Vec3& direction)
{
    float t =
        0.5f *
        (direction.y + 1.0f);

    // Keep sky cheap.

    return
        Vec3(
            0.52f,
            0.68f,
            0.95f
        ) * (1.0f - t)
        +
        Vec3(
            0.10f,
            0.16f,
            0.30f
        ) * t;
}


// ============================================================
// Path tracer
// ============================================================
//
// Deliberately limited to 3 bounces.
// This is much cheaper than a high-quality path tracer.
//
// ============================================================

static Vec3 trace(
    Ray ray,
    RNG& rng,
    Hit* firstHitOut = nullptr)
{
    Vec3 radiance(0.0f);

    Vec3 throughput(1.0f);


    const Vec3 lightPosition(
        -3.0f,
        5.5f,
        -2.0f
    );


    const Vec3 lightColor(
        7.0f,
        7.0f,
        7.0f
    );


    for(int bounce = 0;
        bounce < 3;
        ++bounce)
    {
        Hit hit;


        if(!intersectScene(
            ray,
            hit))
        {
            radiance +=
                multiply(
                    throughput,
                    skyColor(
                        ray.direction
                    )
                );

            break;
        }


        // Save the primary hit so reconstruction does not need
        // to traverse the entire scene a second time.
        if(bounce == 0 && firstHitOut)
        {
            *firstHitOut = hit;
        }

        const Material& m =
            hit.material;


        // ----------------------------------------------------
        // Emissive material
        // ----------------------------------------------------

        if(m.emission > 0.0f)
        {
            radiance +=
                multiply(
                    throughput,
                    m.albedo *
                    m.emission
                );

            break;
        }


        // ----------------------------------------------------
        // Glass / dielectric
        // ----------------------------------------------------
        //
        // Perceptual cheat: instead of Russian-roulette choosing
        // reflection OR refraction, keep both visible. Refraction
        // remains a real ray, while reflection uses a cheap stable
        // environment estimate. This makes the glass stay clear and
        // prevents black/unstable reflections while the camera moves.
        //

        if(m.transmission > 0.5f)
        {
            float F =
                fresnel(
                    ray.direction,
                    hit.normal,
                    m.ior
                );

            Vec3 reflected =
                normalize(
                    reflectVec(
                        ray.direction,
                        hit.normal
                    )
                );

            Vec3 reflectionColor =
                skyColor(reflected);

            // Slightly brighten the environment term so the
            // glass highlight survives low sample counts.
            reflectionColor *= 1.12f;

            Vec3 refracted;

            bool canRefract =
                refractVec(
                    ray.direction,
                    hit.normal,
                    m.ior,
                    refracted
                );

            if(!canRefract)
            {
                radiance +=
                    multiply(
                        throughput,
                        reflectionColor
                    );

                ray.origin =
                    hit.position +
                    hit.normal *
                    0.004f;

                ray.direction =
                    reflected;

                throughput *=
                    0.35f;

                continue;
            }

            // Reflection is added immediately; refraction continues
            // through the object and can see the scene behind it.
            radiance +=
                multiply(
                    throughput,
                    reflectionColor * F
                );

            throughput *=
                (1.0f - F);

            throughput =
                multiply(
                    throughput,
                    m.albedo
                );

            ray.origin =
                hit.position -
                hit.normal *
                0.004f;

            ray.direction =
                normalize(
                    refracted
                );

            continue;
        }


        // ----------------------------------------------------
        // Diffuse direct lighting
        // ----------------------------------------------------

        Vec3 toLight =
            lightPosition -
            hit.position;


        float lightDistance2 =
            lengthSq(toLight);


        float invLightDistance =
            1.0f /
            std::sqrt(
                lightDistance2
            );


        Vec3 lightDirection =
            toLight *
            invLightDistance;


        float NdotL =
            std::max(
                0.0f,
                dot(
                    hit.normal,
                    lightDirection
                )
            );


        if(NdotL > 0.0f &&
           visibleToLight(
               hit.position,
               hit.normal,
               lightPosition))
        {
            float attenuation =
                1.0f /
                (
                    1.0f +
                    0.10f *
                    lightDistance2
                );


            Vec3 direct =
                multiply(
                    m.albedo,
                    lightColor
                ) *
                (
                    NdotL *
                    attenuation
                );


            radiance +=
                multiply(
                    throughput,
                    direct
                );
        }


        // ----------------------------------------------------
        // Diffuse bounce
        // ----------------------------------------------------

        throughput =
            multiply(
                throughput,
                m.albedo
            );


        // ----------------------------------------------------
        // Russian roulette
        // ----------------------------------------------------

        if(bounce == 2)
        {
            float p =
                std::max(
                    0.15f,
                    std::min(
                        0.90f,
                        std::max(
                            throughput.x,
                            std::max(
                                throughput.y,
                                throughput.z
                            )
                        )
                    )
                );


            if(rng.next() > p)
                break;


            throughput /=
                p;
        }


        ray.origin =
            hit.position +
            hit.normal *
            0.003f;


        ray.direction =
            randomHemisphere(
                hit.normal,
                rng
            );
    }


    return radiance;
}


// ============================================================
// Camera
// ============================================================

struct Camera
{
    Vec3 position;

    float yaw;

    float pitch;

    float fov;


    Ray makeRay(
        float px,
        float py,
        float aspect,
        RNG& rng) const
    {
        float jitterX =
            rng.next() - 0.5f;

        float jitterY =
            rng.next() - 0.5f;


        float x =
            (
                (px + jitterX) /
                float(RENDER_WIDTH)
            ) * 2.0f - 1.0f;


        float y =
            1.0f -
            (
                (py + jitterY) /
                float(RENDER_HEIGHT)
            ) * 2.0f;


        float tanFov =
            std::tan(
                fov *
                0.5f *
                PI /
                180.0f
            );


        x *=
            aspect *
            tanFov;


        y *=
            tanFov;


        Vec3 forward(
            std::cos(pitch) *
            std::sin(yaw),

            std::sin(pitch),

            std::cos(pitch) *
            std::cos(yaw)
        );


        forward =
            normalize(
                forward
            );


        Vec3 right =
            normalize(
                cross(
                    forward,
                    Vec3(0, 1, 0)
                )
            );


        Vec3 up =
            normalize(
                cross(
                    right,
                    forward
                )
            );


        Ray ray;

        ray.origin =
            position;


        ray.direction =
            normalize(
                forward +
                right * x +
                up * y
            );


        return ray;
    }
};


// ============================================================
// Pixel metadata
// ============================================================

struct Pixel
{
    Vec3 color;

    Vec3 normal;

    float depth;

    Pixel()
        : color(0),
          normal(0),
          depth(0)
    {
    }
};


// ============================================================
// Global buffers
// ============================================================

static std::vector<Pixel>
    currentFrame(
        RENDER_WIDTH *
        RENDER_HEIGHT
    );


static std::vector<Pixel>
    previousFrame(
        RENDER_WIDTH *
        RENDER_HEIGHT
    );


static std::vector<Vec3>
    accumulation(
        RENDER_WIDTH *
        RENDER_HEIGHT
    );


static std::vector<uint32_t>
    sampleCount(
        RENDER_WIDTH *
        RENDER_HEIGHT,
        0
    );


static std::vector<Vec3>
    output(
        RENDER_WIDTH *
        RENDER_HEIGHT
    );


// ============================================================
// Global camera/input
// ============================================================

static Camera camera;


static bool firstMouse =
    true;


static double lastMouseX =
    WINDOW_WIDTH * 0.5;


static double lastMouseY =
    WINDOW_HEIGHT * 0.5;


static float mouseSensitivity =
    0.0025f;


static float moveSpeed =
    4.0f;


static bool cameraMoving =
    false;


static uint32_t frameNumber =
    0;


// ============================================================
// Reset accumulation
// ============================================================

static void resetAccumulation()
{
    std::fill(
        accumulation.begin(),
        accumulation.end(),
        Vec3(0.0f)
    );


    std::fill(
        sampleCount.begin(),
        sampleCount.end(),
        0
    );


    std::fill(
        previousFrame.begin(),
        previousFrame.end(),
        Pixel()
    );


    frameNumber =
        0;
}


// ============================================================
// Mouse
// ============================================================

static void mouseCallback(
    GLFWwindow* window,
    double xpos,
    double ypos)
{
    if(firstMouse)
    {
        lastMouseX =
            xpos;

        lastMouseY =
            ypos;

        firstMouse =
            false;

        return;
    }


    double dx =
        xpos -
        lastMouseX;


    double dy =
        ypos -
        lastMouseY;


    lastMouseX =
        xpos;

    lastMouseY =
        ypos;


    if(glfwGetMouseButton(
        window,
        GLFW_MOUSE_BUTTON_RIGHT
    ) != GLFW_PRESS)
    {
        return;
    }


    camera.yaw +=
        float(dx) *
        mouseSensitivity;


    camera.pitch -=
        float(dy) *
        mouseSensitivity;


    camera.pitch =
        std::max(
            -1.45f,
            std::min(
                1.45f,
                camera.pitch
            )
        );


    cameraMoving =
        true;
}


// ============================================================
// Camera update
// ============================================================

static bool updateCamera(
    GLFWwindow* window,
    float dt)
{
    bool moved =
        false;


    Vec3 forward(
        std::cos(camera.pitch) *
        std::sin(camera.yaw),

        std::sin(camera.pitch),

        std::cos(camera.pitch) *
        std::cos(camera.yaw)
    );


    forward =
        normalize(
            forward
        );


    Vec3 right =
        normalize(
            cross(
                forward,
                Vec3(0, 1, 0)
            )
        );


    Vec3 movement(0.0f);


    if(glfwGetKey(
        window,
        GLFW_KEY_W
    ) == GLFW_PRESS)
    {
        movement +=
            forward;
    }


    if(glfwGetKey(
        window,
        GLFW_KEY_S
    ) == GLFW_PRESS)
    {
        movement -=
            forward;
    }


    if(glfwGetKey(
        window,
        GLFW_KEY_D
    ) == GLFW_PRESS)
    {
        movement +=
            right;
    }


    if(glfwGetKey(
        window,
        GLFW_KEY_A
    ) == GLFW_PRESS)
    {
        movement -=
            right;
    }


    if(glfwGetKey(
        window,
        GLFW_KEY_SPACE
    ) == GLFW_PRESS)
    {
        movement.y +=
            1.0f;
    }


    if(glfwGetKey(
        window,
        GLFW_KEY_LEFT_SHIFT
    ) == GLFW_PRESS)
    {
        movement.y -=
            1.0f;
    }


    if(lengthSq(movement) >
       0.000001f)
    {
        movement =
            normalize(
                movement
            );


        camera.position +=
            movement *
            moveSpeed *
            dt;


        moved =
            true;
    }


    return moved;
}


// ============================================================
// Render worker
// ============================================================

static void renderRows(
    int y0,
    int y1,
    uint32_t seed)
{
    const float aspect =
        float(RENDER_WIDTH) /
        float(RENDER_HEIGHT);

    const float cx =
        float(RENDER_WIDTH - 1) * 0.5f;

    const float cy =
        float(RENDER_HEIGHT - 1) * 0.5f;

    const float invHalfW =
        2.0f / float(RENDER_WIDTH);

    const float invHalfH =
        2.0f / float(RENDER_HEIGHT);

    for(int y = y0; y < y1; ++y)
    {
        for(int x = 0; x < RENDER_WIDTH; ++x)
        {
            int index =
                y * RENDER_WIDTH + x;

            float dx =
                (float(x) - cx) * invHalfW;

            float dy =
                (float(y) - cy) * invHalfH;

            float radius =
                std::sqrt(dx * dx + dy * dy);

            bool fovea =
                radius < FOVEA_RADIUS;

            bool periphery =
                radius > MID_RADIUS;

            // After the first frame, the far periphery is only
            // refreshed every other frame. Its accumulated image
            // remains visible between updates, like temporal
            // peripheral vision.
            if(frameNumber > 0 &&
               periphery &&
               (frameNumber & 1u))
            {
                continue;
            }

            int samples =
                fovea ? 2 : 1;

            Vec3 colorSum(0.0f);
            Hit firstHit;
            bool gotHit = false;

            for(int s = 0; s < samples; ++s)
            {
                uint32_t pixelSeed =
                    seed ^
                    uint32_t(x * 1973) ^
                    uint32_t(y * 9277) ^
                    uint32_t(s * 26699);

                RNG rng(pixelSeed);

                Ray ray =
                    camera.makeRay(
                        float(x),
                        float(y),
                        aspect,
                        rng
                    );

                Hit primary;

                Vec3 color =
                    trace(
                        ray,
                        rng,
                        &primary
                    );

                colorSum += color;

                if(!gotHit && primary.hit)
                {
                    firstHit = primary;
                    gotHit = true;
                }
            }

            currentFrame[index].color =
                colorSum / float(samples);

            if(gotHit)
            {
                currentFrame[index].normal =
                    firstHit.normal;

                currentFrame[index].depth =
                    firstHit.t;
            }
            else
            {
                currentFrame[index].normal =
                    Vec3(0, 1, 0);

                currentFrame[index].depth =
                    0.0f;
            }
        }
    }
}


// ============================================================
// Render one frame
// ============================================================

static void renderFrame()
{
    unsigned hardwareThreads =
        std::thread::hardware_concurrency();


    if(hardwareThreads == 0)
        hardwareThreads = 2;


    // Leave at least one logical CPU available
    // to the operating system/UI.
    //
    // On a dual-core CPU this becomes 1 worker.
    // On a quad-core CPU this becomes 3 workers.

    unsigned workerCount =
        std::max(
            1u,
            hardwareThreads > 1
            ? hardwareThreads - 1
            : 1u
        );


    workerCount =
        std::min(
            workerCount,
            4u
        );


    std::vector<std::thread>
        workers;


    int rows =
        RENDER_HEIGHT /
        int(workerCount);


    for(unsigned t = 0;
        t < workerCount;
        ++t)
    {
        int y0 =
            int(t) *
            rows;


        int y1 =
            (t == workerCount - 1)
            ? RENDER_HEIGHT
            : y0 + rows;


        workers.emplace_back(
            renderRows,
            y0,
            y1,
            0x1234567u +
            frameNumber * 7919u +
            t * 991u
        );
    }


    for(auto& worker : workers)
        worker.join();


    // --------------------------------------------------------
    // Progressive accumulation
    // --------------------------------------------------------
    //
    // Instead of doing 4 samples every frame:
    //
    //     frame 1 -> 1 sample
    //     frame 2 -> 1 sample
    //     frame 3 -> 1 sample
    //     ...
    //
    // The result becomes cleaner while sitting still.
    //
    // This is MUCH more useful on a slow CPU.
    // --------------------------------------------------------

    for(size_t i = 0;
        i < accumulation.size();
        ++i)
    {
        uint32_t count =
            sampleCount[i];


        Vec3 sample =
            currentFrame[i].color;


        // Limit history to avoid numerical issues.
        //
        // This is effectively a very long moving
        // average.

        if(count < 4096)
        {
            accumulation[i] =
                (
                    accumulation[i] *
                    float(count)
                    +
                    sample
                )
                /
                float(count + 1);


            sampleCount[i] =
                count + 1;
        }
        else
        {
            // Very old pixels still slowly adapt.

            accumulation[i] =
                accumulation[i] *
                0.999f
                +
                sample *
                0.001f;
        }
    }


    // --------------------------------------------------------
    // Cheap edge-aware reconstruction.
    //
    // Only 4 neighbors rather than a 3x3 kernel.
    // --------------------------------------------------------

    static std::vector<Vec3>
        filtered(
            RENDER_WIDTH *
            RENDER_HEIGHT
        );


    for(int y = 1;
        y < RENDER_HEIGHT - 1;
        ++y)
    {
        for(int x = 1;
            x < RENDER_WIDTH - 1;
            ++x)
        {
            int i =
                y *
                RENDER_WIDTH +
                x;


            Vec3 center =
                accumulation[i];


            Vec3 centerNormal =
                currentFrame[i].normal;


            float centerDepth =
                currentFrame[i].depth;


            Vec3 sum =
                center *
                2.0f;


            float weightSum =
                2.0f;


            const int offsets[4] =
            {
                -1,
                1,
                -RENDER_WIDTH,
                RENDER_WIDTH
            };


            for(int k = 0;
                k < 4;
                ++k)
            {
                int j =
                    i + offsets[k];


                Vec3 neighbor =
                    accumulation[j];


                Vec3 neighborNormal =
                    currentFrame[j].normal;


                float normalWeight =
                    std::max(
                        0.0f,
                        dot(
                            centerNormal,
                            neighborNormal
                        )
                    );


                float depthWeight =
                    1.0f;


                float neighborDepth =
                    currentFrame[j].depth;


                if(centerDepth > 0.0f &&
                   neighborDepth > 0.0f)
                {
                    float dz =
                        std::fabs(
                            centerDepth -
                            neighborDepth
                        );


                    depthWeight =
                        std::exp(
                            -dz * 8.0f
                        );
                }


                float w =
                    normalWeight *
                    normalWeight *
                    depthWeight *
                    0.5f;


                sum +=
                    neighbor *
                    w;


                weightSum +=
                    w;
            }


            filtered[i] =
                sum /
                weightSum;
        }
    }


    // --------------------------------------------------------
    // Copy borders and filtered pixels.
    // --------------------------------------------------------

    for(int y = 1;
        y < RENDER_HEIGHT - 1;
        ++y)
    {
        for(int x = 1;
            x < RENDER_WIDTH - 1;
            ++x)
        {
            int i =
                y *
                RENDER_WIDTH +
                x;


            // Only apply a modest amount of filtering.
            //
            // This prevents the "plastic blur" look.

            float dx =
                (float(x) -
                 float(RENDER_WIDTH - 1) * 0.5f) /
                (float(RENDER_WIDTH) * 0.5f);

            float dy =
                (float(y) -
                 float(RENDER_HEIGHT - 1) * 0.5f) /
                (float(RENDER_HEIGHT) * 0.5f);

            float radius =
                std::sqrt(dx * dx + dy * dy);

            // Keep the fovea crisp; let the periphery receive
            // stronger reconstruction where the eye is less
            // sensitive to individual noisy pixels.
            float filterAmount =
                radius < FOVEA_RADIUS
                ? 0.08f
                : (radius > MID_RADIUS ? 0.42f : 0.25f);

            output[i] =
                accumulation[i] *
                (1.0f - filterAmount)
                +
                filtered[i] *
                filterAmount;
        }
    }


    // Borders.

    for(int x = 0;
        x < RENDER_WIDTH;
        ++x)
    {
        output[x] =
            accumulation[x];


        output[
            (RENDER_HEIGHT - 1) *
            RENDER_WIDTH +
            x
        ] =
            accumulation[
                (RENDER_HEIGHT - 1) *
                RENDER_WIDTH +
                x
            ];
    }


    for(int y = 0;
        y < RENDER_HEIGHT;
        ++y)
    {
        output[
            y *
            RENDER_WIDTH
        ] =
            accumulation[
                y *
                RENDER_WIDTH
            ];


        output[
            y *
            RENDER_WIDTH +
            RENDER_WIDTH - 1
        ] =
            accumulation[
                y *
                RENDER_WIDTH +
                RENDER_WIDTH - 1
            ];
    }


    // --------------------------------------------------------
    // Tone mapping.
    // --------------------------------------------------------

    for(size_t i = 0;
        i < output.size();
        ++i)
    {
        Vec3 c =
            output[i];


        // Exposure.

        c *=
            1.10f;


        // Reinhard.

        c.x =
            c.x /
            (1.0f + c.x);


        c.y =
            c.y /
            (1.0f + c.y);


        c.z =
            c.z /
            (1.0f + c.z);


        // Gamma.

        c.x =
            std::pow(
                clamp01(c.x),
                1.0f / 2.2f
            );


        c.y =
            std::pow(
                clamp01(c.y),
                1.0f / 2.2f
            );


        c.z =
            std::pow(
                clamp01(c.z),
                1.0f / 2.2f
            );


        output[i] =
            c;
    }


    previousFrame =
        currentFrame;


    ++frameNumber;
}


// ============================================================
// PNG
// ============================================================

static bool savePNG(
    const char* filename)
{
    FILE* fp =
        std::fopen(
            filename,
            "wb"
        );


    if(!fp)
        return false;


    png_structp png =
        png_create_write_struct(
            PNG_LIBPNG_VER_STRING,
            nullptr,
            nullptr,
            nullptr
        );


    if(!png)
    {
        fclose(fp);

        return false;
    }


    png_infop info =
        png_create_info_struct(
            png
        );


    if(!info)
    {
        png_destroy_write_struct(
            &png,
            nullptr
        );

        fclose(fp);

        return false;
    }


    if(setjmp(
        png_jmpbuf(png)
    ))
    {
        png_destroy_write_struct(
            &png,
            &info
        );

        fclose(fp);

        return false;
    }


    png_init_io(
        png,
        fp
    );


    png_set_IHDR(
        png,
        info,
        RENDER_WIDTH,
        RENDER_HEIGHT,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );


    png_write_info(
        png,
        info
    );


    std::vector<unsigned char>
        row(
            RENDER_WIDTH * 3
        );


    for(int y = 0;
        y < RENDER_HEIGHT;
        ++y)
    {
        for(int x = 0;
            x < RENDER_WIDTH;
            ++x)
        {
            Vec3 c =
                output[
                    y *
                    RENDER_WIDTH +
                    x
                ];


            row[x * 3 + 0] =
                uint8_t(
                    clamp01(c.x) *
                    255.0f
                );


            row[x * 3 + 1] =
                uint8_t(
                    clamp01(c.y) *
                    255.0f
                );


            row[x * 3 + 2] =
                uint8_t(
                    clamp01(c.z) *
                    255.0f
                );
        }


        png_write_row(
            png,
            row.data()
        );
    }


    png_write_end(
        png,
        nullptr
    );


    png_destroy_write_struct(
        &png,
        &info
    );


    fclose(fp);


    return true;
}


// ============================================================
// OpenGL
// ============================================================

static GLuint texture =
    0;


static void initGL()
{
    glGenTextures(
        1,
        &texture
    );


    glBindTexture(
        GL_TEXTURE_2D,
        texture
    );


    // Linear filtering makes the 480x270
    // image look considerably nicer when
    // enlarged to 960x540.

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_LINEAR
    );


    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_LINEAR
    );


    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_S,
        GL_CLAMP
    );


    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_T,
        GL_CLAMP
    );


    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        RENDER_WIDTH,
        RENDER_HEIGHT,
        0,
        GL_RGB,
        GL_FLOAT,
        nullptr
    );
}


// ============================================================
// Display
// ============================================================

static void display()
{
    glBindTexture(
        GL_TEXTURE_2D,
        texture
    );


    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        RENDER_WIDTH,
        RENDER_HEIGHT,
        GL_RGB,
        GL_FLOAT,
        output.data()
    );


    glClear(
        GL_COLOR_BUFFER_BIT
    );


    glEnable(
        GL_TEXTURE_2D
    );


    glBegin(
        GL_QUADS
    );


    // IMPORTANT:
    //
    // CPU image row 0 is the TOP.
    //
    // OpenGL texture coordinate 0 is the
    // BOTTOM.
    //
    // Therefore these Y coordinates are
    // deliberately reversed compared with
    // the old version.


    glTexCoord2f(
        0.0f,
        1.0f
    );

    glVertex2f(
        -1.0f,
        -1.0f
    );


    glTexCoord2f(
        1.0f,
        1.0f
    );

    glVertex2f(
        1.0f,
        -1.0f
    );


    glTexCoord2f(
        1.0f,
        0.0f
    );

    glVertex2f(
        1.0f,
        1.0f
    );


    glTexCoord2f(
        0.0f,
        0.0f
    );

    glVertex2f(
        -1.0f,
        1.0f
    );


    glEnd();


    glDisable(
        GL_TEXTURE_2D
    );
}


// ============================================================
// Scene
// ============================================================

static void setupScene()
{
    // --------------------------------------------------------
    // Red diffuse sphere
    // --------------------------------------------------------

    Material red;

    red.albedo =
        Vec3(
            0.82f,
            0.08f,
            0.05f
        );

    red.roughness =
        0.5f;

    red.transmission =
        0.0f;

    red.ior =
        1.0f;

    red.emission =
        0.0f;


    // --------------------------------------------------------
    // Blue diffuse sphere
    // --------------------------------------------------------

    Material blue;

    blue.albedo =
        Vec3(
            0.06f,
            0.20f,
            0.85f
        );

    blue.roughness =
        0.3f;

    blue.transmission =
        0.0f;

    blue.ior =
        1.0f;

    blue.emission =
        0.0f;


    // --------------------------------------------------------
    // Green diffuse sphere
    // --------------------------------------------------------

    Material green;

    green.albedo =
        Vec3(
            0.08f,
            0.65f,
            0.12f
        );

    green.roughness =
        0.7f;

    green.transmission =
        0.0f;

    green.ior =
        1.0f;

    green.emission =
        0.0f;


    // --------------------------------------------------------
    // Glass
    // --------------------------------------------------------

    Material glass;

    glass.albedo =
        Vec3(
            0.96f,
            0.98f,
            1.0f
        );

    glass.roughness =
        0.0f;

    glass.transmission =
        1.0f;

    glass.ior =
        1.50f;

    glass.emission =
        0.0f;


    // --------------------------------------------------------
    // Floor
    // --------------------------------------------------------

    Material floor;

    floor.albedo =
        Vec3(
            0.72f,
            0.72f,
            0.72f
        );

    floor.roughness =
        0.8f;

    floor.transmission =
        0.0f;

    floor.ior =
        1.0f;

    floor.emission =
        0.0f;


    // --------------------------------------------------------
    // Scene objects
    // --------------------------------------------------------

    spheres.push_back({
        Vec3(
            -1.45f,
            1.0f,
            0.0f
        ),
        1.0f,
        red
    });


    spheres.push_back({
        Vec3(
            1.15f,
            1.0f,
            -0.4f
        ),
        1.0f,
        blue
    });


    spheres.push_back({
        Vec3(
            0.0f,
            0.72f,
            1.55f
        ),
        0.72f,
        green
    });


    // CLEAR GLASS BALL.
    //
    // Placed toward the front so it is
    // obvious that it is transparent.

    spheres.push_back({
        Vec3(
            0.0f,
            1.15f,
            -1.65f
        ),
        1.15f,
        glass
    });


    // Floor.

    floorPlane.point =
        Vec3(
            0.0f,
            0.0f,
            0.0f
        );


    floorPlane.normal =
        Vec3(
            0.0f,
            1.0f,
            0.0f
        );


    floorPlane.material =
        floor;


    // --------------------------------------------------------
    // Camera
    // --------------------------------------------------------

    camera.position =
        Vec3(
            0.0f,
            2.5f,
            7.8f
        );


    // Looking toward the scene.

    camera.yaw =
        PI;


    camera.pitch =
        -0.16f;


    camera.fov =
        48.0f;
}


// ============================================================
// Main
// ============================================================

int main()
{
    // --------------------------------------------------------
    // GLFW
    // --------------------------------------------------------

    if(!glfwInit())
    {
        std::fprintf(
            stderr,
            "GLFW initialization failed.\n"
        );

        return 1;
    }


    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MAJOR,
        2
    );


    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MINOR,
        1
    );


    GLFWwindow* window =
        glfwCreateWindow(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            "Optimized CPU Perceptual Ray Tracer",
            nullptr,
            nullptr
        );


    if(!window)
    {
        std::fprintf(
            stderr,
            "Could not create GLFW window.\n"
        );

        glfwTerminate();

        return 1;
    }


    glfwMakeContextCurrent(
        window
    );


    // --------------------------------------------------------
    // GLEW
    // --------------------------------------------------------

    if(glewInit() != GLEW_OK)
    {
        std::fprintf(
            stderr,
            "GLEW initialization failed.\n"
        );

        glfwDestroyWindow(
            window
        );

        glfwTerminate();

        return 1;
    }


    // --------------------------------------------------------
    // Input
    // --------------------------------------------------------

    glfwSetCursorPosCallback(
        window,
        mouseCallback
    );


    // --------------------------------------------------------
    // Setup
    // --------------------------------------------------------

    setupScene();

    initGL();

    resetAccumulation();


    // --------------------------------------------------------
    // Console
    // --------------------------------------------------------

    std::printf(
        "\n"
        "==============================================\n"
        " OPTIMIZED CPU PERCEPTUAL RAY TRACER\n"
        "==============================================\n"
        "\n"
        "CPU resolution: %dx%d\n"
        "Window:         %dx%d\n"
        "Target FPS:     %.0f\n"
        "\n"
        "W A S D         Move\n"
        "SPACE           Up\n"
        "LEFT SHIFT      Down\n"
        "RIGHT MOUSE     Look\n"
        "R               Reset accumulation\n"
        "P               Save screenshot.png\n"
        "ESC             Quit\n"
        "\n"
        "Glass sphere enabled.\n"
        "Progressive accumulation enabled.\n"
        "Foveated center sampling enabled.\n"
        "Stable transmissive glass cheat enabled.\n"
        "\n"
        "==============================================\n\n",
        RENDER_WIDTH,
        RENDER_HEIGHT,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        TARGET_FPS
    );


    // --------------------------------------------------------
    // Main loop
    // --------------------------------------------------------

    double previousTime =
        glfwGetTime();


    bool previousR =
        false;


    bool previousP =
        false;


    while(!glfwWindowShouldClose(
        window))
    {
        double frameStart =
            glfwGetTime();


        // ----------------------------------------------------
        // Delta time
        // ----------------------------------------------------

        float dt =
            float(
                frameStart -
                previousTime
            );


        previousTime =
            frameStart;


        dt =
            std::min(
                dt,
                0.1f
            );


        // ----------------------------------------------------
        // Events
        // ----------------------------------------------------

        glfwPollEvents();


        // ----------------------------------------------------
        // Camera
        // ----------------------------------------------------

        bool moved =
            updateCamera(
                window,
                dt
            );


        if(moved)
            cameraMoving =
                true;


        // ----------------------------------------------------
        // Reset key
        // ----------------------------------------------------

        bool rDown =
            glfwGetKey(
                window,
                GLFW_KEY_R
            ) == GLFW_PRESS;


        if(rDown && !previousR)
        {
            resetAccumulation();

            std::printf(
                "Accumulation reset.\n"
            );
        }


        previousR =
            rDown;


        // ----------------------------------------------------
        // Screenshot
        // ----------------------------------------------------

        bool pDown =
            glfwGetKey(
                window,
                GLFW_KEY_P
            ) == GLFW_PRESS;


        if(pDown && !previousP)
        {
            if(savePNG(
                "screenshot.png"
            ))
            {
                std::printf(
                    "Saved screenshot.png\n"
                );
            }
            else
            {
                std::printf(
                    "Could not save screenshot.\n"
                );
            }
        }


        previousP =
            pDown;


        // ----------------------------------------------------
        // Escape
        // ----------------------------------------------------

        if(glfwGetKey(
            window,
            GLFW_KEY_ESCAPE
        ) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(
                window,
                GLFW_TRUE
            );
        }


        // ----------------------------------------------------
        // Camera movement:
        // discard old samples so they cannot ghost into the new
        // view. The glass path itself is deterministic, so the
        // moving image remains much more stable than before.
        // ----------------------------------------------------

        if(cameraMoving)
        {
            resetAccumulation();
            cameraMoving = false;
        }


        // ----------------------------------------------------
        // Render
        // ----------------------------------------------------

        renderFrame();


        // ----------------------------------------------------
        // Display
        // ----------------------------------------------------

        display();


        glfwSwapBuffers(
            window
        );


        // ----------------------------------------------------
        // 25 FPS limiter
        // ----------------------------------------------------

        double elapsed =
            glfwGetTime() -
            frameStart;


        double remaining =
            TARGET_FRAME_TIME -
            elapsed;


        if(remaining > 0.0)
        {
            if(remaining > 0.002)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(
                        int(
                            (
                                remaining -
                                0.001
                            ) *
                            1000.0
                        )
                    )
                );
            }


            while(
                glfwGetTime() -
                frameStart <
                TARGET_FRAME_TIME
            )
            {
                std::this_thread::yield();
            }
        }
    }


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    glDeleteTextures(
        1,
        &texture
    );


    glfwDestroyWindow(
        window
    );


    glfwTerminate();


    return 0;
}
