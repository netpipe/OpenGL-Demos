// ============================================================
// GPU PERCEPTUAL RAY TRACER + GPU BVH
// ============================================================
//
// OpenGL 4.3+ compute-shader ray tracer.
//
// CPU:
//   - GLFW window/input
//   - GLEW OpenGL loading
//   - BVH construction/upload
//   - PNG screenshot
//
// GPU:
//   - Ray generation
//   - BVH traversal
//   - Shading
//   - Clear transmissive glass
//   - Foveated sampling
//   - Temporal accumulation
//   - Tone mapping
//
// The BVH is built on the CPU once, then traversed on the GPU.
// This is a good fit for larger scenes: adding hundreds or
// thousands of bounded objects makes BVH traversal much cheaper
// than testing every object for every ray.
//
// Current demo scene remains sphere-heavy so it is easy to extend.
// The infinite floor is tested separately because it is not a
// bounded primitive and therefore does not belong in this BVH.
//
// REQUIREMENTS
//   OpenGL 4.3+
//   GLFW
//   GLEW
//   libpng
//
// Linux build:
//   g++ raytracer_gpu_bvh.cpp -std=c++17 -O2 \
//       -lglfw -lGLEW -lGL -lpng -o raytracer_gpu_bvh
//
// CONTROLS
//   W A S D       Move
//   SPACE         Up
//   LEFT SHIFT    Down
//   RMB           Look
//   R             Reset accumulation
//   P             Save screenshot.png
//   ESC           Quit
//
// PERFORMANCE KNOBS near the top:
//   RENDER_WIDTH / RENDER_HEIGHT
//   TARGET_FPS
//   FOVEA_RADIUS
//   MID_RADIUS
//   FOVEA_SPP
//   MAX_BOUNCES
// ============================================================

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <png.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <thread>
#include <vector>

// ============================================================
// Settings
// ============================================================

static constexpr int WINDOW_WIDTH  = 960;
static constexpr int WINDOW_HEIGHT = 540;

static constexpr int RENDER_WIDTH  = 480;
static constexpr int RENDER_HEIGHT = 270;

static constexpr double TARGET_FPS = 30.0;
static constexpr double TARGET_FRAME_TIME = 1.0 / TARGET_FPS;

// Normalized distance from screen center.
static constexpr float FOVEA_RADIUS = 0.24f;
static constexpr float MID_RADIUS   = 0.62f;

// Samples per pixel in the fovea. GPU makes this affordable.
static constexpr int FOVEA_SPP = 2;

// Maximum path depth.
static constexpr int MAX_BOUNCES = 4;

static constexpr float PI = 3.14159265358979323846f;
static constexpr float EPSILON = 0.001f;

// ============================================================
// Tiny CPU math layer
// ============================================================

struct Vec3
{
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float v) : x(v), y(v), z(v) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    Vec3 operator+(const Vec3& b) const { return Vec3(x+b.x,y+b.y,z+b.z); }
    Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x,y-b.y,z-b.z); }
    Vec3 operator-() const { return Vec3(-x,-y,-z); }
    Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s,y/s,z/s); }
Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
Vec3& operator-=(const Vec3& b) { x-=b.x; y-=b.y; z-=b.z; return *this; }
Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
Vec3& operator/=(float s) { x/=s; y/=s; z/=s; return *this; }
};

static Vec3 minv(const Vec3& a, const Vec3& b)
{
    return Vec3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z));
}

static Vec3 maxv(const Vec3& a, const Vec3& b)
{
    return Vec3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z));
}

static float dotv(const Vec3& a, const Vec3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static float lengthv(const Vec3& a)
{
    return std::sqrt(dotv(a,a));
}

// ============================================================
// Scene data
// ============================================================

struct Material
{
    Vec3 albedo;
    float roughness;
    float transmission;
    float ior;
    float emission;
};

struct Sphere
{
    Vec3 center;
    float radius;
    Material material;
};

struct Plane
{
    Vec3 point;
    Vec3 normal;
    Material material;
};

static std::vector<Sphere> spheres;
static Plane floorPlane;

// ============================================================
// GPU std430-compatible records
// ============================================================
// vec4 alignment is deliberately used so the C++ and GLSL
// layouts remain simple and portable.

struct GPUSphere
{
    float centerRadius[4]; // xyz=center, w=radius
    float albedo[4];       // xyz=albedo, w=roughness
    float params[4];       // transmission, ior, emission, unused
};

struct GPUBVHNode
{
    float bmin[4];         // xyz=min, w=unused
    float bmax[4];         // xyz=max, w=unused
    int32_t leftFirst;
    int32_t count;
    int32_t pad0;
    int32_t pad1;
};

static_assert(sizeof(GPUSphere) == 48, "GPUSphere layout unexpected");
static_assert(sizeof(GPUBVHNode) == 48, "GPUBVHNode layout unexpected");

// ============================================================
// BVH
// ============================================================

struct BuildNode
{
    Vec3 bmin;
    Vec3 bmax;
    int leftFirst = 0;
    int count = 0;
    int rightChild = -1;
};

static std::vector<BuildNode> bvh;
static std::vector<int> bvhIndices;

static Vec3 sphereMin(const Sphere& s)
{
    return s.center - Vec3(s.radius);
}

static Vec3 sphereMax(const Sphere& s)
{
    return s.center + Vec3(s.radius);
}

static Vec3 centroid(const Sphere& s)
{
    return s.center;
}

static int longestAxis(const Vec3& e)
{
    if(e.x > e.y && e.x > e.z) return 0;
    if(e.y > e.z) return 1;
    return 2;
}

static float axisValue(const Vec3& v, int axis)
{
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

static void computeNodeBounds(int nodeIndex, int first, int count)
{
    Vec3 mn(std::numeric_limits<float>::max());
    Vec3 mx(-std::numeric_limits<float>::max());

    for(int i=0;i<count;i++)
    {
        const Sphere& s = spheres[bvhIndices[first+i]];
        mn = minv(mn, sphereMin(s));
        mx = maxv(mx, sphereMax(s));
    }

    bvh[nodeIndex].bmin = mn;
    bvh[nodeIndex].bmax = mx;
    bvh[nodeIndex].leftFirst = first;
    bvh[nodeIndex].count = count;
}

static int buildBVHRecursive(int first, int count)
{
    const int nodeIndex = static_cast<int>(bvh.size());
    bvh.push_back(BuildNode{});
    computeNodeBounds(nodeIndex, first, count);

    // Four objects per leaf keeps traversal cheap for small scenes
    // while still producing useful depth for large scenes.
    constexpr int LEAF_SIZE = 4;
    if(count <= LEAF_SIZE)
        return nodeIndex;

    Vec3 cmin(std::numeric_limits<float>::max());
    Vec3 cmax(-std::numeric_limits<float>::max());

    for(int i=0;i<count;i++)
    {
        Vec3 c = centroid(spheres[bvhIndices[first+i]]);
        cmin = minv(cmin,c);
        cmax = maxv(cmax,c);
    }

    int axis = longestAxis(cmax-cmin);
    int mid = first + count/2;

    std::nth_element(
        bvhIndices.begin()+first,
        bvhIndices.begin()+mid,
        bvhIndices.begin()+first+count,
        [axis](int a, int b)
        {
            return axisValue(centroid(spheres[a]),axis) <
                   axisValue(centroid(spheres[b]),axis);
        }
    );

    int left = buildBVHRecursive(first, mid-first);
    int right = buildBVHRecursive(mid, first+count-mid);

    // Internal node: leftFirst stores left child index, count=0.
    bvh[nodeIndex].leftFirst = left;
    bvh[nodeIndex].count = 0;
    // Store the explicit right child; the left subtree can contain many nodes.
    // This avoids assuming right == left + 1.
    bvh[nodeIndex].rightChild = right;

    return nodeIndex;
}

static void buildBVH()
{
    bvh.clear();
    bvhIndices.resize(spheres.size());
    for(size_t i=0;i<spheres.size();i++)
        bvhIndices[i] = static_cast<int>(i);

    if(!spheres.empty())
        buildBVHRecursive(0, static_cast<int>(spheres.size()));

    std::printf("BVH: %zu spheres, %zu nodes\n", spheres.size(), bvh.size());
}

// ============================================================
// Camera
// ============================================================

struct Camera
{
    Vec3 position;
    float yaw = PI;
    float pitch = -0.16f;
    float fov = 48.0f;
};

static Camera camera;
static Camera lastRenderedCamera;

static bool firstMouse = true;
static double lastMouseX = WINDOW_WIDTH * 0.5;
static double lastMouseY = WINDOW_HEIGHT * 0.5;
static float mouseSensitivity = 0.0025f;
static float moveSpeed = 4.0f;
static bool cameraMoving = false;

// ============================================================
// OpenGL handles
// ============================================================

static GLuint computeProgram = 0;
static GLuint displayProgram = 0;
static GLuint accumulationTexture = 0;
static GLuint displayTexture = 0;
static GLuint sphereSSBO = 0;
static GLuint bvhSSBO = 0;
static GLuint vao = 0;
static GLuint displaySampler = 0;

static int frameIndex = 0;
static bool accumulationReset = true;

// ============================================================
// GLSL compute shader
// ============================================================

static const char* computeShaderSource = R"GLSL(
#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32f, binding = 0) uniform image2D accumImage;
layout(rgba8,   binding = 1) uniform image2D outputImage;

struct Sphere
{
    vec4 centerRadius;
    vec4 albedoRoughness;
    vec4 params;
};

struct BVHNode
{
    vec4 bmin;
    vec4 bmax;
    int leftFirst;
    int count;
    int pad0;
    int pad1;
};

layout(std430, binding = 2) readonly buffer SphereBuffer
{
    Sphere spheres[];
};

layout(std430, binding = 3) readonly buffer BVHBuffer
{
    BVHNode nodes[];
};

uniform vec3 uCameraPosition;
uniform vec3 uForward;
uniform vec3 uRight;
uniform vec3 uUp;
uniform float uAspect;
uniform float uTanFov;
uniform int uFrame;
uniform int uReset;
uniform int uRenderWidth;
uniform int uRenderHeight;
uniform float uFoveaRadius;
uniform float uMidRadius;
uniform int uFoveaSPP;
uniform int uMaxBounces;
uniform vec3 uLightPosition;
uniform vec3 uLightColor;

const float PI = 3.14159265358979323846;
const float EPS = 0.001;
const float INF = 1e30;

struct Hit
{
    float t;
    vec3 position;
    vec3 normal;
    vec3 albedo;
    float roughness;
    float transmission;
    float ior;
    float emission;
    bool hit;
};

uint hashU(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand(inout uint state)
{
    state = hashU(state + 0x9e3779b9u);
    return float(state & 0x00ffffffu) / 16777216.0;
}

vec3 sky(vec3 d)
{
    float t = 0.5 * (d.y + 1.0);
    vec3 horizon = vec3(0.72, 0.80, 0.95);
    vec3 zenith = vec3(0.16, 0.25, 0.42);
    return mix(horizon, zenith, clamp(t,0.0,1.0));
}

bool hitAABB(vec3 ro, vec3 rd, vec3 mn, vec3 mx, float maxT)
{
    vec3 inv = 1.0 / rd;
    vec3 t0 = (mn-ro)*inv;
    vec3 t1 = (mx-ro)*inv;
    vec3 lo = min(t0,t1);
    vec3 hi = max(t0,t1);
    float enter = max(max(lo.x,lo.y),lo.z);
    float exit  = min(min(hi.x,hi.y),hi.z);
    return exit >= max(enter,0.0) && enter < maxT;
}

bool hitSphere(vec3 ro, vec3 rd, Sphere s, inout Hit h)
{
    vec3 c = s.centerRadius.xyz;
    float r = s.centerRadius.w;
    vec3 oc = ro-c;
    float b = dot(oc,rd);
    float c2 = dot(oc,oc)-r*r;
    float disc = b*b-c2;
    if(disc < 0.0) return false;

    float root = sqrt(disc);
    float t = -b-root;
    if(t < EPS) t = -b+root;
    if(t < EPS || t >= h.t) return false;

    h.t = t;
    h.position = ro+rd*t;
    h.normal = normalize(h.position-c);
    h.albedo = s.albedoRoughness.xyz;
    h.roughness = s.albedoRoughness.w;
    h.transmission = s.params.x;
    h.ior = s.params.y;
    h.emission = s.params.z;
    h.hit = true;
    return true;
}

bool intersectScene(vec3 ro, vec3 rd, out Hit outHit)
{
    Hit h;
    h.t = INF;
    h.hit = false;

    // Infinite floor outside the BVH.
    float denom = dot(rd, vec3(0,1,0));
    if(abs(denom) > 1e-6)
    {
        float t = -ro.y / denom;
        if(t > EPS && t < h.t)
        {
            h.t = t;
            h.position = ro+rd*t;
            h.normal = vec3(0,1,0);
            h.albedo = vec3(0.34,0.36,0.39);
            h.roughness = 0.85;
            h.transmission = 0.0;
            h.ior = 1.0;
            h.emission = 0.0;
            h.hit = true;
        }
    }

    if(nodes.length() == 0)
    {
        outHit = h;
        return h.hit;
    }

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while(sp > 0)
    {
        int nodeIndex = stack[--sp];
        BVHNode n = nodes[nodeIndex];

        if(!hitAABB(ro,rd,n.bmin.xyz,n.bmax.xyz,h.t))
            continue;

        if(n.count > 0)
        {
            for(int i=0;i<n.count;i++)
            {
                int sphereIndex = n.leftFirst+i;
                hitSphere(ro,rd,spheres[sphereIndex],h);
            }
        }
        else
        {
            int left = n.leftFirst;
            int right = n.pad0;
            if(sp+2 < 64)
            {
                // No costly child sorting here; near/far ordering is
                // intentionally simple for predictable GPU work.
                stack[sp++] = right;
                stack[sp++] = left;
            }
        }
    }

    outHit = h;
    return h.hit;
}

vec3 cosineHemisphere(vec3 n, inout uint state)
{
    float u1 = rand(state);
    float u2 = rand(state);
    float r = sqrt(u1);
    float a = 2.0*PI*u2;

    vec3 t = normalize(abs(n.x) > 0.1 ? cross(vec3(0,1,0),n)
                                      : cross(vec3(1,0,0),n));
    vec3 b = cross(n,t);
    return normalize(t*(r*cos(a)) + b*(r*sin(a)) + n*sqrt(max(0.0,1.0-u1)));
}

vec3 reflectDir(vec3 d, vec3 n)
{
    return d - 2.0*dot(d,n)*n;
}

bool refractDir(vec3 d, vec3 n, float eta, out vec3 result)
{
    float cosi = clamp(-dot(d,n),-1.0,1.0);
    float etai = 1.0;
    float etat = eta;
    vec3 nn = n;

    if(cosi < 0.0)
    {
        cosi = -cosi;
        float tmp = etai; etai = etat; etat = tmp;
        nn = -nn;
    }

    float ratio = etai/etat;
    float k = 1.0-ratio*ratio*(1.0-cosi*cosi);
    if(k < 0.0) return false;

    result = normalize(d*ratio + nn*(ratio*cosi-sqrt(k)));
    return true;
}

float fresnelSchlick(vec3 d, vec3 n, float ior)
{
    float cosi = clamp(-dot(d,n),0.0,1.0);
    float r0 = (1.0-ior)/(1.0+ior);
    r0 *= r0;
    float q = 1.0-cosi;
    return r0 + (1.0-r0)*q*q*q*q*q;
}

vec3 traceReflection(vec3 ro, vec3 rd)
{
    Hit h;
    if(!intersectScene(ro,rd,h))
        return sky(rd);

    vec3 base = h.albedo;
    float lightDist = length(uLightPosition-h.position);
    vec3 toLight = normalize(uLightPosition-h.position);
    Hit shadowHit;
    bool blocked = intersectScene(h.position+h.normal*EPS,toLight,shadowHit) && shadowHit.t < lightDist;
    float ndl = max(dot(h.normal,toLight),0.0);
    vec3 direct = blocked ? vec3(0.0) : base*uLightColor*ndl/(1.0+0.08*lightDist*lightDist);
    return direct + base*0.08 + base*h.emission;
}

vec3 tracePath(vec3 ro, vec3 rd, inout uint state)
{
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);

    for(int bounce=0; bounce<16; ++bounce)
    {
        if(bounce >= uMaxBounces) break;

        Hit h;
        if(!intersectScene(ro,rd,h))
        {
            radiance += throughput*sky(rd);
            break;
        }

        radiance += throughput*h.albedo*h.emission;

        if(h.transmission > 0.5)
        {
            // Stable glass cheat: evaluate reflection and refraction
            // deterministically instead of Russian roulette. This keeps
            // the orb visibly clear and prevents black flashes while moving.
            float F = fresnelSchlick(rd,h.normal,h.ior);
            vec3 reflDir = normalize(reflectDir(rd,h.normal));
            vec3 reflOrigin = h.position+h.normal*EPS;
            vec3 refl = traceReflection(reflOrigin,reflDir);

            vec3 refrDir;
            vec3 refr = sky(rd);
            if(refractDir(rd,h.normal,h.ior,refrDir))
            {
                Hit inside;
                vec3 refrOrigin = h.position-refrDir*EPS;
                if(intersectScene(refrOrigin,refrDir,inside))
                {
                    // One more refraction leg is enough to make the
                    // sphere read as transparent without exploding cost.
                    vec3 exitDir;
                    if(inside.transmission > 0.5 &&
                       refractDir(refrDir,inside.normal,inside.ior,exitDir))
                    {
                        Hit outside;
                        if(intersectScene(inside.position+exitDir*EPS,exitDir,outside))
                            refr = outside.albedo*0.18 + sky(exitDir)*0.82;
                        else
                            refr = sky(exitDir);
                    }
                    else
                    {
                        refr = inside.albedo*0.18 + sky(refrDir)*0.82;
                    }
                }
                else
                {
                    refr = sky(refrDir);
                }
            }

            radiance += throughput*(refl*F + refr*(1.0-F))*h.albedo;
            break;
        }

        vec3 toLight = uLightPosition-h.position;
        float lightDist = length(toLight);
        toLight /= max(lightDist,0.0001);

        Hit shadow;
        bool blocked = intersectScene(h.position+h.normal*EPS,toLight,shadow) && shadow.t < lightDist;
        float ndl = max(dot(h.normal,toLight),0.0);
        vec3 direct = blocked ? vec3(0.0)
                              : h.albedo*uLightColor*ndl/(1.0+0.08*lightDist*lightDist);

        radiance += throughput*(direct + h.albedo*0.025);

        float specWeight = 0.12*(1.0-h.roughness);
        if(rand(state) < specWeight)
        {
            rd = normalize(reflectDir(rd,h.normal));
            ro = h.position+h.normal*EPS;
            throughput *= 0.55;
        }
        else
        {
            rd = cosineHemisphere(h.normal,state);
            ro = h.position+h.normal*EPS;
            throughput *= h.albedo;
        }

        if(bounce >= 2)
        {
            float survive = max(throughput.x,max(throughput.y,throughput.z));
            survive = clamp(survive,0.05,0.95);
            if(rand(state) > survive) break;
            throughput /= survive;
        }
    }

    return radiance;
}

vec3 toneMap(vec3 c)
{
    c *= 1.15;
    c = c/(1.0+c);
    return pow(max(c,vec3(0.0)),vec3(1.0/2.2));
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if(p.x >= uRenderWidth || p.y >= uRenderHeight) return;

    vec2 uv = (vec2(p)+0.5)/vec2(uRenderWidth,uRenderHeight);
    vec2 centered = uv*2.0-1.0;
    centered.x *= uRenderHeight/float(uRenderWidth);
    float radius = length(centered);

    int spp = 1;
    if(radius < uFoveaRadius) spp = uFoveaSPP;
    else if(radius > uMidRadius && (uFrame & 1) == 1) spp = 0;

    vec4 old = imageLoad(accumImage,p);
    vec3 oldSum = old.rgb;
    float oldCount = old.a;

    if(uReset != 0)
    {
        oldSum = vec3(0.0);
        oldCount = 0.0;
    }

    vec3 sum = vec3(0.0);

    for(int s=0;s<spp;s++)
    {
        uint state = uint(p.x+1)*1973u ^ uint(p.y+1)*9277u ^ uint(uFrame+1)*26699u ^ uint(s+1)*31847u;
        vec2 jitter = vec2(rand(state)-0.5,rand(state)-0.5);
        vec2 ndc = ((vec2(p)+0.5+jitter)/vec2(uRenderWidth,uRenderHeight))*2.0-1.0;
        ndc.x *= uAspect;
        ndc.y *= -1.0;

        vec3 rd = normalize(uForward + uRight*(ndc.x*uTanFov) + uUp*(ndc.y*uTanFov));
        sum += tracePath(uCameraPosition,rd,state);
    }

    if(spp > 0)
    {
        oldSum += sum;
        oldCount += float(spp);
    }

    imageStore(accumImage,p,vec4(oldSum,oldCount));

    vec3 avg = oldCount > 0.0 ? oldSum/oldCount : sky(uForward);
    imageStore(outputImage,p,vec4(toneMap(avg),1.0));
}
)GLSL";

// ============================================================
// Display shaders
// ============================================================

static const char* displayVertexShader = R"GLSL(
#version 330 core
out vec2 uv;
void main()
{
    const vec2 pos[3] = vec2[](
        vec2(-1.0,-1.0),
        vec2( 3.0,-1.0),
        vec2(-1.0, 3.0)
    );
    vec2 p = pos[gl_VertexID];
    uv = p*0.5+0.5;
    gl_Position = vec4(p,0.0,1.0);
}
)GLSL";

static const char* displayFragmentShader = R"GLSL(
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D uImage;
void main()
{
    // Flip Y because image coordinates start at the top in the
    // compute shader while OpenGL texture coordinates start at bottom.
    fragColor = texture(uImage,vec2(uv.x,1.0-uv.y));
}
)GLSL";

// ============================================================
// Shader helpers
// ============================================================

static GLuint compileShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader,1,&source,nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if(!ok)
    {
        GLint len = 0;
        glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&len);
        std::vector<char> log(static_cast<size_t>(std::max(1,len)));
        glGetShaderInfoLog(shader,len,nullptr,log.data());
        std::fprintf(stderr,"Shader compile error:\n%s\n",log.data());
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint linkProgram(std::initializer_list<GLuint> shaders)
{
    GLuint program = glCreateProgram();
    for(GLuint s : shaders) glAttachShader(program,s);
    glLinkProgram(program);

    GLint ok = GL_FALSE;
    glGetProgramiv(program,GL_LINK_STATUS,&ok);
    if(!ok)
    {
        GLint len = 0;
        glGetProgramiv(program,GL_INFO_LOG_LENGTH,&len);
        std::vector<char> log(static_cast<size_t>(std::max(1,len)));
        glGetProgramInfoLog(program,len,nullptr,log.data());
        std::fprintf(stderr,"Program link error:\n%s\n",log.data());
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

static bool initShaders()
{
    GLuint cs = compileShader(GL_COMPUTE_SHADER,computeShaderSource);
    if(!cs) return false;
    computeProgram = linkProgram({cs});
    glDeleteShader(cs);
    if(!computeProgram) return false;

    GLuint vs = compileShader(GL_VERTEX_SHADER,displayVertexShader);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER,displayFragmentShader);
    if(!vs || !fs)
    {
        if(vs) glDeleteShader(vs);
        if(fs) glDeleteShader(fs);
        return false;
    }
    displayProgram = linkProgram({vs,fs});
    glDeleteShader(vs);
    glDeleteShader(fs);
    return displayProgram != 0;
}

// ============================================================
// GPU buffers/textures
// ============================================================

static void uploadScene()
{
    std::vector<GPUSphere> gpuSpheres;
    gpuSpheres.reserve(spheres.size());

    for(const Sphere& s : spheres)
    {
        GPUSphere g{};
        g.centerRadius[0]=s.center.x;
        g.centerRadius[1]=s.center.y;
        g.centerRadius[2]=s.center.z;
        g.centerRadius[3]=s.radius;

        g.albedo[0]=s.material.albedo.x;
        g.albedo[1]=s.material.albedo.y;
        g.albedo[2]=s.material.albedo.z;
        g.albedo[3]=s.material.roughness;

        g.params[0]=s.material.transmission;
        g.params[1]=s.material.ior;
        g.params[2]=s.material.emission;
        g.params[3]=0.0f;
        gpuSpheres.push_back(g);
    }

    std::vector<GPUBVHNode> gpuNodes;
    gpuNodes.reserve(bvh.size());

    // Leaf indices must refer to the compacted sphere array, not the
    // original order. Reorder spheres into BVH index order.
    std::vector<GPUSphere> orderedSpheres(gpuSpheres.size());
    for(size_t i=0;i<bvhIndices.size();i++)
        orderedSpheres[i] = gpuSpheres[bvhIndices[i]];

    // Convert leaf leftFirst from bvhIndices positions to ordered sphere positions.
    // Internal child indices already point directly into the node array.
    for(const BuildNode& n : bvh)
    {
        GPUBVHNode g{};
        g.bmin[0]=n.bmin.x; g.bmin[1]=n.bmin.y; g.bmin[2]=n.bmin.z;
        g.bmax[0]=n.bmax.x; g.bmax[1]=n.bmax.y; g.bmax[2]=n.bmax.z;
        g.leftFirst=n.leftFirst;
        g.count=n.count;
        g.pad0=n.rightChild;
        gpuNodes.push_back(g);
    }

    glGenBuffers(1,&sphereSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,sphereSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 static_cast<GLsizeiptr>(orderedSpheres.size()*sizeof(GPUSphere)),
                 orderedSpheres.empty()?nullptr:orderedSpheres.data(),GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,sphereSSBO);

    glGenBuffers(1,&bvhSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,bvhSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 static_cast<GLsizeiptr>(gpuNodes.size()*sizeof(GPUBVHNode)),
                 gpuNodes.empty()?nullptr:gpuNodes.data(),GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,bvhSSBO);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,0);
}

static void initTextures()
{
    glGenTextures(1,&accumulationTexture);
    glBindTexture(GL_TEXTURE_2D,accumulationTexture);
    glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32F,RENDER_WIDTH,RENDER_HEIGHT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D,0);

    glGenTextures(1,&displayTexture);
    glBindTexture(GL_TEXTURE_2D,displayTexture);
    glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,RENDER_WIDTH,RENDER_HEIGHT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D,0);
}

static void clearAccumulation()
{
    // The compute shader handles the actual clear when uReset=1.
    // This keeps the program compatible with OpenGL 4.3, where
    // compute shaders are core but glClearTexImage is not required.
    accumulationReset = true;
    frameIndex = 0;
}

// ============================================================
// Scene setup
// ============================================================

static Material makeMaterial(Vec3 albedo,float roughness,float transmission,float ior,float emission=0.0f)
{
    Material m{};
    m.albedo=albedo;
    m.roughness=roughness;
    m.transmission=transmission;
    m.ior=ior;
    m.emission=emission;
    return m;
}

static void setupScene()
{
    spheres.clear();

    spheres.push_back({Vec3(-1.45f,1.0f,0.0f),1.0f,
                       makeMaterial(Vec3(0.85f,0.08f,0.07f),0.72f,0.0f,1.0f)});

    spheres.push_back({Vec3(1.15f,1.0f,-0.4f),1.0f,
                       makeMaterial(Vec3(0.08f,0.20f,0.85f),0.68f,0.0f,1.0f)});

    spheres.push_back({Vec3(0.0f,0.72f,1.55f),0.72f,
                       makeMaterial(Vec3(0.10f,0.78f,0.18f),0.64f,0.0f,1.0f)});

    // Crystal-clear glass orb.
    spheres.push_back({Vec3(0.0f,1.15f,-1.65f),1.15f,
                       makeMaterial(Vec3(0.96f,0.98f,1.0f),0.0f,1.0f,1.5f)});

    floorPlane.point = Vec3(0,0,0);
    floorPlane.normal = Vec3(0,1,0);
    floorPlane.material = makeMaterial(Vec3(0.34f,0.36f,0.39f),0.85f,0.0f,1.0f);

    camera.position = Vec3(0.0f,2.5f,7.8f);
    camera.yaw = PI;
    camera.pitch = -0.16f;
    camera.fov = 48.0f;
}

// ============================================================
// Camera/input
// ============================================================

static void mouseCallback(GLFWwindow*,double xpos,double ypos)
{
    if(firstMouse)
    {
        lastMouseX=xpos;
        lastMouseY=ypos;
        firstMouse=false;
        return;
    }

    double dx=xpos-lastMouseX;
    double dy=ypos-lastMouseY;
    lastMouseX=xpos;
    lastMouseY=ypos;

    // Only rotate while RMB is held.
    if(glfwGetMouseButton(glfwGetCurrentContext(),GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS)
        return;

    camera.yaw += static_cast<float>(dx)*mouseSensitivity;
    camera.pitch -= static_cast<float>(dy)*mouseSensitivity;
    camera.pitch = std::clamp(camera.pitch,-1.45f,1.45f);

    cameraMoving=true;
}

static bool updateCamera(GLFWwindow* window,float dt)
{
    bool moved=false;

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS)
    {
        glfwGetCursorPos(window,&lastMouseX,&lastMouseY);
        firstMouse=true;
    }

    Vec3 forward(
        std::cos(camera.pitch)*std::sin(camera.yaw),
        std::sin(camera.pitch),
        std::cos(camera.pitch)*std::cos(camera.yaw)
    );
    forward = forward / std::max(lengthv(forward),0.0001f);

    Vec3 right(forward.z,0.0f,-forward.x);
    float rl=lengthv(right);
    if(rl>0.0001f) right=right/rl;

    Vec3 up(0,1,0);

    float speed=moveSpeed*dt;
    if(glfwGetKey(window,GLFW_KEY_W)==GLFW_PRESS) { camera.position+=forward*speed; moved=true; }
    if(glfwGetKey(window,GLFW_KEY_S)==GLFW_PRESS) { camera.position-=forward*speed; moved=true; }
    if(glfwGetKey(window,GLFW_KEY_A)==GLFW_PRESS) { camera.position-=right*speed; moved=true; }
    if(glfwGetKey(window,GLFW_KEY_D)==GLFW_PRESS) { camera.position+=right*speed; moved=true; }
    if(glfwGetKey(window,GLFW_KEY_SPACE)==GLFW_PRESS) { camera.position+=up*speed; moved=true; }
    if(glfwGetKey(window,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS) { camera.position-=up*speed; moved=true; }

    return moved;
}

// ============================================================
// Render
// ============================================================

static void setCameraUniforms()
{
    Vec3 forward(
        std::cos(camera.pitch)*std::sin(camera.yaw),
        std::sin(camera.pitch),
        std::cos(camera.pitch)*std::cos(camera.yaw)
    );
    forward = forward / std::max(lengthv(forward),0.0001f);

    Vec3 right(
        forward.z,
        0.0f,
        -forward.x
    );
    right = right / std::max(lengthv(right),0.0001f);

    Vec3 up(
        -right.z*forward.y,
        right.x*forward.z-right.z*forward.x,
        right.x*forward.y
    );
    // Cross(right,forward), expanded.
    up = Vec3(
        right.y*forward.z-right.z*forward.y,
        right.z*forward.x-right.x*forward.z,
        right.x*forward.y-right.y*forward.x
    );
    up = up / std::max(lengthv(up),0.0001f);

    auto loc=[&](const char* n){ return glGetUniformLocation(computeProgram,n); };

    glUniform3f(loc("uCameraPosition"),camera.position.x,camera.position.y,camera.position.z);
    glUniform3f(loc("uForward"),forward.x,forward.y,forward.z);
    glUniform3f(loc("uRight"),right.x,right.y,right.z);
    glUniform3f(loc("uUp"),up.x,up.y,up.z);
    glUniform1f(loc("uAspect"),float(RENDER_WIDTH)/float(RENDER_HEIGHT));
    glUniform1f(loc("uTanFov"),std::tan(camera.fov*0.5f*PI/180.0f));
    glUniform1i(loc("uFrame"),frameIndex);
    glUniform1i(loc("uReset"),accumulationReset?1:0);
    glUniform1i(loc("uRenderWidth"),RENDER_WIDTH);
    glUniform1i(loc("uRenderHeight"),RENDER_HEIGHT);
    glUniform1f(loc("uFoveaRadius"),FOVEA_RADIUS);
    glUniform1f(loc("uMidRadius"),MID_RADIUS);
    glUniform1i(loc("uFoveaSPP"),FOVEA_SPP);
    glUniform1i(loc("uMaxBounces"),MAX_BOUNCES);
    glUniform3f(loc("uLightPosition"),-3.0f,5.5f,-2.0f);
    glUniform3f(loc("uLightColor"),7.0f,7.0f,7.0f);
}

static void renderFrame()
{
    glUseProgram(computeProgram);

    glBindImageTexture(0,accumulationTexture,0,GL_FALSE,0,GL_READ_WRITE,GL_RGBA32F);
    glBindImageTexture(1,displayTexture,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA8);

    setCameraUniforms();

    GLuint groupsX=(RENDER_WIDTH+7)/8;
    GLuint groupsY=(RENDER_HEIGHT+7)/8;
    glDispatchCompute(groupsX,groupsY,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    accumulationReset=false;
    frameIndex++;
}

static void display()
{
    glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(displayProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,displayTexture);
    glUniform1i(glGetUniformLocation(displayProgram,"uImage"),0);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES,0,3);
    glBindVertexArray(0);
}

// ============================================================
// PNG screenshot
// ============================================================

static bool savePNG(const char* filename)
{
    std::vector<unsigned char> pixels(static_cast<size_t>(RENDER_WIDTH)*RENDER_HEIGHT*4);

    glBindTexture(GL_TEXTURE_2D,displayTexture);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    glBindTexture(GL_TEXTURE_2D,0);

    FILE* fp=std::fopen(filename,"wb");
    if(!fp) return false;

    png_structp png=png_create_write_struct(PNG_LIBPNG_VER_STRING,nullptr,nullptr,nullptr);
    if(!png) { std::fclose(fp); return false; }
    png_infop info=png_create_info_struct(png);
    if(!info) { png_destroy_write_struct(&png,nullptr); std::fclose(fp); return false; }

    if(setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png,&info);
        std::fclose(fp);
        return false;
    }

    png_init_io(png,fp);
    png_set_IHDR(png,info,RENDER_WIDTH,RENDER_HEIGHT,8,PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png,info);

    std::vector<png_bytep> rows(RENDER_HEIGHT);
    for(int y=0;y<RENDER_HEIGHT;y++)
        rows[y]=pixels.data()+static_cast<size_t>(y)*RENDER_WIDTH*4;

    png_write_image(png,rows.data());
    png_write_end(png,nullptr);
    png_destroy_write_struct(&png,&info);
    std::fclose(fp);
    return true;
}

// ============================================================
// Cleanup
// ============================================================

static void cleanup()
{
    if(displaySampler) glDeleteSamplers(1,&displaySampler);
    if(vao) glDeleteVertexArrays(1,&vao);
    if(displayTexture) glDeleteTextures(1,&displayTexture);
    if(accumulationTexture) glDeleteTextures(1,&accumulationTexture);
    if(sphereSSBO) glDeleteBuffers(1,&sphereSSBO);
    if(bvhSSBO) glDeleteBuffers(1,&bvhSSBO);
    if(computeProgram) glDeleteProgram(computeProgram);
    if(displayProgram) glDeleteProgram(displayProgram);
}

// ============================================================
// Main
// ============================================================

int main()
{
    if(!glfwInit())
    {
        std::fprintf(stderr,"GLFW initialization failed.\n");
        return 1;
    }

    // Compute shaders require OpenGL 4.3.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window=glfwCreateWindow(
        WINDOW_WIDTH,WINDOW_HEIGHT,
        "GPU Perceptual Ray Tracer + BVH",
        nullptr,nullptr
    );

    if(!window)
    {
        std::fprintf(stderr,
                     "Could not create an OpenGL 4.3 window.\n"
                     "Your GPU/driver must support compute shaders.\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // application controls the frame limiter

    glewExperimental=GL_TRUE;
    if(glewInit()!=GLEW_OK)
    {
        std::fprintf(stderr,"GLEW initialization failed.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const GLubyte* version=glGetString(GL_VERSION);
    const GLubyte* renderer=glGetString(GL_RENDERER);
    std::printf("OpenGL:  %s\n",version?reinterpret_cast<const char*>(version):"unknown");
    std::printf("GPU:     %s\n",renderer?reinterpret_cast<const char*>(renderer):"unknown");

    int major=0,minor=0;
    glGetIntegerv(GL_MAJOR_VERSION,&major);
    glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4 || (major==4 && minor<3))
    {
        std::fprintf(stderr,"OpenGL 4.3+ is required for compute shaders.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glfwSetCursorPosCallback(window,mouseCallback);

    setupScene();
    buildBVH();

    if(!initShaders())
    {
        cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    uploadScene();
    initTextures();
    glGenVertexArrays(1,&vao);

    clearAccumulation();

    std::printf(
        "\n"
        "==================================================\n"
        " GPU PERCEPTUAL RAY TRACER + BVH\n"
        "==================================================\n"
        "Render:          %dx%d\n"
        "Window:          %dx%d\n"
        "Target FPS:      %.0f\n"
        "Fovea radius:    %.2f\n"
        "Fovea samples:   %d\n"
        "Max bounces:     %d\n"
        "BVH nodes:       %zu\n"
        "\n"
        "W A S D          Move\n"
        "SPACE            Up\n"
        "LEFT SHIFT       Down\n"
        "RIGHT MOUSE      Look\n"
        "R                Reset accumulation\n"
        "P                Save screenshot.png\n"
        "ESC              Quit\n"
        "\n"
        "GPU compute ray tracing enabled.\n"
        "GPU BVH traversal enabled.\n"
        "Deterministic clear-glass reflection/refraction enabled.\n"
        "Foveated + temporal peripheral sampling enabled.\n"
        "==================================================\n\n",
        RENDER_WIDTH,RENDER_HEIGHT,
        WINDOW_WIDTH,WINDOW_HEIGHT,
        TARGET_FPS,
        FOVEA_RADIUS,
        FOVEA_SPP,
        MAX_BOUNCES,
        bvh.size()
    );

    double previousTime=glfwGetTime();
    double fpsAccumulator=0.0;
    int fpsFrames=0;
    bool previousR=false;
    bool previousP=false;

    while(!glfwWindowShouldClose(window))
    {
        double frameStart=glfwGetTime();
        float dt=static_cast<float>(std::min(frameStart-previousTime,0.1));
        previousTime=frameStart;

        glfwPollEvents();

        bool moved=updateCamera(window,dt) || cameraMoving;

        if(moved)
        {
            clearAccumulation();
            cameraMoving=false;
        }

        bool rDown=glfwGetKey(window,GLFW_KEY_R)==GLFW_PRESS;
        if(rDown && !previousR)
        {
            clearAccumulation();
            std::printf("Accumulation reset.\n");
        }
        previousR=rDown;

        bool pDown=glfwGetKey(window,GLFW_KEY_P)==GLFW_PRESS;
        if(pDown && !previousP)
        {
            if(savePNG("screenshot.png"))
                std::printf("Saved screenshot.png\n");
            else
                std::printf("Could not save screenshot.\n");
        }
        previousP=pDown;

        if(glfwGetKey(window,GLFW_KEY_ESCAPE)==GLFW_PRESS)
            glfwSetWindowShouldClose(window,GL_TRUE);

        renderFrame();
        display();
        glfwSwapBuffers(window);

        fpsAccumulator += glfwGetTime()-frameStart;
        fpsFrames++;

        if(fpsAccumulator >= 1.0)
        {
            std::printf("GPU FPS: %.1f\n",fpsFrames/fpsAccumulator);
            fpsAccumulator=0.0;
            fpsFrames=0;
        }

        double elapsed=glfwGetTime()-frameStart;
        if(elapsed < TARGET_FRAME_TIME)
        {
            double remaining=TARGET_FRAME_TIME-elapsed;
            if(remaining > 0.002)
                std::this_thread::sleep_for(std::chrono::duration<double>(remaining-0.001));
            while(glfwGetTime()-frameStart < TARGET_FRAME_TIME)
                ;
        }
    }

    cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
