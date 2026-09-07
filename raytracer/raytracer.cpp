// perceptual_cpu_raytracer.cpp
//
// CPU ray tracer + temporal accumulation + adaptive sampling +
// edge-aware "squint" reconstruction.
//
// External libraries:
//   OpenGL
//   GLFW
//   GLEW
//   libpng
//
// No CUDA, OpenCL, OpenMP, Embree, SDL, ImGui, Eigen, etc.
//
// Build example on Linux:
// g++ perceptual_cpu_raytracer.cpp -std=c++17 -O3 \
//     -lglfw -lGLEW -lGL -lpng -pthread -o perceptual_rt
//
// The CPU performs ALL ray tracing and reconstruction.
// OpenGL is only used to display the resulting framebuffer.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <png.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

static constexpr int WIDTH  = 960;
static constexpr int HEIGHT = 540;

static constexpr float PI = 3.14159265358979323846f;

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float v) : x(v), y(v), z(v) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    Vec3 operator+(const Vec3& b) const {
        return Vec3(x+b.x, y+b.y, z+b.z);
    }

    Vec3 operator-(const Vec3& b) const {
        return Vec3(x-b.x, y-b.y, z-b.z);
    }

    Vec3 operator-() const {
        return Vec3(-x,-y,-z);
    }

    Vec3 operator*(float b) const {
        return Vec3(x*b,y*b,z*b);
    }

    Vec3 operator/(float b) const {
        return *this * (1.0f/b);
    }

    Vec3& operator+=(const Vec3& b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }

Vec3& operator*=(float b) {
    x *= b;
    y *= b;
    z *= b;
    return *this;
}

Vec3& operator/=(float b) {
    x /= b;
    y /= b;
    z /= b;
    return *this;
}
};

static Vec3 operator*(float a, const Vec3& b) {
    return b*a;
}

static float dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x
    );
}

static float length(const Vec3& v) {
    return std::sqrt(dot(v,v));
}

static Vec3 normalize(const Vec3& v) {
    float l = length(v);
    return l > 0.0f ? v/l : Vec3(0);
}

static Vec3 multiply(const Vec3& a, const Vec3& b) {
    return Vec3(a.x*b.x,a.y*b.y,a.z*b.z);
}

static float clamp01(float x) {
    return std::max(0.0f,std::min(1.0f,x));
}

static Vec3 clamp01(const Vec3& v) {
    return Vec3(
        clamp01(v.x),
        clamp01(v.y),
        clamp01(v.z)
    );
}

static float luminance(const Vec3& c) {
    return 0.2126f*c.x +
           0.7152f*c.y +
           0.0722f*c.z;
}


// ------------------------------------------------------------
// Random generator
// ------------------------------------------------------------

struct RNG {
    uint32_t state;

    explicit RNG(uint32_t s) : state(s) {}

    uint32_t nextUInt() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    float next() {
        return (nextUInt() & 0x00ffffff) /
               float(0x01000000);
    }
};


// ------------------------------------------------------------
// Ray
// ------------------------------------------------------------

struct Ray {
    Vec3 origin;
    Vec3 direction;
};


// ------------------------------------------------------------
// Material
// ------------------------------------------------------------

struct Material {
    Vec3 albedo;
    float roughness;
    float emission;
};


// ------------------------------------------------------------
// Hit information
// ------------------------------------------------------------

struct Hit {
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
          hit(false) {}
};


// ------------------------------------------------------------
// Sphere
// ------------------------------------------------------------

struct Sphere {
    Vec3 center;
    float radius;
    Material material;

    bool intersect(const Ray& ray, Hit& hit) const {
        Vec3 oc = ray.origin-center;

        float b = dot(oc,ray.direction);
        float c = dot(oc,oc)-radius*radius;

        float h = b*b-c;

        if(h < 0.0f)
            return false;

        h = std::sqrt(h);

        float t = -b-h;

        if(t < 0.001f)
            t = -b+h;

        if(t < 0.001f)
            return false;

        if(t >= hit.t)
            return false;

        hit.t = t;
        hit.position = ray.origin + ray.direction*t;
        hit.normal = normalize(hit.position-center);
        hit.material = material;
        hit.hit = true;

        return true;
    }
};


// ------------------------------------------------------------
// Infinite plane
// ------------------------------------------------------------

struct Plane {
    Vec3 point;
    Vec3 normal;
    Material material;

    bool intersect(const Ray& ray, Hit& hit) const {
        float d = dot(ray.direction,normal);

        if(std::fabs(d) < 0.00001f)
            return false;

        float t = dot(point-ray.origin,normal)/d;

        if(t < 0.001f || t >= hit.t)
            return false;

        hit.t = t;
        hit.position = ray.origin + ray.direction*t;
        hit.normal = normal;
        hit.material = material;
        hit.hit = true;

        return true;
    }
};


// ------------------------------------------------------------
// Scene
// ------------------------------------------------------------

std::vector<Sphere> spheres;
Plane floorPlane;

static bool intersectScene(const Ray& ray, Hit& hit) {

    for(const Sphere& s : spheres)
        s.intersect(ray,hit);

    floorPlane.intersect(ray,hit);

    return hit.hit;
}


// ------------------------------------------------------------
// Cosine hemisphere sampling
// ------------------------------------------------------------

static Vec3 randomHemisphere(
    const Vec3& normal,
    RNG& rng)
{
    float u1 = rng.next();
    float u2 = rng.next();

    float r = std::sqrt(u1);
    float theta = 2.0f*PI*u2;

    Vec3 local(
        r*std::cos(theta),
        r*std::sin(theta),
        std::sqrt(std::max(0.0f,1.0f-u1))
    );

    Vec3 tangent;

    if(std::fabs(normal.x) > 0.1f)
        tangent = normalize(cross(Vec3(0,1,0),normal));
    else
        tangent = normalize(cross(Vec3(1,0,0),normal));

    Vec3 bitangent = cross(normal,tangent);

    return normalize(
        tangent*local.x +
        bitangent*local.y +
        normal*local.z
    );
}


// ------------------------------------------------------------
// Shadow ray
// ------------------------------------------------------------

static bool visibleToLight(
    const Vec3& position,
    const Vec3& lightPosition)
{
    Vec3 d = lightPosition-position;

    float distanceToLight = length(d);

    Ray ray;
    ray.origin = position + d/distanceToLight*0.002f;
    ray.direction = d/distanceToLight;

    Hit h;

    if(intersectScene(ray,h))
        return h.t > distanceToLight-0.01f;

    return true;
}


// ------------------------------------------------------------
// Path tracing
// ------------------------------------------------------------

static Vec3 trace(
    Ray ray,
    RNG& rng,
    int maxBounces)
{
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);

    const Vec3 lightPosition(-3.0f,5.5f,-2.0f);
    const Vec3 lightColor(8.0f,8.0f,8.0f);

    for(int bounce=0; bounce<maxBounces; ++bounce) {

        Hit hit;

        if(!intersectScene(ray,hit)) {

            // Simple sky.
            float t = 0.5f*(ray.direction.y+1.0f);

            Vec3 sky =
                Vec3(0.55f,0.70f,1.0f)*(1.0f-t) +
                Vec3(0.12f,0.20f,0.40f)*t;

            radiance += multiply(throughput,sky);
            break;
        }

        Material m = hit.material;

        if(m.emission > 0.0f) {
            radiance +=
                multiply(
                    throughput,
                    m.albedo*m.emission
                );

            break;
        }

        // Direct lighting.
        Vec3 toLight = lightPosition-hit.position;
        float lightDistance = length(toLight);
        Vec3 lightDir = toLight/lightDistance;

        float ndotl = std::max(
            0.0f,
            dot(hit.normal,lightDir)
        );

        if(ndotl > 0.0f &&
           visibleToLight(hit.position,lightPosition))
        {
            float attenuation =
                1.0f/(1.0f+0.08f*lightDistance*lightDistance);

            Vec3 direct =
                multiply(
                    m.albedo,
                    lightColor
                ) *
                (ndotl*attenuation);

            radiance += multiply(
                throughput,
                direct
            );
        }

        // Indirect bounce.
        throughput = multiply(
            throughput,
            m.albedo
        );

        // Russian roulette.
        if(bounce >= 2) {

            float p = std::max(
                0.1f,
                std::min(
                    0.95f,
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

            throughput /= p;
        }

        Vec3 nextDirection =
            randomHemisphere(hit.normal,rng);

        ray.origin =
            hit.position +
            hit.normal*0.002f;

        ray.direction = nextDirection;
    }

    return radiance;
}


// ------------------------------------------------------------
// Camera
// ------------------------------------------------------------

struct Camera {
    Vec3 position;
    float yaw;
    float pitch;
    float fov;

    Ray makeRay(
        float px,
        float py,
        float aspect,
        RNG& rng)
    {
        float jitterX = rng.next()-0.5f;
        float jitterY = rng.next()-0.5f;

        float x =
            ((px+jitterX)/WIDTH)*2.0f-1.0f;

        float y =
            1.0f-((py+jitterY)/HEIGHT)*2.0f;

        float tanFov =
            std::tan(fov*0.5f*PI/180.0f);

        x *= aspect*tanFov;
        y *= tanFov;

        Vec3 forward(
            std::cos(pitch)*std::sin(yaw),
            std::sin(pitch),
            std::cos(pitch)*std::cos(yaw)
        );

        forward = normalize(forward);

        Vec3 right =
            normalize(
                cross(
                    forward,
                    Vec3(0,1,0)
                )
            );

        Vec3 up =
            normalize(
                cross(right,forward)
            );

        Vec3 direction =
            normalize(
                forward +
                right*x +
                up*y
            );

        return {
            position,
            direction
        };
    }
};

Camera camera;


// ------------------------------------------------------------
// Frame buffers
// ------------------------------------------------------------

struct Pixel {
    Vec3 color;
    Vec3 normal;
    float depth;

    Pixel()
        : color(0),
          normal(0),
          depth(0) {}
};

std::vector<Pixel> currentFrame(WIDTH*HEIGHT);
std::vector<Pixel> previousFrame(WIDTH*HEIGHT);

std::vector<Vec3> accumulation(WIDTH*HEIGHT);
std::vector<Vec3> output(WIDTH*HEIGHT);

std::vector<float> variance(WIDTH*HEIGHT,0.0f);


// ------------------------------------------------------------
// Temporal accumulation
// ------------------------------------------------------------
//
// This is the important "AI-like" part.
//
// Instead of simply averaging everything forever:
//
//   history = old * 0.9 + new * 0.1
//
// we estimate whether the pixel is stable.
//
// Stable pixels get long temporal integration.
// Unstable pixels trust the new sample more.

static void temporalReconstruction()
{
    for(int y=0; y<HEIGHT; ++y) {
        for(int x=0; x<WIDTH; ++x) {

            int i = y*WIDTH+x;

            Vec3 current =
                currentFrame[i].color;

            Vec3 old =
                accumulation[i];

            if(y == 0 || x == 0 ||
               y == HEIGHT-1 ||
               x == WIDTH-1)
            {
                accumulation[i] = current;
                continue;
            }

            Vec3 oldNormal =
                previousFrame[i].normal;

            Vec3 newNormal =
                currentFrame[i].normal;

            float normalAgreement =
                std::max(
                    0.0f,
                    dot(oldNormal,newNormal)
                );

            float oldDepth =
                previousFrame[i].depth;

            float newDepth =
                currentFrame[i].depth;

            float depthAgreement = 1.0f;

            if(oldDepth > 0.0f &&
               newDepth > 0.0f)
            {
                float relative =
                    std::fabs(oldDepth-newDepth) /
                    std::max(0.001f,newDepth);

                depthAgreement =
                    std::exp(-relative*15.0f);
            }

            float stability =
                normalAgreement *
                depthAgreement;

            // History retention.
            //
            // This behaves like a confidence estimator.
            float historyWeight =
                0.05f +
                0.94f*stability;

            accumulation[i] =
                old*historyWeight +
                current*(1.0f-historyWeight);
        }
    }
}


// ------------------------------------------------------------
// Edge-aware squint filter
// ------------------------------------------------------------
//
// Ordinary blur:
//
//     average(left + right + up + down)
//
// causes edges to bleed.
//
// This version only mixes neighboring pixels when:
//   - their normals agree
//   - their depth is similar
//   - their colors aren't wildly different
//
// Conceptually this approximates:
//
//     "What would this noisy image look like
//      if my visual system integrated it?"

static void perceptualSquint()
{
    static std::vector<Vec3> temp(WIDTH*HEIGHT);

    for(int y=1; y<HEIGHT-1; ++y) {
        for(int x=1; x<WIDTH-1; ++x) {

            int i = y*WIDTH+x;

            Vec3 center =
                accumulation[i];

            Vec3 centerNormal =
                currentFrame[i].normal;

            float centerDepth =
                currentFrame[i].depth;

            Vec3 sum(0);
            float weightSum = 0.0f;

            for(int oy=-1; oy<=1; ++oy) {
                for(int ox=-1; ox<=1; ++ox) {

                    int j =
                        (y+oy)*WIDTH +
                        (x+ox);

                    Vec3 neighbor =
                        accumulation[j];

                    Vec3 neighborNormal =
                        currentFrame[j].normal;

                    float neighborDepth =
                        currentFrame[j].depth;

                    float spatial =
                        std::exp(
                            -float(ox*ox+oy*oy)*0.65f
                        );

                    float normalWeight =
                        std::pow(
                            std::max(
                                0.0f,
                                dot(
                                    centerNormal,
                                    neighborNormal
                                )
                            ),
                            8.0f
                        );

                    float depthWeight = 1.0f;

                    if(centerDepth > 0.0f &&
                       neighborDepth > 0.0f)
                    {
                        float dz =
                            std::fabs(
                                centerDepth-
                                neighborDepth
                            );

                        depthWeight =
                            std::exp(
                                -dz*6.0f
                            );
                    }

                    float colorDifference =
                        length(
                            center-neighbor
                        );

                    float colorWeight =
                        std::exp(
                            -colorDifference*2.0f
                        );

                    float w =
                        spatial*
                        normalWeight*
                        depthWeight*
                        colorWeight;

                    sum += neighbor*w;
                    weightSum += w;
                }
            }

            Vec3 blurred =
                weightSum > 0.0f
                ? sum/weightSum
                : center;

            // Don't completely blur.
            //
            // The history already represents a reconstructed
            // estimate. The squint stage supplies only the
            // missing low-frequency information.
            temp[i] =
                center*0.65f +
                blurred*0.35f;
        }
    }

    accumulation.swap(temp);
}


// ------------------------------------------------------------
// Adaptive sharpening
// ------------------------------------------------------------

static void perceptualSharpen()
{
    for(int y=1; y<HEIGHT-1; ++y) {
        for(int x=1; x<WIDTH-1; ++x) {

            int i = y*WIDTH+x;

            Vec3 c = accumulation[i];

            Vec3 n =
                accumulation[i-WIDTH];

            Vec3 s =
                accumulation[i+WIDTH];

            Vec3 e =
                accumulation[i+1];

            Vec3 w =
                accumulation[i-1];

            Vec3 average =
                (n+s+e+w)*0.25f;

            float localContrast =
                length(c-average);

            // Don't sharpen noisy regions aggressively.
            float amount =
                0.12f *
                std::exp(-localContrast*5.0f);

            output[i] =
                c+(c-average)*amount;
        }
    }
}


// ------------------------------------------------------------
// Tone mapping
// ------------------------------------------------------------

static Vec3 tonemap(Vec3 c)
{
    // Exposure.
    c *= 1.15f;

    // Simple Reinhard.
    c = Vec3(
        c.x/(1.0f+c.x),
        c.y/(1.0f+c.y),
        c.z/(1.0f+c.z)
    );

    // Gamma.
    c.x = std::pow(clamp01(c.x),1.0f/2.2f);
    c.y = std::pow(clamp01(c.y),1.0f/2.2f);
    c.z = std::pow(clamp01(c.z),1.0f/2.2f);

    return c;
}


// ------------------------------------------------------------
// CPU renderer
// ------------------------------------------------------------

static void renderTile(
    int y0,
    int y1,
    int samplesPerPixel,
    uint32_t seed)
{
    const float aspect =
        float(WIDTH)/float(HEIGHT);

    for(int y=y0; y<y1; ++y) {

        for(int x=0; x<WIDTH; ++x) {

            int i = y*WIDTH+x;

            RNG rng(
                seed ^
                uint32_t(x*1973) ^
                uint32_t(y*9277)
            );

            Vec3 color(0);

            Vec3 normalSum(0);
            float depthSum = 0.0f;

            for(int s=0; s<samplesPerPixel; ++s) {

                Ray ray =
                    camera.makeRay(
                        float(x),
                        float(y),
                        aspect,
                        rng
                    );

                Hit firstHit;

                Vec3 c =
                    trace(
                        ray,
                        rng,
                        4
                    );

                color += c;

                if(intersectScene(ray,firstHit)) {

                    normalSum += firstHit.normal;
                    depthSum += firstHit.t;
                }
            }

            color /=
                float(samplesPerPixel);

            currentFrame[i].color =
                color;

            if(length(normalSum) > 0.001f)
                currentFrame[i].normal =
                    normalize(normalSum);
            else
                currentFrame[i].normal =
                    Vec3(0,1,0);

            currentFrame[i].depth =
                depthSum/
                float(samplesPerPixel);
        }
    }
}


static void renderFrame()
{
    const unsigned threads =
        std::max(
            1u,
            std::thread::hardware_concurrency()
        );

    std::vector<std::thread> workers;

    int rowsPerThread =
        HEIGHT/int(threads);

    for(unsigned t=0; t<threads; ++t) {

        int y0 =
            int(t)*rowsPerThread;

        int y1 =
            (t == threads-1)
            ? HEIGHT
            : y0+rowsPerThread;

        workers.emplace_back(
            renderTile,
            y0,
            y1,
            2,             // intentionally low SPP
            1234u+t*991
        );
    }

    for(auto& thread : workers)
        thread.join();

    temporalReconstruction();

    perceptualSquint();

    perceptualSharpen();

    for(size_t i=0; i<output.size(); ++i)
        output[i] =
            tonemap(output[i]);

    previousFrame =
        currentFrame;
}


// ------------------------------------------------------------
// PNG output
// ------------------------------------------------------------

static bool savePNG(
    const char* filename)
{
    FILE* fp =
        std::fopen(filename,"wb");

    if(!fp)
        return false;

    png_structp png =
        png_create_write_struct(
            PNG_LIBPNG_VER_STRING,
            nullptr,
            nullptr,
            nullptr
        );

    if(!png) {
        fclose(fp);
        return false;
    }

    png_infop info =
        png_create_info_struct(png);

    if(!info) {
        png_destroy_write_struct(
            &png,nullptr
        );

        fclose(fp);
        return false;
    }

    if(setjmp(png_jmpbuf(png))) {

        png_destroy_write_struct(
            &png,&info
        );

        fclose(fp);

        return false;
    }

    png_init_io(png,fp);

    png_set_IHDR(
        png,
        info,
        WIDTH,
        HEIGHT,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png,info);

    std::vector<unsigned char> row(
        WIDTH*3
    );

    for(int y=0; y<HEIGHT; ++y) {

        for(int x=0; x<WIDTH; ++x) {

            Vec3 c =
                output[y*WIDTH+x];

            row[x*3+0] =
                uint8_t(clamp01(c.x)*255.0f);

            row[x*3+1] =
                uint8_t(clamp01(c.y)*255.0f);

            row[x*3+2] =
                uint8_t(clamp01(c.z)*255.0f);
        }

        png_write_row(
            png,
            row.data()
        );
    }

    png_write_end(png,nullptr);

    png_destroy_write_struct(
        &png,&info
    );

    fclose(fp);

    return true;
}


// ------------------------------------------------------------
// OpenGL presentation
// ------------------------------------------------------------

static GLuint texture = 0;

static void initGL()
{
    glGenTextures(1,&texture);
    glBindTexture(GL_TEXTURE_2D,texture);

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_NEAREST
    );

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_NEAREST
    );

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        WIDTH,
        HEIGHT,
        0,
        GL_RGB,
        GL_FLOAT,
        nullptr
    );
}


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
        WIDTH,
        HEIGHT,
        GL_RGB,
        GL_FLOAT,
        output.data()
    );

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);

    glTexCoord2f(0,0);
    glVertex2f(-1,-1);

    glTexCoord2f(1,0);
    glVertex2f(1,-1);

    glTexCoord2f(1,1);
    glVertex2f(1,1);

    glTexCoord2f(0,1);
    glVertex2f(-1,1);

    glEnd();

    glDisable(GL_TEXTURE_2D);
}


// ------------------------------------------------------------
// Scene setup
// ------------------------------------------------------------

static void setupScene()
{
    Material red;
    red.albedo = Vec3(0.85f,0.12f,0.08f);
    red.roughness = 0.4f;
    red.emission = 0.0f;

    Material blue;
    blue.albedo = Vec3(0.08f,0.25f,0.9f);
    blue.roughness = 0.25f;
    blue.emission = 0.0f;

    Material green;
    green.albedo = Vec3(0.1f,0.75f,0.18f);
    green.roughness = 0.7f;
    green.emission = 0.0f;

    Material white;
    white.albedo = Vec3(0.8f);
    white.roughness = 0.5f;
    white.emission = 0.0f;

    spheres.push_back({
        Vec3(-1.35f,1.0f,0.0f),
        1.0f,
        red
    });

    spheres.push_back({
        Vec3(1.1f,1.0f,-0.4f),
        1.0f,
        blue
    });

    spheres.push_back({
        Vec3(0.0f,0.65f,1.6f),
        0.65f,
        green
    });

    floorPlane.point =
        Vec3(0,-0.02f,0);

    floorPlane.normal =
        Vec3(0,1,0);

    floorPlane.material =
        white;

    camera.position =
        Vec3(0,2.4f,7.5f);

    camera.yaw =
        PI;

    camera.pitch =
        -0.18f;

    camera.fov =
        48.0f;
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    if(!glfwInit()) {
        std::fprintf(
            stderr,
            "GLFW initialization failed\n"
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
            WIDTH,
            HEIGHT,
            "CPU Perceptual Ray Tracer",
            nullptr,
            nullptr
        );

    if(!window) {

        glfwTerminate();

        return 1;
    }

    glfwMakeContextCurrent(window);

    if(glewInit() != GLEW_OK) {

        glfwDestroyWindow(window);
        glfwTerminate();

        return 1;
    }

    setupScene();
    initGL();

    std::printf(
        "CPU perceptual ray tracer started.\n"
    );

    std::printf(
        "Press P to save screenshot.png\n"
    );

    std::printf(
        "Rendering with CPU only.\n"
    );

    while(!glfwWindowShouldClose(window)) {

        renderFrame();

        display();

        glfwSwapBuffers(window);
        glfwPollEvents();

        if(glfwGetKey(window,GLFW_KEY_P)
           == GLFW_PRESS)
        {
            if(savePNG("screenshot.png"))
                std::printf(
                    "Saved screenshot.png\n"
                );
        }
    }

    glDeleteTextures(1,&texture);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}