// ============================================================
// GPU PERCEPTUAL RAY TRACER + BVH - OpenGL 4.1 / macOS
// ============================================================
//
// This version deliberately avoids compute shaders and SSBOs so it
// works with Apple's OpenGL 4.1 ceiling (including macOS Big Sur).
//
// GPU:
//   - Fragment-shader ray generation
//   - BVH traversal using scene-data textures
//   - Clear transmissive glass
//   - Foveated sampling
//   - Temporal accumulation (ping-pong RGBA32F textures)
//   - Tone mapping/display
//
// CPU:
//   - GLFW input/window
//   - GLEW loading
//   - BVH construction/upload
//   - PNG screenshots
//
// The BVH is built once on the CPU. Spheres and BVH nodes are uploaded
// as RGBA32F textures because OpenGL 4.1 has no shader-storage buffers.
// Integer indices are stored as floats; all indices here are small
// enough to be represented exactly by a 32-bit float.
//
// macOS build example:
//   clang++ raytracer_gpu_bvh_gl41.cpp -std=c++17 -O2 \
//       -lglfw -lGLEW -lpng -framework OpenGL -o raytracer_gpu_bvh_gl41
//
// CONTROLS
//   W A S D       Move
//   SPACE         Up
//   LEFT SHIFT    Down
//   RMB           Look
//   R             Reset accumulation
//   P             Save screenshot.png
//   ESC           Quit
// ============================================================

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <png.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

static constexpr float FOVEA_RADIUS = 0.24f;
static constexpr float MID_RADIUS   = 0.62f;
static constexpr int FOVEA_SPP      = 2;
static constexpr int MAX_BOUNCES    = 4;
static constexpr float PI           = 3.14159265358979323846f;
static constexpr float EPSILON      = 0.001f;

// ============================================================
// CPU math
// ============================================================

struct Vec3
{
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float v) : x(v), y(v), z(v) {}
    Vec3(float X,float Y,float Z) : x(X),y(Y),z(Z) {}

    Vec3 operator+(const Vec3& b) const { return Vec3(x+b.x,y+b.y,z+b.z); }
    Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x,y-b.y,z-b.z); }
    Vec3 operator-() const { return Vec3(-x,-y,-z); }
    Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s,y/s,z/s); }
    Vec3& operator+=(const Vec3& b) { x+=b.x;y+=b.y;z+=b.z;return *this; }
    Vec3& operator-=(const Vec3& b) { x-=b.x;y-=b.y;z-=b.z;return *this; }
    Vec3& operator*=(float s) { x*=s;y*=s;z*=s;return *this; }
    Vec3& operator/=(float s) { x/=s;y/=s;z/=s;return *this; }
};

static Vec3 minv(const Vec3& a,const Vec3& b)
{
    return Vec3(std::min(a.x,b.x),std::min(a.y,b.y),std::min(a.z,b.z));
}
static Vec3 maxv(const Vec3& a,const Vec3& b)
{
    return Vec3(std::max(a.x,b.x),std::max(a.y,b.y),std::max(a.z,b.z));
}
static float dotv(const Vec3& a,const Vec3& b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
static float lengthv(const Vec3& a) { return std::sqrt(dotv(a,a)); }

// ============================================================
// Scene
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

static Vec3 sphereMin(const Sphere& s) { return s.center-Vec3(s.radius); }
static Vec3 sphereMax(const Sphere& s) { return s.center+Vec3(s.radius); }
static Vec3 centroid(const Sphere& s) { return s.center; }

static Vec3 axisExtent(const Vec3& e) { return Vec3(std::fabs(e.x),std::fabs(e.y),std::fabs(e.z)); }

static int longestAxis(const Vec3& e)
{
    Vec3 a=axisExtent(e);
    if(a.x>a.y && a.x>a.z) return 0;
    if(a.y>a.z) return 1;
    return 2;
}

static float axisValue(const Vec3& v,int axis)
{
    return axis==0?v.x:(axis==1?v.y:v.z);
}

static void computeNodeBounds(int nodeIndex,int first,int count)
{
    Vec3 mn(std::numeric_limits<float>::max());
    Vec3 mx(-std::numeric_limits<float>::max());
    for(int i=0;i<count;i++)
    {
        const Sphere& s=spheres[bvhIndices[first+i]];
        mn=minv(mn,sphereMin(s));
        mx=maxv(mx,sphereMax(s));
    }
    bvh[nodeIndex].bmin=mn;
    bvh[nodeIndex].bmax=mx;
    bvh[nodeIndex].leftFirst=first;
    bvh[nodeIndex].count=count;
}

static int buildBVHRecursive(int first,int count)
{
    int nodeIndex=(int)bvh.size();
    bvh.push_back(BuildNode{});
    computeNodeBounds(nodeIndex,first,count);

    constexpr int LEAF_SIZE=4;
    if(count<=LEAF_SIZE) return nodeIndex;

    Vec3 cmin(std::numeric_limits<float>::max());
    Vec3 cmax(-std::numeric_limits<float>::max());
    for(int i=0;i<count;i++)
    {
        Vec3 c=centroid(spheres[bvhIndices[first+i]]);
        cmin=minv(cmin,c);
        cmax=maxv(cmax,c);
    }

    int axis=longestAxis(cmax-cmin);
    int mid=first+count/2;
    std::nth_element(bvhIndices.begin()+first,
                     bvhIndices.begin()+mid,
                     bvhIndices.begin()+first+count,
                     [axis](int a,int b)
                     {
                         return axisValue(centroid(spheres[a]),axis)<
                                axisValue(centroid(spheres[b]),axis);
                     });

    int left=buildBVHRecursive(first,mid-first);
    int right=buildBVHRecursive(mid,first+count-mid);
    bvh[nodeIndex].leftFirst=left;
    bvh[nodeIndex].count=0;
    bvh[nodeIndex].rightChild=right;
    return nodeIndex;
}

static void buildBVH()
{
    bvh.clear();
    bvhIndices.resize(spheres.size());
    for(size_t i=0;i<spheres.size();i++) bvhIndices[i]=(int)i;
    if(!spheres.empty()) buildBVHRecursive(0,(int)spheres.size());
    std::printf("BVH: %zu spheres, %zu nodes\n",spheres.size(),bvh.size());
}

// ============================================================
// Camera
// ============================================================

struct Camera
{
    Vec3 position;
    float yaw=PI;
    float pitch=-0.16f;
    float fov=48.0f;
};

static Camera camera;
static bool firstMouse=true;
static double lastMouseX=WINDOW_WIDTH*0.5;
static double lastMouseY=WINDOW_HEIGHT*0.5;
static float mouseSensitivity=0.0025f;
static float moveSpeed=4.0f;
static bool cameraMoving=false;

// ============================================================
// OpenGL handles
// ============================================================

static GLuint rayProgram=0;
static GLuint displayProgram=0;
static GLuint sphereTexture=0;
static GLuint bvhTexture=0;
static GLuint accumTextures[2]={0,0};
static GLuint accumFBO=0;
static GLuint displayVAO=0;
static int readAccum=0;
static int frameIndex=0;
static bool accumulationReset=true;

// ============================================================
// GLSL 4.10 fragment-shader ray tracer
// ============================================================

static const char* rayVertexShaderSource=R"GLSL(
#version 410 core
out vec2 vUV;
void main()
{
    const vec2 pos[3]=vec2[](
        vec2(-1.0,-1.0),
        vec2( 3.0,-1.0),
        vec2(-1.0, 3.0)
    );
    vec2 p=pos[gl_VertexID];
    vUV=p*0.5+0.5;
    gl_Position=vec4(p,0.0,1.0);
}
)GLSL";

static const char* rayFragmentShaderSource=R"GLSL(
#version 410 core

in vec2 vUV;
layout(location=0) out vec4 fragAccum;

uniform sampler2D uPreviousAccum;
uniform sampler2D uSphereData;
uniform sampler2D uBVHData;

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

const float PI=3.14159265358979323846;
const float EPS=0.001;
const float INF=1e30;

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
    x^=x>>16;
    x*=0x7feb352du;
    x^=x>>15;
    x*=0x846ca68bu;
    x^=x>>16;
    return x;
}

float rand(inout uint state)
{
    state=hashU(state+0x9e3779b9u);
    return float(state&0x00ffffffu)/16777216.0;
}

vec4 sphereTexel(int sphereIndex,int row)
{
    return texelFetch(uSphereData,ivec2(sphereIndex,row),0);
}

vec4 bvhTexel(int nodeIndex,int row)
{
    return texelFetch(uBVHData,ivec2(nodeIndex,row),0);
}

vec3 sky(vec3 d)
{
    float t=0.5*(d.y+1.0);
    vec3 horizon=vec3(0.72,0.80,0.95);
    vec3 zenith=vec3(0.16,0.25,0.42);
    return mix(horizon,zenith,clamp(t,0.0,1.0));
}

bool hitAABB(vec3 ro,vec3 rd,vec3 mn,vec3 mx,float maxT)
{
    // Small directional epsilon avoids NaNs from exactly-zero ray components.
    vec3 safeRd=rd;
    safeRd.x=(abs(safeRd.x)<1e-8)?(safeRd.x<0.0?-1e-8:1e-8):safeRd.x;
    safeRd.y=(abs(safeRd.y)<1e-8)?(safeRd.y<0.0?-1e-8:1e-8):safeRd.y;
    safeRd.z=(abs(safeRd.z)<1e-8)?(safeRd.z<0.0?-1e-8:1e-8):safeRd.z;
    vec3 inv=1.0/safeRd;
    vec3 t0=(mn-ro)*inv;
    vec3 t1=(mx-ro)*inv;
    vec3 lo=min(t0,t1);
    vec3 hi=max(t0,t1);
    float enter=max(max(lo.x,lo.y),lo.z);
    float exit=min(min(hi.x,hi.y),hi.z);
    return exit>=max(enter,0.0) && enter<maxT;
}

bool hitSphere(vec3 ro,vec3 rd,int sphereIndex,inout Hit h)
{
    vec4 cr=sphereTexel(sphereIndex,0);
    vec4 ar=sphereTexel(sphereIndex,1);
    vec4 pa=sphereTexel(sphereIndex,2);
    vec3 c=cr.xyz;
    float r=cr.w;
    vec3 oc=ro-c;
    float b=dot(oc,rd);
    float c2=dot(oc,oc)-r*r;
    float disc=b*b-c2;
    if(disc<0.0) return false;
    float root=sqrt(disc);
    float t=-b-root;
    if(t<EPS) t=-b+root;
    if(t<EPS || t>=h.t) return false;

    h.t=t;
    h.position=ro+rd*t;
    h.normal=normalize(h.position-c);
    h.albedo=ar.xyz;
    h.roughness=ar.w;
    h.transmission=pa.x;
    h.ior=pa.y;
    h.emission=pa.z;
    h.hit=true;
    return true;
}

bool intersectScene(vec3 ro,vec3 rd,out Hit outHit)
{
    Hit h;
    h.t=INF;
    h.hit=false;

    // Infinite floor, deliberately outside the BVH.
    float denom=dot(rd,vec3(0,1,0));
    if(abs(denom)>1e-6)
    {
        float t=-ro.y/denom;
        if(t>EPS && t<h.t)
        {
            h.t=t;
            h.position=ro+rd*t;
            h.normal=vec3(0,1,0);
            h.albedo=vec3(0.34,0.36,0.39);
            h.roughness=0.85;
            h.transmission=0.0;
            h.ior=1.0;
            h.emission=0.0;
            h.hit=true;
        }
    }

    int nodeCount=textureSize(uBVHData,0).x;
    if(nodeCount==0)
    {
        outHit=h;
        return h.hit;
    }

    int stack[64];
    int sp=0;
    stack[sp++]=0;

    while(sp>0)
    {
        int nodeIndex=stack[--sp];
        vec4 mn4=bvhTexel(nodeIndex,0);
        vec4 mx4=bvhTexel(nodeIndex,1);
        vec4 info=bvhTexel(nodeIndex,2);
        int leftFirst=int(info.x+0.5);
        int count=int(info.y+0.5);
        int rightChild=int(info.z+0.5);

        if(!hitAABB(ro,rd,mn4.xyz,mx4.xyz,h.t)) continue;

        if(count>0)
        {
            for(int i=0;i<count;i++)
                hitSphere(ro,rd,leftFirst+i,h);
        }
        else
        {
            if(sp+2<64)
            {
                // Cheap ordering: both children are pushed without a
                // distance sort. For coherent primary rays this is fine,
                // and keeps the shader simple on older GPUs.
                stack[sp++]=rightChild;
                stack[sp++]=leftFirst;
            }
        }
    }

    outHit=h;
    return h.hit;
}

vec3 cosineHemisphere(vec3 n,inout uint state)
{
    float u1=rand(state);
    float u2=rand(state);
    float r=sqrt(u1);
    float a=2.0*PI*u2;
    vec3 t=normalize(abs(n.x)>0.1?cross(vec3(0,1,0),n):cross(vec3(1,0,0),n));
    vec3 b=cross(n,t);
    return normalize(t*(r*cos(a))+b*(r*sin(a))+n*sqrt(max(0.0,1.0-u1)));
}

vec3 reflectDir(vec3 d,vec3 n)
{
    return d-2.0*dot(d,n)*n;
}

bool refractDir(vec3 d,vec3 n,float eta,out vec3 result)
{
    float cosi=clamp(-dot(d,n),-1.0,1.0);
    float etai=1.0;
    float etat=eta;
    vec3 nn=n;
    if(cosi<0.0)
    {
        cosi=-cosi;
        float tmp=etai; etai=etat; etat=tmp;
        nn=-nn;
    }
    float ratio=etai/etat;
    float k=1.0-ratio*ratio*(1.0-cosi*cosi);
    if(k<0.0) return false;
    result=normalize(d*ratio+nn*(ratio*cosi-sqrt(k)));
    return true;
}

float fresnelSchlick(vec3 d,vec3 n,float ior)
{
    float cosi=clamp(-dot(d,n),0.0,1.0);
    float r0=(1.0-ior)/(1.0+ior);
    r0*=r0;
    float q=1.0-cosi;
    return r0+(1.0-r0)*q*q*q*q*q;
}

vec3 directLight(Hit h)
{
    vec3 toLight=uLightPosition-h.position;
    float lightDist=length(toLight);
    toLight/=max(lightDist,0.0001);

    Hit shadow;
    bool blocked=intersectScene(h.position+h.normal*EPS,toLight,shadow) && shadow.t<lightDist;
    float ndl=max(dot(h.normal,toLight),0.0);
    if(blocked) return vec3(0.0);
    return h.albedo*uLightColor*ndl/(1.0+0.08*lightDist*lightDist);
}

vec3 traceReflection(vec3 ro,vec3 rd)
{
    Hit h;
    if(!intersectScene(ro,rd,h)) return sky(rd);
    if(h.transmission>0.5) return sky(rd);
    return directLight(h)+h.albedo*0.06+h.albedo*h.emission;
}

vec3 tracePath(vec3 ro,vec3 rd,inout uint state)
{
    vec3 radiance=vec3(0.0);
    vec3 throughput=vec3(1.0);

    for(int bounce=0;bounce<16;bounce++)
    {
        if(bounce>=uMaxBounces) break;

        Hit h;
        if(!intersectScene(ro,rd,h))
        {
            radiance+=throughput*sky(rd);
            break;
        }

        radiance+=throughput*h.albedo*h.emission;

        if(h.transmission>0.5)
        {
            // Stable clear-glass cheat: evaluate both components instead
            // of Russian roulette. This makes the orb remain see-through
            // and keeps reflections visible at low sample counts.
            float F=fresnelSchlick(rd,h.normal,h.ior);
            vec3 reflDir=normalize(reflectDir(rd,h.normal));

            // Perceptual glass cheat carried over from the CPU version:
            // keep a stable environment reflection even while the camera
            // is moving or the accumulation buffer has very few samples.
            // A small scene reflection is added only as a subtle detail.
            vec3 refl=sky(reflDir)*1.12;
            Hit reflHit;
            if(intersectScene(h.position+h.normal*EPS,reflDir,reflHit) && reflHit.transmission<=0.5)
                refl=mix(refl,reflHit.albedo*0.35+directLight(reflHit)*0.65,0.35);

            vec3 refrDir;
            vec3 refr=sky(rd);
            if(refractDir(rd,h.normal,h.ior,refrDir))
            {
                Hit inside;
                if(intersectScene(h.position-refrDir*EPS,refrDir,inside))
                {
                    if(inside.transmission>0.5)
                    {
                        vec3 exitDir;
                        if(refractDir(refrDir,inside.normal,inside.ior,exitDir))
                        {
                            Hit outside;
                            if(intersectScene(inside.position+exitDir*EPS,exitDir,outside))
                                refr=outside.albedo*0.12+sky(exitDir)*0.88;
                            else
                                refr=sky(exitDir);
                        }
                        else refr=sky(refrDir);
                    }
                    else
                    {
                        refr=inside.albedo*0.18+sky(refrDir)*0.82;
                    }
                }
                else refr=sky(refrDir);
            }

            radiance+=throughput*(refl*F+refr*(1.0-F))*h.albedo;
            break;
        }

        radiance+=throughput*(directLight(h)+h.albedo*0.025);

        float specWeight=0.12*(1.0-h.roughness);
        if(rand(state)<specWeight)
        {
            rd=normalize(reflectDir(rd,h.normal));
            ro=h.position+h.normal*EPS;
            throughput*=0.55;
        }
        else
        {
            rd=cosineHemisphere(h.normal,state);
            ro=h.position+h.normal*EPS;
            throughput*=h.albedo;
        }

        if(bounce>=2)
        {
            float survive=max(throughput.x,max(throughput.y,throughput.z));
            survive=clamp(survive,0.05,0.95);
            if(rand(state)>survive) break;
            throughput/=survive;
        }
    }
    return radiance;
}

vec3 toneMap(vec3 c)
{
    c*=1.15;
    c=c/(1.0+c);
    return pow(max(c,vec3(0.0)),vec3(1.0/2.2));
}

void main()
{
    vec2 pixel=vUV*vec2(uRenderWidth,uRenderHeight)-vec2(0.5);
    ivec2 ip=ivec2(clamp(floor(pixel+vec2(0.5)),vec2(0.0),vec2(uRenderWidth-1,uRenderHeight-1)));
    vec2 uv=(vec2(ip)+0.5)/vec2(uRenderWidth,uRenderHeight);

    vec2 centered=uv*2.0-1.0;
    centered.x*=float(uRenderHeight)/float(uRenderWidth);
    float radius=length(centered);

    int spp=1;
    if(radius<uFoveaRadius) spp=uFoveaSPP;
    else if(radius>uMidRadius && (uFrame&1)==1) spp=0;

    vec4 old=texture(uPreviousAccum,uv);
    vec3 oldSum=old.rgb;
    float oldCount=old.a;
    if(uReset!=0)
    {
        oldSum=vec3(0.0);
        oldCount=0.0;
    }

    vec3 sum=vec3(0.0);
    for(int s=0;s<spp;s++)
    {
        uint state=uint(ip.x+1)*1973u ^ uint(ip.y+1)*9277u ^ uint(uFrame+1)*26699u ^ uint(s+1)*31847u;
        vec2 jitter=vec2(rand(state)-0.5,rand(state)-0.5);
        vec2 ndc=((vec2(ip)+0.5+jitter)/vec2(uRenderWidth,uRenderHeight))*2.0-1.0;
        ndc.x*=uAspect;
        ndc.y*=-1.0;
        vec3 rd=normalize(uForward+uRight*(ndc.x*uTanFov)+uUp*(ndc.y*uTanFov));
        sum+=tracePath(uCameraPosition,rd,state);
    }

    if(spp>0)
    {
        oldSum+=sum;
        oldCount+=float(spp);
    }

    fragAccum=vec4(oldSum,oldCount);
}
)GLSL";

// ============================================================
// Display shader
// ============================================================

static const char* displayVertexShaderSource=R"GLSL(
#version 410 core
out vec2 vUV;
void main()
{
    const vec2 pos[3]=vec2[](
        vec2(-1.0,-1.0),
        vec2( 3.0,-1.0),
        vec2(-1.0, 3.0)
    );
    vec2 p=pos[gl_VertexID];
    vUV=p*0.5+0.5;
    gl_Position=vec4(p,0.0,1.0);
}
)GLSL";

static const char* displayFragmentShaderSource=R"GLSL(
#version 410 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uAccum;
void main()
{
    // The accumulation texture uses the same top-to-bottom convention
    // as the ray image, so flip Y for the normal OpenGL display quad.
    vec2 uv=vec2(vUV.x,1.0-vUV.y);
    vec4 a=texture(uAccum,uv);
    vec3 c=a.a>0.0?a.rgb/a.a:vec3(0.0);
    c*=1.15;
    c=c/(1.0+c);
    c=pow(max(c,vec3(0.0)),vec3(1.0/2.2));
    fragColor=vec4(c,1.0);
}
)GLSL";

// ============================================================
// Shader helpers
// ============================================================

static GLuint compileShader(GLenum type,const char* source)
{
    GLuint shader=glCreateShader(type);
    glShaderSource(shader,1,&source,nullptr);
    glCompileShader(shader);
    GLint ok=GL_FALSE;
    glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if(!ok)
    {
        GLint len=0;
        glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&len);
        std::vector<char> log((size_t)std::max(1,len));
        glGetShaderInfoLog(shader,len,nullptr,log.data());
        std::fprintf(stderr,"Shader compile error:\n%s\n",log.data());
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint linkProgram(std::initializer_list<GLuint> shaders)
{
    GLuint program=glCreateProgram();
    for(GLuint s:shaders) glAttachShader(program,s);
    glLinkProgram(program);
    GLint ok=GL_FALSE;
    glGetProgramiv(program,GL_LINK_STATUS,&ok);
    if(!ok)
    {
        GLint len=0;
        glGetProgramiv(program,GL_INFO_LOG_LENGTH,&len);
        std::vector<char> log((size_t)std::max(1,len));
        glGetProgramInfoLog(program,len,nullptr,log.data());
        std::fprintf(stderr,"Program link error:\n%s\n",log.data());
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

static bool initShaders()
{
    GLuint vs=compileShader(GL_VERTEX_SHADER,rayVertexShaderSource);
    GLuint fs=compileShader(GL_FRAGMENT_SHADER,rayFragmentShaderSource);
    if(!vs||!fs)
    {
        if(vs) glDeleteShader(vs);
        if(fs) glDeleteShader(fs);
        return false;
    }
    rayProgram=linkProgram({vs,fs});
    glDeleteShader(vs);
    glDeleteShader(fs);
    if(!rayProgram) return false;

    vs=compileShader(GL_VERTEX_SHADER,displayVertexShaderSource);
    fs=compileShader(GL_FRAGMENT_SHADER,displayFragmentShaderSource);
    if(!vs||!fs)
    {
        if(vs) glDeleteShader(vs);
        if(fs) glDeleteShader(fs);
        return false;
    }
    displayProgram=linkProgram({vs,fs});
    glDeleteShader(vs);
    glDeleteShader(fs);
    return displayProgram!=0;
}

// ============================================================
// Scene textures
// ============================================================

static void initSceneTextures()
{
    // Each sphere occupies one texel in each of three rows:
    // row 0 = center.xyz, radius
    // row 1 = albedo.xyz, roughness
    // row 2 = transmission, IOR, emission, unused
    glGenTextures(1,&sphereTexture);
    glBindTexture(GL_TEXTURE_2D,sphereTexture);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,(GLsizei)spheres.size(),3,0,
                 GL_RGBA,GL_FLOAT,nullptr);

    std::vector<float> sphereData(spheres.size()*3*4,0.0f);
    // The BVH leaves reference positions in bvhIndices. Store the spheres
    // in that same compact order so leaf indices remain valid for large scenes.
    for(size_t orderedIndex=0;orderedIndex<spheres.size();orderedIndex++)
    {
        const Sphere& s=spheres[bvhIndices[orderedIndex]];
        size_t i=orderedIndex;
        size_t base=i*4;
        sphereData[base+0]=s.center.x;
        sphereData[base+1]=s.center.y;
        sphereData[base+2]=s.center.z;
        sphereData[base+3]=s.radius;

        base=spheres.size()*4+i*4;
        sphereData[base+0]=s.material.albedo.x;
        sphereData[base+1]=s.material.albedo.y;
        sphereData[base+2]=s.material.albedo.z;
        sphereData[base+3]=s.material.roughness;

        base=2*spheres.size()*4+i*4;
        sphereData[base+0]=s.material.transmission;
        sphereData[base+1]=s.material.ior;
        sphereData[base+2]=s.material.emission;
        sphereData[base+3]=0.0f;
    }
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,(GLsizei)spheres.size(),3,GL_RGBA,GL_FLOAT,sphereData.data());
    glBindTexture(GL_TEXTURE_2D,0);

    // Each BVH node occupies one texel in each of three rows:
    // row 0 = bmin
    // row 1 = bmax
    // row 2 = leftFirst, count, rightChild, unused
    glGenTextures(1,&bvhTexture);
    glBindTexture(GL_TEXTURE_2D,bvhTexture);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,(GLsizei)bvh.size(),3,0,
                 GL_RGBA,GL_FLOAT,nullptr);

    std::vector<float> bvhData(bvh.size()*3*4,0.0f);
    for(size_t i=0;i<bvh.size();i++)
    {
        const BuildNode& n=bvh[i];
        size_t base=i*4;
        bvhData[base+0]=n.bmin.x;
        bvhData[base+1]=n.bmin.y;
        bvhData[base+2]=n.bmin.z;
        bvhData[base+3]=0.0f;

        base=bvh.size()*4+i*4;
        bvhData[base+0]=n.bmax.x;
        bvhData[base+1]=n.bmax.y;
        bvhData[base+2]=n.bmax.z;
        bvhData[base+3]=0.0f;

        base=2*bvh.size()*4+i*4;
        bvhData[base+0]=(float)n.leftFirst;
        bvhData[base+1]=(float)n.count;
        bvhData[base+2]=(float)n.rightChild;
        bvhData[base+3]=0.0f;
    }
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,(GLsizei)bvh.size(),3,GL_RGBA,GL_FLOAT,bvhData.data());
    glBindTexture(GL_TEXTURE_2D,0);
}

static void initAccumulation()
{
    glGenTextures(2,accumTextures);
    for(int i=0;i<2;i++)
    {
        glBindTexture(GL_TEXTURE_2D,accumTextures[i]);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,RENDER_WIDTH,RENDER_HEIGHT,0,
                     GL_RGBA,GL_FLOAT,nullptr);
    }
    glBindTexture(GL_TEXTURE_2D,0);

    glGenFramebuffers(1,&accumFBO);
    glBindFramebuffer(GL_FRAMEBUFFER,accumFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,accumTextures[0],0);
    GLenum draw=GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1,&draw);
    GLenum status=glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status!=GL_FRAMEBUFFER_COMPLETE)
        std::fprintf(stderr,"Accumulation framebuffer is incomplete: 0x%x\n",status);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}

static void clearAccumulation()
{
    accumulationReset=true;
    frameIndex=0;
    readAccum=0;
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

    floorPlane.point=Vec3(0,0,0);
    floorPlane.normal=Vec3(0,1,0);
    floorPlane.material=makeMaterial(Vec3(0.34f,0.36f,0.39f),0.85f,0.0f,1.0f);

    camera.position=Vec3(0.0f,2.5f,7.8f);
    camera.yaw=PI;
    camera.pitch=-0.16f;
    camera.fov=48.0f;
}

// ============================================================
// Camera/input
// ============================================================

static void mouseCallback(GLFWwindow* window,double xpos,double ypos)
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

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_RIGHT)!=GLFW_PRESS)
        return;

    camera.yaw+=(float)dx*mouseSensitivity;
    camera.pitch-=(float)dy*mouseSensitivity;
    camera.pitch=std::clamp(camera.pitch,-1.45f,1.45f);
    cameraMoving=true;
}

static bool updateCamera(GLFWwindow* window,float dt)
{
    bool moved=false;

    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_RIGHT)!=GLFW_PRESS)
    {
        glfwGetCursorPos(window,&lastMouseX,&lastMouseY);
        firstMouse=true;
    }

    Vec3 forward(
        std::cos(camera.pitch)*std::sin(camera.yaw),
        std::sin(camera.pitch),
        std::cos(camera.pitch)*std::cos(camera.yaw));
    forward/=std::max(lengthv(forward),0.0001f);

    Vec3 right(forward.z,0.0f,-forward.x);
    right/=std::max(lengthv(right),0.0001f);
    Vec3 up(0,1,0);

    float speed=moveSpeed*dt;
    if(glfwGetKey(window,GLFW_KEY_W)==GLFW_PRESS){camera.position+=forward*speed;moved=true;}
    if(glfwGetKey(window,GLFW_KEY_S)==GLFW_PRESS){camera.position-=forward*speed;moved=true;}
    if(glfwGetKey(window,GLFW_KEY_A)==GLFW_PRESS){camera.position-=right*speed;moved=true;}
    if(glfwGetKey(window,GLFW_KEY_D)==GLFW_PRESS){camera.position+=right*speed;moved=true;}
    if(glfwGetKey(window,GLFW_KEY_SPACE)==GLFW_PRESS){camera.position+=up*speed;moved=true;}
    if(glfwGetKey(window,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS){camera.position-=up*speed;moved=true;}

    return moved;
}

// ============================================================
// Render
// ============================================================

static void setRayUniforms(int writeIndex)
{
    Vec3 forward(
        std::cos(camera.pitch)*std::sin(camera.yaw),
        std::sin(camera.pitch),
        std::cos(camera.pitch)*std::cos(camera.yaw));
    forward/=std::max(lengthv(forward),0.0001f);

    Vec3 right(forward.z,0.0f,-forward.x);
    right/=std::max(lengthv(right),0.0001f);

    Vec3 up=Vec3(
        right.y*forward.z-right.z*forward.y,
        right.z*forward.x-right.x*forward.z,
        right.x*forward.y-right.y*forward.x);
    up/=std::max(lengthv(up),0.0001f);

    auto loc=[&](const char* n){return glGetUniformLocation(rayProgram,n);};

    glUniform1i(loc("uPreviousAccum"),0);
    glUniform1i(loc("uSphereData"),1);
    glUniform1i(loc("uBVHData"),2);
    glUniform3f(loc("uCameraPosition"),camera.position.x,camera.position.y,camera.position.z);
    glUniform3f(loc("uForward"),forward.x,forward.y,forward.z);
    glUniform3f(loc("uRight"),right.x,right.y,right.z);
    glUniform3f(loc("uUp"),up.x,up.y,up.z);
    glUniform1f(loc("uAspect"),(float)RENDER_WIDTH/(float)RENDER_HEIGHT);
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

    (void)writeIndex;
}

static void renderFrame()
{
    int writeAccum=1-readAccum;

    glBindFramebuffer(GL_FRAMEBUFFER,accumFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,accumTextures[writeAccum],0);
    glViewport(0,0,RENDER_WIDTH,RENDER_HEIGHT);

    glUseProgram(rayProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,accumTextures[readAccum]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D,sphereTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D,bvhTexture);

    setRayUniforms(writeAccum);
    glDrawArrays(GL_TRIANGLES,0,3);

    glBindTexture(GL_TEXTURE_2D,0);
    readAccum=writeAccum;
    accumulationReset=false;
    frameIndex++;
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}

static void display()
{
    glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(displayProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,accumTextures[readAccum]);
    glUniform1i(glGetUniformLocation(displayProgram,"uAccum"),0);
    glBindVertexArray(displayVAO);
    glDrawArrays(GL_TRIANGLES,0,3);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D,0);
}

// ============================================================
// Screenshot
// ============================================================

static bool savePNG(const char* filename)
{
    std::vector<unsigned char> pixels((size_t)RENDER_WIDTH*RENDER_HEIGHT*4);

    glBindTexture(GL_TEXTURE_2D,accumTextures[readAccum]);
    std::vector<float> accum((size_t)RENDER_WIDTH*RENDER_HEIGHT*4);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,accum.data());
    glBindTexture(GL_TEXTURE_2D,0);

    for(int y=0;y<RENDER_HEIGHT;y++)
    {
        for(int x=0;x<RENDER_WIDTH;x++)
        {
            size_t i=((size_t)y*RENDER_WIDTH+(size_t)x)*4;
            float count=accum[i+3];
            Vec3 c(0.0f);
            if(count>0.0f)
                c=Vec3(accum[i+0],accum[i+1],accum[i+2])/count;
            c*=1.15f;
            c=Vec3(c.x/(1.0f+c.x),c.y/(1.0f+c.y),c.z/(1.0f+c.z));
            c=Vec3(std::pow(std::max(c.x,0.0f),1.0f/2.2f),
                   std::pow(std::max(c.y,0.0f),1.0f/2.2f),
                   std::pow(std::max(c.z,0.0f),1.0f/2.2f));
            pixels[i+0]=(unsigned char)std::clamp(c.x*255.0f,0.0f,255.0f);
            pixels[i+1]=(unsigned char)std::clamp(c.y*255.0f,0.0f,255.0f);
            pixels[i+2]=(unsigned char)std::clamp(c.z*255.0f,0.0f,255.0f);
            pixels[i+3]=255;
        }
    }

    FILE* fp=std::fopen(filename,"wb");
    if(!fp) return false;
    png_structp png=png_create_write_struct(PNG_LIBPNG_VER_STRING,nullptr,nullptr,nullptr);
    if(!png){std::fclose(fp);return false;}
    png_infop info=png_create_info_struct(png);
    if(!info){png_destroy_write_struct(&png,nullptr);std::fclose(fp);return false;}
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
    // glGetTexImage is bottom-to-top, while PNG viewers expect top-to-bottom.
    for(int y=0;y<RENDER_HEIGHT;y++)
        rows[y]=pixels.data()+(size_t)(RENDER_HEIGHT-1-y)*RENDER_WIDTH*4;
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
    if(displayVAO) glDeleteVertexArrays(1,&displayVAO);
    if(accumFBO) glDeleteFramebuffers(1,&accumFBO);
    if(accumTextures[0]||accumTextures[1]) glDeleteTextures(2,accumTextures);
    if(sphereTexture) glDeleteTextures(1,&sphereTexture);
    if(bvhTexture) glDeleteTextures(1,&bvhTexture);
    if(rayProgram) glDeleteProgram(rayProgram);
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

    // Apple exposes OpenGL 4.1 as its maximum core profile.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,1);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
#endif

    GLFWwindow* window=glfwCreateWindow(
        WINDOW_WIDTH,WINDOW_HEIGHT,
        "GPU Perceptual Ray Tracer + BVH (OpenGL 4.1)",
        nullptr,nullptr);

    if(!window)
    {
        std::fprintf(stderr,"Could not create an OpenGL 4.1 window.\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glewExperimental=GL_TRUE;
    GLenum glewStatus=glewInit();
    // GLEW can generate one harmless GL_INVALID_ENUM on core macOS contexts.
    glGetError();
    if(glewStatus!=GLEW_OK)
    {
        std::fprintf(stderr,"GLEW initialization failed.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const GLubyte* version=glGetString(GL_VERSION);
    const GLubyte* renderer=glGetString(GL_RENDERER);
    const GLubyte* glsl=glGetString(GL_SHADING_LANGUAGE_VERSION);
    std::printf("OpenGL:   %s\n",version?reinterpret_cast<const char*>(version):"unknown");
    std::printf("GPU:      %s\n",renderer?reinterpret_cast<const char*>(renderer):"unknown");
    std::printf("GLSL:     %s\n",glsl?reinterpret_cast<const char*>(glsl):"unknown");

    int major=0,minor=0;
    glGetIntegerv(GL_MAJOR_VERSION,&major);
    glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4 || (major==4 && minor<1))
    {
        std::fprintf(stderr,"OpenGL 4.1 is required.\n");
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

    initSceneTextures();
    initAccumulation();
    glGenVertexArrays(1,&displayVAO);

    clearAccumulation();

    std::printf(
        "\n==================================================\n"
        " GPU PERCEPTUAL RAY TRACER + BVH\n"
        " OpenGL 4.1 fragment-shader backend\n"
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
        "GPU fragment-shader ray tracing enabled.\n"
        "GPU BVH traversal enabled through scene textures.\n"
        "Deterministic clear-glass reflection/refraction enabled.\n"
        "Foveated + temporal peripheral sampling enabled.\n"
        "==================================================\n\n",
        RENDER_WIDTH,RENDER_HEIGHT,
        WINDOW_WIDTH,WINDOW_HEIGHT,
        TARGET_FPS,FOVEA_RADIUS,FOVEA_SPP,MAX_BOUNCES,bvh.size());

    double previousTime=glfwGetTime();
    double fpsAccumulator=0.0;
    int fpsFrames=0;
    bool previousR=false;
    bool previousP=false;

    while(!glfwWindowShouldClose(window))
    {
        double frameStart=glfwGetTime();
        float dt=(float)std::min(frameStart-previousTime,0.1);
        previousTime=frameStart;

        glfwPollEvents();

        bool moved=updateCamera(window,dt)||cameraMoving;
        if(moved)
        {
            clearAccumulation();
            cameraMoving=false;
        }

        bool rDown=glfwGetKey(window,GLFW_KEY_R)==GLFW_PRESS;
        if(rDown&&!previousR)
        {
            clearAccumulation();
            std::printf("Accumulation reset.\n");
        }
        previousR=rDown;

        bool pDown=glfwGetKey(window,GLFW_KEY_P)==GLFW_PRESS;
        if(pDown&&!previousP)
        {
            if(savePNG("screenshot.png")) std::printf("Saved screenshot.png\n");
            else std::printf("Could not save screenshot.\n");
        }
        previousP=pDown;

        if(glfwGetKey(window,GLFW_KEY_ESCAPE)==GLFW_PRESS)
            glfwSetWindowShouldClose(window,GL_TRUE);

        renderFrame();
        display();
        glfwSwapBuffers(window);

        fpsAccumulator+=glfwGetTime()-frameStart;
        fpsFrames++;
        if(fpsAccumulator>=1.0)
        {
            std::printf("GPU FPS: %.1f\n",fpsFrames/fpsAccumulator);
            fpsAccumulator=0.0;
            fpsFrames=0;
        }

        double elapsed=glfwGetTime()-frameStart;
        if(elapsed<TARGET_FRAME_TIME)
        {
            double remaining=TARGET_FRAME_TIME-elapsed;
            if(remaining>0.002)
                std::this_thread::sleep_for(std::chrono::duration<double>(remaining-0.001));
            while(glfwGetTime()-frameStart<TARGET_FRAME_TIME) {}
        }
    }

    cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
