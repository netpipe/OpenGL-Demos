#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define PI 3.14159265359f

// ============================================================================
// MATH UTILITIES
// ============================================================================
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3& b) const { return Vec3(x+b.x, y+b.y, z+b.z); }
    Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x, y-b.y, z-b.z); }
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s, y/s, z/s); } 
    Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
    Vec3& operator-=(const Vec3& b) { x-=b.x; y-=b.y; z-=b.z; return *this; } 
    float dot(const Vec3& b) const { return x*b.x + y*b.y + z*b.z; }
    Vec3 cross(const Vec3& b) const { return Vec3(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); }
    float length() const { return sqrtf(x*x + y*y + z*z); }
    Vec3 normalize() const { float l = length(); return l > 0 ? *this / l : Vec3(0,0,0); }
};

struct Mat4 {
    float m[16];
    Mat4() { for(int i=0; i<16; ++i) m[i] = (i%5==0) ? 1.0f : 0.0f; }
    float* ptr() { return m; }
};

Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Mat4 res;
    Vec3 f = (center - eye).normalize();
    Vec3 s = f.cross(up).normalize();
    Vec3 u = s.cross(f);
    res.m[0]=s.x; res.m[4]=s.y; res.m[8]=s.z;
    res.m[1]=u.x; res.m[5]=u.y; res.m[9]=u.z;
    res.m[2]=-f.x; res.m[6]=-f.y; res.m[10]=-f.z;
    res.m[12]=-s.dot(eye); res.m[13]=-u.dot(eye); res.m[14]=f.dot(eye);
    return res;
}

Mat4 perspective(float fov, float aspect, float zNear, float zFar) {
    Mat4 res;
    float tanHalfFov = tanf(fov / 2.0f);
    res.m[0] = 1.0f / (aspect * tanHalfFov);
    res.m[5] = 1.0f / tanHalfFov;
    res.m[10] = - (zFar + zNear) / (zFar - zNear);
    res.m[11] = -1.0f;
    res.m[14] = - (2.0f * zFar * zNear) / (zFar - zNear);
    res.m[15] = 0.0f;
    return res;
}

// ============================================================================
// AABB & QUADTREE (BSP HYBRID)
// ============================================================================
struct AABB {
    Vec3 min, max;
    bool intersectsSphere(Vec3 c, float r) const {
        float distSq = 0;
        if (c.x < min.x) distSq += (c.x - min.x) * (c.x - min.x);
        else if (c.x > max.x) distSq += (c.x - max.x) * (c.x - max.x);
        if (c.y < min.y) distSq += (c.y - min.y) * (c.y - min.y);
        else if (c.y > max.y) distSq += (c.y - max.y) * (c.y - max.y);
        if (c.z < min.z) distSq += (c.z - min.z) * (c.z - min.z);
        else if (c.z > max.z) distSq += (c.z - max.z) * (c.z - max.z);
        return distSq <= r * r;
    }
    Vec3 center() const { return (min + max) * 0.5f; }
};

class SceneNode {
public:
    enum class Type { OBJ_MESH, SPHERE, BOX, PLANE, WATER } type;
    Vec3 position;
    Vec3 dimensions = Vec3(1,1,1);
    float radius = 1.0f;
    Vec3 normal = Vec3(0,1,0);
    Vec3 color = Vec3(1,1,1);
    float reflectivity = 0.1f; 
    std::string objPath;
    GLuint VAO = 0;
    int indexCount = 0;
    AABB worldBounds;
};

class QuadtreeNode {
public:
    AABB bounds;
    std::vector<SceneNode*> objects;
    std::unique_ptr<QuadtreeNode> nw, ne, sw, se;
    bool divided = false;

    QuadtreeNode(AABB b) : bounds(b) {}

    void insert(SceneNode* node) {
        if (!bounds.intersectsSphere(node->position, node->radius)) return;
        if (objects.size() < 4 || !divided) {
            objects.push_back(node);
        } else {
            nw->insert(node); ne->insert(node); sw->insert(node); se->insert(node);
        }
        if (objects.size() > 4 && !divided) subdivide();
    }

    void subdivide() {
        Vec3 c = bounds.center();
        nw = std::unique_ptr<QuadtreeNode>(new QuadtreeNode(AABB{bounds.min, c}));
        ne = std::unique_ptr<QuadtreeNode>(new QuadtreeNode(AABB{Vec3(c.x, bounds.min.y, bounds.min.z), Vec3(bounds.max.x, c.y, c.z)}));
        sw = std::unique_ptr<QuadtreeNode>(new QuadtreeNode(AABB{Vec3(bounds.min.x, bounds.min.y, c.z), Vec3(c.x, c.y, bounds.max.z)}));
        se = std::unique_ptr<QuadtreeNode>(new QuadtreeNode(AABB{c, bounds.max}));
        divided = true;
    }

    void query(Vec3 center, float radius, std::vector<SceneNode*>& found) {
        if (!bounds.intersectsSphere(center, radius)) return;
        for (auto n : objects) {
            if ((n->position - center).length() < radius + n->radius) found.push_back(n);
        }
        if (divided) {
            nw->query(center, radius, found);
            ne->query(center, radius, found);
            sw->query(center, radius, found);
            se->query(center, radius, found);
        }
    }
};

struct Vec2Int {
    int x, y;
    bool operator==(const Vec2Int& o) const { return x == o.x && y == o.y; }
};
struct Vec2IntHash {
    size_t operator()(const Vec2Int& v) const {
        return std::hash<int>()(v.x) ^ (std::hash<int>()(v.y) << 1);
    }
};

// ============================================================================
// HOLODECK GRID
// ============================================================================
class HolodeckGrid {
public:
    struct Chunk {
        QuadtreeNode qtree;
        Chunk(AABB b) : qtree(b) {}
    };
    std::unordered_map<Vec2Int, std::unique_ptr<Chunk>, Vec2IntHash> chunks;
    float chunkSize;

    HolodeckGrid(float size = 20.0f) : chunkSize(size) {}

    void insert(SceneNode* node) {
        int cx = (int)std::floor(node->position.x / chunkSize);
        int cy = (int)std::floor(node->position.z / chunkSize);
        Vec2Int key = {cx, cy};
        if (chunks.find(key) == chunks.end()) {
            AABB bounds = {
                Vec3(key.x * chunkSize, -100, key.y * chunkSize),
                Vec3((key.x+1) * chunkSize, 100, (key.y+1) * chunkSize)
            };
            chunks[key] = std::unique_ptr<Chunk>(new Chunk(bounds));
        }
        chunks[key]->qtree.insert(node);
    }

    void collect(Vec3 camPos, float radius, std::vector<SceneNode*>& out) {
        int cx = (int)std::floor(camPos.x / chunkSize);
        int cy = (int)std::floor(camPos.z / chunkSize);
        int r = (int)std::ceil(radius / chunkSize) + 1;
        for (int i = -r; i <= r; ++i) {
            for (int j = -r; j <= r; ++j) {
                Vec2Int key = {cx+i, cy+j};
                if (chunks.find(key) != chunks.end()) {
                    chunks[key]->qtree.query(camPos, radius, out);
                }
            }
        }
    }
};

// ============================================================================
// OBJ LOADER
// ============================================================================
struct MeshData {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
};
MeshData loadOBJ(const std::string& path) {
    MeshData mesh;
    std::ifstream file(path);
    if (!file.is_open()) return mesh;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;
        if (prefix == "v") {
            float x, y, z; ss >> x >> y >> z;
            mesh.vertices.push_back(x); mesh.vertices.push_back(y); mesh.vertices.push_back(z);
        } else if (prefix == "f") {
            std::string s1, s2, s3; ss >> s1 >> s2 >> s3;
            auto getIdx = [](const std::string& s) { return std::stoi(s.substr(0, s.find('/'))) - 1; };
            mesh.indices.push_back(getIdx(s1));
            mesh.indices.push_back(getIdx(s2));
            mesh.indices.push_back(getIdx(s3));
        }
    }
    return mesh;
}

// ============================================================================
// SHADERS
// ============================================================================
const char* rasterVS = R"GLSL(
#version 410 core
layout (location = 0) in vec3 aPos;
uniform mat4 uModel, uView, uProj;
void main() { gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); }
)GLSL";

const char* rasterFS = R"GLSL(
#version 410 core
out vec4 FragColor;
uniform vec3 uColor;
void main() { FragColor = vec4(uColor, 1.0); }
)GLSL";

const char* rtVS = R"GLSL(
#version 410 core
layout (location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0); 
}
)GLSL";

const char* rtFS = R"GLSL(
#version 410 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uPrevFrame;
uniform float uFrameIndex;
uniform int uReset;
uniform vec2 uResolution;
uniform float uAspectRatio;

uniform vec3 uCamPos;
uniform vec3 uCamForward, uCamRight, uCamUp;
uniform float uScale;
uniform mat4 uView, uProj;
uniform float uTime;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform int hasWater;

struct Light {
    vec4 type_info; 
    vec4 pos;
    vec4 dir;
    vec4 color;
    vec4 params; 
    vec4 u;
    vec4 v;
};

struct Sphere { vec4 posRad; vec4 color; };
struct Box { vec4 pos; vec4 ext; vec4 color; };
struct Plane { vec4 pos; vec4 norm; vec4 color; };

layout(std140) uniform SphereBlock { Sphere spheres[256]; };
layout(std140) uniform BoxBlock { Box boxes[256]; };
layout(std140) uniform PlaneBlock { Plane planes[256]; };
layout(std140) uniform LightBlock { Light lights[32]; };

uniform int numSpheres, numBoxes, numPlanes, numLights;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

bool intersectSphere(vec3 ro, vec3 rd, vec3 c, float r, out float t) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float c2 = dot(oc, oc) - r*r;
    float h = b*b - c2;
    if(h < 0.0) return false;
    h = sqrt(h);
    t = -b - h;
    if(t < 0.0) t = -b + h;
    return t > 0.0;
}

bool intersectAABB(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out float t) {
    vec3 invDir = 1.0 / rd;
    vec3 t0 = (bmin - ro) * invDir;
    vec3 t1 = (bmax - ro) * invDir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tEnter = max(max(tmin.x, tmin.y), tmin.z);
    float tExit = min(min(tmax.x, tmax.y), tmax.z);
    if (tEnter > tExit || tExit < 0.0) return false;
    t = tEnter > 0.0 ? tEnter : tExit;
    return true;
}

bool intersectPlane(vec3 ro, vec3 rd, vec3 p, vec3 n, out float t) {
    float denom = dot(n, rd);
    if (abs(denom) > 1e-6) {
        t = dot(p - ro, n) / denom;
        return t > 0.0;
    }
    return false;
}

bool intersectWater(vec3 ro, vec3 rd, out float t, out vec3 normal) {
    vec3 p = vec3(0, -5, 0);
    vec3 n = vec3(0, 1, 0);
    if (!intersectPlane(ro, rd, p, n, t)) return false;
    vec3 hit = ro + rd * t;
    float dx = cos(hit.x * 0.5 + uTime) * 0.5 * cos(hit.z * 0.5 + uTime) * 0.2;
    float dz = sin(hit.x * 0.5 + uTime) * -sin(hit.z * 0.5 + uTime) * 0.5 * 0.2;
    normal = normalize(vec3(-dx, 1.0, -dz));
    return true;
}

// RESTORED FROM OLD CODE: Shadow Rays for crispness
bool inShadow(vec3 ro, vec3 rd, float maxDist) {
    float t;
    for(int i=0; i<numSpheres; ++i) {
        if(intersectSphere(ro, rd, spheres[i].posRad.xyz, spheres[i].posRad.w, t)) {
            if(t < maxDist) return true;
        }
    }
    for(int i=0; i<numBoxes; ++i) {
        vec3 bmin = boxes[i].pos.xyz - boxes[i].ext.xyz;
        vec3 bmax = boxes[i].pos.xyz + boxes[i].ext.xyz;
        if(intersectAABB(ro, rd, bmin, bmax, t)) {
            if(t < maxDist) return true;
        }
    }
    for(int i=0; i<numPlanes; ++i) {
        if(intersectPlane(ro, rd, planes[i].pos.xyz, planes[i].norm.xyz, t)) {
            if(t < maxDist) return true;
        }
    }
    return false;
}

vec3 calcLighting(vec3 pos, vec3 normal, vec3 albedo, vec3 viewDir) {
    vec3 result = vec3(0.05) * albedo; 
    vec3 shadowRo = pos + normal * 0.002;

    for(int i=0; i<numLights; ++i) {
        int lType = int(lights[i].type_info.x);
        vec3 lPos = lights[i].pos.xyz;
        vec3 lDir = lights[i].dir.xyz;
        vec3 lColor = lights[i].color.xyz;
        float intensity = lights[i].params.x;
        float range = lights[i].params.y;
        float cutoff = lights[i].params.z;
        float outerCutoff = lights[i].params.w;
        vec3 lU = lights[i].u.xyz;
        vec3 lV = lights[i].v.xyz;

        if (lType == 0) { 
            vec3 l = normalize(-lDir);
            float shadow = inShadow(shadowRo, l, 1e10) ? 0.15 : 1.0;
            float diff = max(dot(normal, l), 0.0) * shadow;
            vec3 h = normalize(l + viewDir);
            float spec = pow(max(dot(normal, h), 0.0), 32.0) * shadow;
            result += (diff * albedo + spec * 0.5) * lColor * intensity;
        } 
        else if (lType == 1) { 
            vec3 l = lPos - pos;
            float dist = length(l);
            l /= dist;
            float shadow = inShadow(shadowRo, l, dist) ? 0.15 : 1.0;
            float atten = 1.0 / (1.0 + 0.09*dist + 0.032*dist*dist);
            float diff = max(dot(normal, l), 0.0) * shadow;
            vec3 h = normalize(l + viewDir);
            float spec = pow(max(dot(normal, h), 0.0), 32.0) * shadow;
            result += (diff * albedo + spec * 0.5) * lColor * intensity * atten;
        }
        else if (lType == 2) { 
            vec3 l = lPos - pos;
            float dist = length(l);
            l /= dist;
            float shadow = inShadow(shadowRo, l, dist) ? 0.15 : 1.0;
            float theta = dot(l, normalize(-lDir));
            float epsilon = cutoff - outerCutoff;
            float inten = clamp((theta - outerCutoff) / epsilon, 0.0, 1.0);
            float diff = max(dot(normal, l), 0.0) * shadow;
            vec3 h = normalize(l + viewDir);
            float spec = pow(max(dot(normal, h), 0.0), 32.0) * shadow;
            result += (diff * albedo + spec * 0.5) * lColor * intensity * inten / (dist*dist);
        }
        else if (lType == 3) { 
            vec3 sum = vec3(0.0);
            int samples = 4;
            for(int x=0; x<samples; ++x) {
                for(int y=0; y<samples; ++y) {
                    vec3 samplePos = lPos + lU * ((float(x)+0.5)/float(samples) - 0.5) + lV * ((float(y)+0.5)/float(samples) - 0.5);
                    vec3 l = samplePos - pos;
                    float dist = length(l);
                    l /= dist;
                    float shadow = inShadow(shadowRo, l, dist) ? 0.15 : 1.0;
                    float diff = max(dot(normal, l), 0.0) * shadow;
                    sum += diff * albedo * lColor * intensity / (dist*dist);
                }
            }
            result += sum / float(samples*samples);
        }
    }
    return result;
}

vec3 calcGodrays(vec3 pos, vec3 lightDir, vec3 lightColor) {
    vec3 result = vec3(0.0);
    float stepSize = 5.0;
    vec3 rayDir = -lightDir;
    for(int i=1; i<=5; ++i) {
        vec3 p = pos + rayDir * stepSize * float(i);
        bool shadow = inShadow(p, rayDir, 1e10);
        if(!shadow) result += lightColor * 0.08;
    }
    return result;
}

void main() {
    // Temporal Jitter for Anti-Aliasing
    vec2 seed = gl_FragCoord.xy + vec2(uFrameIndex * 0.6180339887, uFrameIndex * 0.3247179572);
    vec2 jitter = vec2(hash(seed), hash(seed.yx)) - 0.5;
    vec2 uv = (gl_FragCoord.xy + jitter) / uResolution;
    
    vec3 rd = normalize(uCamForward + uCamRight * ((uv.x * 2.0 - 1.0) * uScale * uAspectRatio) + uCamUp * ((uv.y * 2.0 - 1.0) * uScale));
    vec3 ro = uCamPos;
    
    vec3 finalColor = vec3(0.0);
    vec3 mask = vec3(1.0);
    float firstHitT = -1.0;
    
    for (int bounce = 0; bounce < 4; ++bounce) {
        float minT = 1e10;
        vec3 hitPos, hitNormal, hitColor;
        float hitReflect = 0.0;
        bool hitWater = false;
        float t;
        
        for(int i=0; i<numSpheres; ++i) {
            if(intersectSphere(ro, rd, spheres[i].posRad.xyz, spheres[i].posRad.w, t)) {
                if(t < minT) { 
                    minT = t; 
                    hitPos = ro + rd*t; 
                    hitNormal = normalize(hitPos - spheres[i].posRad.xyz); 
                    hitColor = spheres[i].color.rgb; 
                    hitReflect = spheres[i].color.a;
                }
            }
        }
        for(int i=0; i<numBoxes; ++i) {
            vec3 bmin = boxes[i].pos.xyz - boxes[i].ext.xyz;
            vec3 bmax = boxes[i].pos.xyz + boxes[i].ext.xyz;
            if(intersectAABB(ro, rd, bmin, bmax, t)) {
                if(t < minT) { 
                    minT = t; 
                    hitPos = ro + rd*t; 
                    vec3 rel = hitPos - boxes[i].pos.xyz;
                    vec3 e = boxes[i].ext.xyz;
                    hitNormal = normalize(vec3(
                        (abs(rel.x - e.x) < 0.01 ? 1.0 : 0.0) - (abs(rel.x + e.x) < 0.01 ? 1.0 : 0.0),
                        (abs(rel.y - e.y) < 0.01 ? 1.0 : 0.0) - (abs(rel.y + e.y) < 0.01 ? 1.0 : 0.0),
                        (abs(rel.z - e.z) < 0.01 ? 1.0 : 0.0) - (abs(rel.z + e.z) < 0.01 ? 1.0 : 0.0)
                    ));
                    hitColor = boxes[i].color.rgb; 
                    hitReflect = boxes[i].color.a;
                }
            }
        }
        for(int i=0; i<numPlanes; ++i) {
            if(intersectPlane(ro, rd, planes[i].pos.xyz, planes[i].norm.xyz, t)) {
                if(t < minT) { minT = t; hitPos = ro + rd*t; hitNormal = planes[i].norm.xyz; hitColor = planes[i].color.rgb; hitReflect = planes[i].color.a; }
            }
        }
        if(hasWater == 1 && intersectWater(ro, rd, t, hitNormal)) {
            if(t < minT) { minT = t; hitPos = ro + rd*t; hitColor = vec3(0.1, 0.3, 0.8); hitWater = true; hitReflect = 0.02; }
        }
        
        if(minT < 1e10) {
            if (firstHitT < 0.0) firstHitT = minT;
            
            vec3 viewDir = -rd;
            vec3 baseColor = calcLighting(hitPos, hitNormal, hitColor, viewDir);
            
            for(int i=0; i<numLights; ++i) {
                if(int(lights[i].type_info.x) == 0) baseColor += calcGodrays(hitPos, lights[i].dir.xyz, lights[i].color.xyz);
            }
            
            float F;
            if (hitWater) {
                vec3 refl = reflect(rd, hitNormal);
                float sky = max(refl.y, 0.0);
                vec3 skyColor = mix(vec3(0.4, 0.6, 0.8), vec3(0.1, 0.2, 0.4), sky);
                
                F = 0.02 + 0.98 * pow(1.0 - max(dot(hitNormal, viewDir), 0.0), 5.0);
                baseColor = mix(baseColor, skyColor, F);
                hitReflect = F;
            } else {
                F = hitReflect + (1.0 - hitReflect) * pow(1.0 - max(dot(hitNormal, viewDir), 0.0), 5.0);
                hitReflect = F;
            }
            
            finalColor += baseColor * mask * (1.0 - F);
            
            if (F < 0.01) break;
            
            mask *= mix(vec3(1.0), hitColor, 0.5) * F; 
            
            ro = hitPos + hitNormal * 0.002;
            rd = reflect(rd, hitNormal);
        } else {
            float sky = max(rd.y, 0.0);
            vec3 skyColor = mix(vec3(0.7, 0.8, 0.9), vec3(0.2, 0.3, 0.5), sky);
            
            for(int i=0; i<numLights; ++i) {
                if(int(lights[i].type_info.x) == 0) skyColor += calcGodrays(ro, lights[i].dir.xyz, lights[i].color.xyz) * 2.0;
            }
            
            finalColor += skyColor * mask;
            break;
        }
    }
    
    float fogDist = (firstHitT > 0.0) ? firstHitT : 100.0;
    float fogFactor = 1.0 - exp(-fogDist * uFogDensity);
    finalColor = mix(finalColor, uFogColor, fogFactor);
    
    // Write depth based on first hit for OBJ compositing
    if (firstHitT > 0.0) {
        vec3 firstHitPos = uCamPos + rd * firstHitT; // Note: rd is the last bounce dir, need original rd
        // To fix this, we should store original rd. 
        // Actually, let's just use the original ray direction from camera
        vec3 origRd = normalize(uCamForward + uCamRight * ((uv.x * 2.0 - 1.0) * uScale * uAspectRatio) + uCamUp * ((uv.y * 2.0 - 1.0) * uScale));
        vec3 firstHitPosCorrect = uCamPos + origRd * firstHitT;
        mat4 viewProj = uProj * uView;
        vec4 clipPos = viewProj * vec4(firstHitPosCorrect, 1.0);
        gl_FragDepth = (clipPos.z / clipPos.w) * 0.5 + 0.5;
    } else {
        gl_FragDepth = 1.0;
    }
    
    // Accumulation Logic
    if (uReset == 1) {
        FragColor = vec4(finalColor, 1.0);
    } else {
        vec3 prevColor = texture(uPrevFrame, gl_FragCoord.xy / uResolution).rgb;
        FragColor = vec4(mix(finalColor, prevColor, 0.85), 1.0);
    }
}
)GLSL";

const char* blitVS = R"GLSL(
#version 410 core
layout (location = 0) in vec2 aPos;
out vec2 vUV;
void main() { vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
)GLSL";

const char* blitFS = R"GLSL(
#version 410 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main() { FragColor = vec4(texture(uTex, vUV).rgb, 1.0); }
)GLSL";

// ============================================================================
// GLOBALS & ENGINE STATE
// ============================================================================
struct alignas(16) RT_Sphere {
    float pos[4];
    float color[4];
};
struct alignas(16) RT_Box {
    float pos[4];
    float ext[4];
    float color[4];
};
struct alignas(16) RT_Plane {
    float pos[4];
    float norm[4];
    float color[4];
};
struct alignas(16) RT_Light {
    float type; float _p0[3]; 
    float pos[4];
    float dir[4];
    float color[4];
    float params[4]; 
    float u[4];
    float v[4];
};

RT_Sphere makeSphere(float x, float y, float z, float r, Vec3 col, float reflect) {
    RT_Sphere s = {};
    s.pos[0] = x; s.pos[1] = y; s.pos[2] = z; s.pos[3] = r;
    s.color[0] = col.x; s.color[1] = col.y; s.color[2] = col.z; s.color[3] = reflect;
    return s;
}

RT_Box makeBox(float x, float y, float z, Vec3 ext, Vec3 col, float reflect) {
    RT_Box b = {};
    b.pos[0] = x; b.pos[1] = y; b.pos[2] = z; b.pos[3] = 0.0f;
    b.ext[0] = ext.x; b.ext[1] = ext.y; b.ext[2] = ext.z; b.ext[3] = 0.0f;
    b.color[0] = col.x; b.color[1] = col.y; b.color[2] = col.z; b.color[3] = reflect;
    return b;
}

RT_Plane makePlane(float x, float y, float z, Vec3 norm, Vec3 col, float reflect) {
    RT_Plane p = {};
    p.pos[0] = x; p.pos[1] = y; p.pos[2] = z; p.pos[3] = 0.0f;
    p.norm[0] = norm.x; p.norm[1] = norm.y; p.norm[2] = norm.z; p.norm[3] = 0.0f;
    p.color[0] = col.x; p.color[1] = col.y; p.color[2] = col.z; p.color[3] = reflect;
    return p;
}

RT_Light makeLight(int type, Vec3 pos, Vec3 dir, Vec3 col, float intensity, float range, float cutoff, float outerCutoff, Vec3 u, Vec3 v) {
    RT_Light l = {};
    l.type = (float)type;
    l.pos[0] = pos.x; l.pos[1] = pos.y; l.pos[2] = pos.z; l.pos[3] = 0.0f;
    l.dir[0] = dir.x; l.dir[1] = dir.y; l.dir[2] = dir.z; l.dir[3] = 0.0f;
    l.color[0] = col.x; l.color[1] = col.y; l.color[2] = col.z; l.color[3] = 0.0f;
    l.params[0] = intensity; l.params[1] = range; l.params[2] = cutoff; l.params[3] = outerCutoff;
    l.u[0] = u.x; l.u[1] = u.y; l.u[2] = u.z; l.u[3] = 0.0f;
    l.v[0] = v.x; l.v[1] = v.y; l.v[2] = v.z; l.v[3] = 0.0f;
    return l;
}

int wind_width = 800;
int wind_height = 600;
Vec3 camPos(0, 5, -15);
float yaw = 0.0f, pitch = 0.0f;
bool cursorCaptured = true;
double lastX = 640, lastY = 360;
bool firstMouse = true;

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "Shader Error: " << infoLog << std::endl;
    }
    return shader;
}

// ============================================================================
// MAIN ENTRY
// ============================================================================
int main() {
    if (!glfwInit()) return -1;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    
    GLFWwindow* window = glfwCreateWindow(wind_width, wind_height, "Holodeck Raytracer w/ Accumulation", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    std::ofstream("cube.obj") << "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\n"
        "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
        "f 1 2 3\nf 1 3 4\nf 5 8 7\nf 5 7 6\n"
        "f 1 4 8\nf 1 8 5\nf 2 6 7\nf 2 7 3\n"
        "f 4 3 7\nf 4 7 8\nf 1 5 6\nf 1 6 2\n";

    GLuint rastProg = glCreateProgram();
    glAttachShader(rastProg, compileShader(GL_VERTEX_SHADER, rasterVS));
    glAttachShader(rastProg, compileShader(GL_FRAGMENT_SHADER, rasterFS));
    glLinkProgram(rastProg);

    GLuint rtProg = glCreateProgram();
    glAttachShader(rtProg, compileShader(GL_VERTEX_SHADER, rtVS));
    glAttachShader(rtProg, compileShader(GL_FRAGMENT_SHADER, rtFS));
    glLinkProgram(rtProg);

    GLuint blitProg = glCreateProgram();
    glAttachShader(blitProg, compileShader(GL_VERTEX_SHADER, blitVS));
    glAttachShader(blitProg, compileShader(GL_FRAGMENT_SHADER, blitFS));
    glLinkProgram(blitProg);

    float quadVertices[] = { -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO); glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    // Setup Ping-Pong FBOs for Accumulation
    GLuint fbo[2], tex[2], depth[2];
    for(int i=0; i<2; ++i) {
        glGenTextures(1, &tex[i]);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, wind_width, wind_height, 0, GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        glGenRenderbuffers(1, &depth[i]);
        glBindRenderbuffer(GL_RENDERBUFFER, depth[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, wind_width, wind_height);
        
        glGenFramebuffers(1, &fbo[i]);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[i], 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth[i]);
    }
    int readFBO = 0, writeFBO = 1;

    GLuint uboSpheres, uboBoxes, uboPlanes, uboLights;
    
    glGenBuffers(1, &uboSpheres); 
    glBindBuffer(GL_UNIFORM_BUFFER, uboSpheres);
    glBufferData(GL_UNIFORM_BUFFER, 256 * sizeof(RT_Sphere), NULL, GL_DYNAMIC_DRAW);
    
    glGenBuffers(1, &uboBoxes); 
    glBindBuffer(GL_UNIFORM_BUFFER, uboBoxes);
    glBufferData(GL_UNIFORM_BUFFER, 256 * sizeof(RT_Box), NULL, GL_DYNAMIC_DRAW);
    
    glGenBuffers(1, &uboPlanes); 
    glBindBuffer(GL_UNIFORM_BUFFER, uboPlanes);
    glBufferData(GL_UNIFORM_BUFFER, 256 * sizeof(RT_Plane), NULL, GL_DYNAMIC_DRAW);
    
    glGenBuffers(1, &uboLights); 
    glBindBuffer(GL_UNIFORM_BUFFER, uboLights);
    glBufferData(GL_UNIFORM_BUFFER, 32 * sizeof(RT_Light), NULL, GL_DYNAMIC_DRAW);

    GLuint blockIndex;
    blockIndex = glGetUniformBlockIndex(rtProg, "SphereBlock");
    glUniformBlockBinding(rtProg, blockIndex, 1);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, uboSpheres);

    blockIndex = glGetUniformBlockIndex(rtProg, "BoxBlock");
    glUniformBlockBinding(rtProg, blockIndex, 2);
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, uboBoxes);

    blockIndex = glGetUniformBlockIndex(rtProg, "PlaneBlock");
    glUniformBlockBinding(rtProg, blockIndex, 3);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, uboPlanes);

    blockIndex = glGetUniformBlockIndex(rtProg, "LightBlock");
    glUniformBlockBinding(rtProg, blockIndex, 4);
    glBindBufferBase(GL_UNIFORM_BUFFER, 4, uboLights);

    HolodeckGrid world(30.0f);
    std::vector<SceneNode*> allNodes;
    std::vector<RT_Light> lights;

    lights.push_back(makeLight(0, Vec3(0,0,0), Vec3(-0.5f, -1.0f, -0.5f), Vec3(1.0f, 0.9f, 0.8f), 1.5f, 0, 0, 0, Vec3(0,0,0), Vec3(0,0,0))); 
    lights.push_back(makeLight(1, Vec3(10, 5, 10), Vec3(0,0,0), Vec3(1.0f, 0.2f, 0.2f), 15.0f, 20.0f, 0, 0, Vec3(0,0,0), Vec3(0,0,0))); 
    lights.push_back(makeLight(2, Vec3(-10, 15, -10), Vec3(0,0,0), Vec3(0.2f, 1.0f, 0.2f), 30.0f, 0, 0.9f, 0.8f, Vec3(0,0,0), Vec3(0,0,0))); 
    lights.push_back(makeLight(3, Vec3(0, 15, 0), Vec3(0,0,0), Vec3(1.0f, 1.0f, 1.0f), 30.0f, 0, 0, 0, Vec3(5,0,0), Vec3(0,0,5))); 

    SceneNode* water = new SceneNode();
    water->type = SceneNode::Type::WATER;
    water->position = Vec3(0, -5, 0);
    water->radius = 1000.0f;
    allNodes.push_back(water);
    world.insert(water);

    SceneNode* ground = new SceneNode();
    ground->type = SceneNode::Type::PLANE;
    ground->position = Vec3(0, -6, 0);
    ground->normal = Vec3(0, 1, 0);
    ground->color = Vec3(0.3, 0.5, 0.3);
    ground->radius = 1000.0f;
    ground->reflectivity = 0.1f; 
    allNodes.push_back(ground);
    world.insert(ground);

    SceneNode* mirrorSphere = new SceneNode();
    mirrorSphere->type = SceneNode::Type::SPHERE;
    mirrorSphere->position = Vec3(5, 2, -5);
    mirrorSphere->radius = 3.0f;
    mirrorSphere->dimensions = Vec3(3,3,3);
    mirrorSphere->color = Vec3(0.95, 0.95, 0.95);
    mirrorSphere->reflectivity = 0.9f;
    allNodes.push_back(mirrorSphere);
    world.insert(mirrorSphere);

    for (int i = 0; i < 30; ++i) {
        SceneNode* node = new SceneNode();
        node->type = (i % 2 == 0) ? SceneNode::Type::SPHERE : SceneNode::Type::BOX;
        node->position = Vec3(rand()%60 - 30, rand()%10 - 2, rand()%60 - 30);
        node->radius = 1.0f + (rand()%10)/10.0f;
        node->dimensions = Vec3(node->radius, node->radius, node->radius);
        node->color = Vec3((rand()%10)/10.0f, (rand()%10)/10.0f, (rand()%10)/10.0f);
        node->reflectivity = (rand()%10) / 10.0f;
        if (i % 5 == 0) node->reflectivity = 0.8f; 
        allNodes.push_back(node);
        world.insert(node);
    }

    SceneNode* objNode = new SceneNode();
    objNode->type = SceneNode::Type::OBJ_MESH;
    objNode->position = Vec3(0, 0, 0);
    objNode->color = Vec3(0.8, 0.8, 0.2);
    MeshData mesh = loadOBJ("cube.obj");
    objNode->indexCount = mesh.indices.size();

    glGenVertexArrays(1, &objNode->VAO);
    GLuint vbo, ebo;
    glGenBuffers(1, &vbo); glGenBuffers(1, &ebo);
    glBindVertexArray(objNode->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(float), mesh.vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), mesh.indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    allNodes.push_back(objNode);
    world.insert(objNode);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    double lastFrameTime = glfwGetTime();
    Vec3 lastCamPos = camPos;
    float lastYaw = yaw, lastPitch = pitch;
    int frameIndex = 0;

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, GL_TRUE);

        if (cursorCaptured) {
            double mx, my; glfwGetCursorPos(window, &mx, &my);
            if (firstMouse) { lastX = mx; lastY = my; firstMouse = false; }
            yaw += (mx - lastX) * 0.002f; pitch += (lastY - my) * 0.002f;
            if (pitch > 1.5f) pitch = 1.5f; if (pitch < -1.5f) pitch = -1.5f;
            lastX = mx; lastY = my;
        }
        
        Vec3 forward(cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw));
        forward = forward.normalize();
        Vec3 right = forward.cross(Vec3(0,1,0)).normalize();
        Vec3 up = right.cross(forward).normalize();

        float speed = 10.0f * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camPos += forward * speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camPos -= forward * speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camPos -= right * speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camPos += right * speed;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camPos += up * speed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) camPos -= up * speed;

        // Detect camera movement to reset accumulation
        bool cameraMoved = (camPos - lastCamPos).length() > 0.001f || abs(yaw - lastYaw) > 0.001f || abs(pitch - lastPitch) > 0.001f;
        if (cameraMoved) {
            lastCamPos = camPos;
            lastYaw = yaw;
            lastPitch = pitch;
            frameIndex = 0;
        } else {
            frameIndex++;
        }
        int resetAccum = cameraMoved ? 1 : 0;

        std::vector<SceneNode*> visibleNodes;
        world.collect(camPos, 150.0f, visibleNodes);

        std::vector<RT_Sphere> vSpheres;
        std::vector<RT_Box> vBoxes;
        std::vector<RT_Plane> vPlanes;
        bool hasWater = false;

        for (auto n : visibleNodes) {
            if (n->type == SceneNode::Type::SPHERE) {
                vSpheres.push_back(makeSphere(n->position.x, n->position.y, n->position.z, n->radius, n->color, n->reflectivity));
            } else if (n->type == SceneNode::Type::BOX) {
                vBoxes.push_back(makeBox(n->position.x, n->position.y, n->position.z, n->dimensions * 0.5f, n->color, n->reflectivity));
            } else if (n->type == SceneNode::Type::PLANE) {
                vPlanes.push_back(makePlane(n->position.x, n->position.y, n->position.z, n->normal, n->color, n->reflectivity));
            } else if (n->type == SceneNode::Type::WATER) {
                hasWater = true;
            }
        }

        glBindBuffer(GL_UNIFORM_BUFFER, uboSpheres);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, vSpheres.size() * sizeof(RT_Sphere), vSpheres.data());
        glBindBuffer(GL_UNIFORM_BUFFER, uboBoxes);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, vBoxes.size() * sizeof(RT_Box), vBoxes.data());
        glBindBuffer(GL_UNIFORM_BUFFER, uboPlanes);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, vPlanes.size() * sizeof(RT_Plane), vPlanes.data());
        glBindBuffer(GL_UNIFORM_BUFFER, uboLights);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, lights.size() * sizeof(RT_Light), lights.data());

        Mat4 view = lookAt(camPos, camPos + forward, Vec3(0,1,0));
        Mat4 proj = perspective(1.2f, (float)wind_width/wind_height, 0.1f, 500.0f);

        // ==========================================
        // PASS 1: Rasterization (OBJ Models) into Write FBO
        // ==========================================
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[writeFBO]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(rastProg);
        glUniformMatrix4fv(glGetUniformLocation(rastProg, "uView"), 1, GL_FALSE, view.ptr());
        glUniformMatrix4fv(glGetUniformLocation(rastProg, "uProj"), 1, GL_FALSE, proj.ptr());
        
        for (auto n : visibleNodes) {
            if (n->type == SceneNode::Type::OBJ_MESH) {
                Mat4 model;
                model.m[12] = n->position.x; model.m[13] = n->position.y; model.m[14] = n->position.z;
                model.m[0] = n->dimensions.x; model.m[5] = n->dimensions.y; model.m[10] = n->dimensions.z;
                glUniformMatrix4fv(glGetUniformLocation(rastProg, "uModel"), 1, GL_FALSE, model.ptr());
                glUniform3f(glGetUniformLocation(rastProg, "uColor"), n->color.x, n->color.y, n->color.z);
                glBindVertexArray(n->VAO);
                glDrawElements(GL_TRIANGLES, n->indexCount, GL_UNSIGNED_INT, 0);
            }
        }

        // ==========================================
        // PASS 2: Raytracing with Accumulation into Write FBO
        // ==========================================
        glUseProgram(rtProg);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex[readFBO]);
        glUniform1i(glGetUniformLocation(rtProg, "uPrevFrame"), 0);
        glUniform1f(glGetUniformLocation(rtProg, "uFrameIndex"), (float)frameIndex);
        glUniform1i(glGetUniformLocation(rtProg, "uReset"), resetAccum);
        glUniform2f(glGetUniformLocation(rtProg, "uResolution"), (float)wind_width, (float)wind_height);
        glUniform1f(glGetUniformLocation(rtProg, "uAspectRatio"), (float)wind_width / (float)wind_height);
        
        glUniform3f(glGetUniformLocation(rtProg, "uCamPos"), camPos.x, camPos.y, camPos.z);
        glUniform3f(glGetUniformLocation(rtProg, "uCamForward"), forward.x, forward.y, forward.z);
        glUniform3f(glGetUniformLocation(rtProg, "uCamRight"), right.x, right.y, right.z);
        glUniform3f(glGetUniformLocation(rtProg, "uCamUp"), up.x, up.y, up.z);
        glUniform1f(glGetUniformLocation(rtProg, "uScale"), tanf(1.2f / 2.0f));
        glUniform1f(glGetUniformLocation(rtProg, "uTime"), (float)glfwGetTime());
        glUniform3f(glGetUniformLocation(rtProg, "uFogColor"), 0.5f, 0.6f, 0.7f);
        glUniform1f(glGetUniformLocation(rtProg, "uFogDensity"), 0.008f);
        glUniformMatrix4fv(glGetUniformLocation(rtProg, "uView"), 1, GL_FALSE, view.ptr());
        glUniformMatrix4fv(glGetUniformLocation(rtProg, "uProj"), 1, GL_FALSE, proj.ptr());
        
        glUniform1i(glGetUniformLocation(rtProg, "numSpheres"), vSpheres.size());
        glUniform1i(glGetUniformLocation(rtProg, "numBoxes"), vBoxes.size());
        glUniform1i(glGetUniformLocation(rtProg, "numPlanes"), vPlanes.size());
        glUniform1i(glGetUniformLocation(rtProg, "numLights"), lights.size());
        glUniform1i(glGetUniformLocation(rtProg, "hasWater"), hasWater ? 1 : 0);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        // Swap FBOs
        std::swap(readFBO, writeFBO);

        // ==========================================
        // PASS 3: Blit Accumulated Result to Screen
        // ==========================================
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(blitProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex[readFBO]);
        glUniform1i(glGetUniformLocation(blitProg, "uTex"), 0);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
        usleep(50000);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}