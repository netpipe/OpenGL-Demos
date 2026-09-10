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
    Vec3 operator/(float s) const { return Vec3(x/s, y/s, z/s); } // FIX: Added division
    Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
    Vec3& operator-=(const Vec3& b) { x-=b.x; y-=b.y; z-=b.z; return *this; } // FIX: Added subtraction assignment
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
        nw = std::make_unique<QuadtreeNode>(AABB{bounds.min, c});
        ne = std::make_unique<QuadtreeNode>(AABB{Vec3(c.x, bounds.min.y, bounds.min.z), Vec3(bounds.max.x, c.y, c.z)});
        sw = std::make_unique<QuadtreeNode>(AABB{Vec3(bounds.min.x, bounds.min.y, c.z), Vec3(c.x, c.y, bounds.max.z)});
        se = std::make_unique<QuadtreeNode>(AABB{c, bounds.max});
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
            chunks[key] = std::make_unique<Chunk>(bounds);
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
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uModel, uView, uProj;
void main() { gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); }
)GLSL";

const char* rasterFS = R"GLSL(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() { FragColor = vec4(uColor, 1.0); }
)GLSL";

const char* rtVS = R"GLSL(
#version 430 core
layout (location = 0) in vec2 aPos;
out vec3 vViewDir;
uniform vec3 uCamForward, uCamRight, uCamUp;
uniform float uScale;
void main() {
    // Set Z to 0.0 (Near Plane) to ensure Early Z always passes, letting gl_FragDepth handle depth testing.
    gl_Position = vec4(aPos, 0.0, 1.0); 
    vViewDir = uCamForward + uCamRight * (aPos.x * uScale) + uCamUp * (aPos.y * uScale);
}
)GLSL";

const char* rtFS = R"GLSL(
#version 430 core
in vec3 vViewDir;
out vec4 FragColor;

uniform vec3 uCamPos;
uniform mat4 uView, uProj;
uniform float uTime;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform int hasWater;

struct Light {
    int type; // 0: Sun, 1: Point, 2: Spot, 3: Area
    vec3 pos;
    vec3 dir;
    vec3 color;
    float intensity, range, cutoff, outerCutoff;
    vec3 u, v;
};

struct Sphere { vec4 posRad; vec4 color; };
struct Box { vec4 pos; vec4 ext; vec4 color; };
struct Plane { vec4 pos; vec4 norm; vec4 color; };

layout(std430, binding = 1) buffer SphereBuffer { Sphere spheres[]; };
layout(std430, binding = 2) buffer BoxBuffer { Box boxes[]; };
layout(std430, binding = 3) buffer PlaneBuffer { Plane planes[]; };
layout(std430, binding = 4) buffer LightBuffer { Light lights[]; };

uniform int numSpheres, numBoxes, numPlanes, numLights;

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

vec3 calcLighting(vec3 pos, vec3 normal, vec3 albedo, vec3 viewDir) {
    vec3 result = vec3(0.05) * albedo; 
    for(int i=0; i<numLights; ++i) {
        if (lights[i].type == 0) { // Sun
            vec3 l = normalize(-lights[i].dir);
            float diff = max(dot(normal, l), 0.0);
            vec3 h = normalize(l + viewDir);
            float spec = pow(max(dot(normal, h), 0.0), 32.0);
            result += (diff * albedo + spec * 0.5) * lights[i].color * lights[i].intensity;
        } 
        else if (lights[i].type == 1) { // Point
            vec3 l = lights[i].pos - pos;
            float dist = length(l);
            l /= dist;
            float atten = 1.0 / (1.0 + 0.09*dist + 0.032*dist*dist);
            float diff = max(dot(normal, l), 0.0);
            vec3 h = normalize(l + viewDir);
            float spec = pow(max(dot(normal, h), 0.0), 32.0);
            result += (diff * albedo + spec * 0.5) * lights[i].color * lights[i].intensity * atten;
        }
        else if (lights[i].type == 2) { // Spot
            vec3 l = lights[i].pos - pos;
            float dist = length(l);
            l /= dist;
            float theta = dot(l, normalize(-lights[i].dir));
            float epsilon = lights[i].cutoff - lights[i].outerCutoff;
            float intensity = clamp((theta - lights[i].outerCutoff) / epsilon, 0.0, 1.0);
            float diff = max(dot(normal, l), 0.0);
            vec3 h = normalize(l + viewDir);
            float spec = pow(max(dot(normal, h), 0.0), 32.0);
            result += (diff * albedo + spec * 0.5) * lights[i].color * lights[i].intensity * intensity / (dist*dist);
        }
        else if (lights[i].type == 3) { // Area (Sampled Grid)
            vec3 sum = vec3(0.0);
            int samples = 4;
            for(int x=0; x<samples; ++x) {
                for(int y=0; y<samples; ++y) {
                    vec3 samplePos = lights[i].pos + lights[i].u * ((float(x)+0.5)/samples - 0.5) + lights[i].v * ((float(y)+0.5)/samples - 0.5);
                    vec3 l = samplePos - pos;
                    float dist = length(l);
                    l /= dist;
                    float diff = max(dot(normal, l), 0.0);
                    sum += diff * albedo * lights[i].color * lights[i].intensity / (dist*dist);
                }
            }
            result += sum / (samples*samples);
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
        bool inShadow = false;
        float t;
        for(int j=0; j<numSpheres; ++j) {
            if(intersectSphere(p, rayDir, spheres[j].posRad.xyz, spheres[j].posRad.w, t)) { inShadow = true; break; }
        }
        for(int j=0; j<numBoxes; ++j) {
            vec3 bmin = boxes[j].pos.xyz - boxes[j].ext.xyz;
            vec3 bmax = boxes[j].pos.xyz + boxes[j].ext.xyz;
            if(intersectAABB(p, rayDir, bmin, bmax, t)) { inShadow = true; break; }
        }
        if(!inShadow) result += lightColor * 0.08;
    }
    return result;
}

void main() {
    vec3 ro = uCamPos;
    vec3 rd = normalize(vViewDir);

    float minT = 1e10;
    vec3 hitPos, hitNormal, hitColor;
    bool hitWater = false;
    float t;

    for(int i=0; i<numSpheres; ++i) {
        if(intersectSphere(ro, rd, spheres[i].posRad.xyz, spheres[i].posRad.w, t)) {
            if(t < minT) { minT = t; hitPos = ro + rd*t; hitNormal = normalize(hitPos - spheres[i].posRad.xyz); hitColor = spheres[i].color.rgb; }
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
            }
        }
    }
    for(int i=0; i<numPlanes; ++i) {
        if(intersectPlane(ro, rd, planes[i].pos.xyz, planes[i].norm.xyz, t)) {
            if(t < minT) { minT = t; hitPos = ro + rd*t; hitNormal = planes[i].norm.xyz; hitColor = planes[i].color.rgb; }
        }
    }
    if(hasWater == 1 && intersectWater(ro, rd, t, hitNormal)) {
        if(t < minT) { minT = t; hitPos = ro + rd*t; hitColor = vec3(0.1, 0.3, 0.8); hitWater = true; }
    }

    if(minT < 1e10) {
        vec3 viewDir = -rd;
        vec3 color = calcLighting(hitPos, hitNormal, hitColor, viewDir);
        
        for(int i=0; i<numLights; ++i) {
            if(lights[i].type == 0) color += calcGodrays(hitPos, lights[i].dir, lights[i].color);
        }
        
        if(hitWater) {
            vec3 refl = reflect(rd, hitNormal);
            float sky = max(refl.y, 0.0);
            vec3 skyColor = mix(vec3(0.4, 0.6, 0.8), vec3(0.1, 0.2, 0.4), sky);
            color = mix(color, skyColor, 0.4); 
        }

        float fogFactor = 1.0 - exp(-minT * uFogDensity);
        color = mix(color, uFogColor, fogFactor);

        mat4 viewProj = uProj * uView;
        vec4 clipPos = viewProj * vec4(hitPos, 1.0);
        gl_FragDepth = (clipPos.z / clipPos.w) * 0.5 + 0.5;

        FragColor = vec4(color, 1.0);
    } else {
        float sky = max(rd.y, 0.0);
        vec3 skyColor = mix(vec3(0.7, 0.8, 0.9), vec3(0.2, 0.3, 0.5), sky);
        float fogFactor = 1.0 - exp(-100.0 * uFogDensity);
        skyColor = mix(skyColor, uFogColor, fogFactor);
        
        for(int i=0; i<numLights; ++i) {
            if(lights[i].type == 0) skyColor += calcGodrays(ro, lights[i].dir, lights[i].color) * 2.0;
        }
        
        gl_FragDepth = 1.0;
        FragColor = vec4(skyColor, 1.0);
    }
}
)GLSL";

// ============================================================================
// GLOBALS & ENGINE STATE
// ============================================================================
struct alignas(16) RT_Sphere {
    float pos[3], rad;
    float color[3], _pad;
};
struct alignas(16) RT_Box {
    float pos[3], _p1;
    float ext[3], _p2;
    float color[3], _p3;
};
struct alignas(16) RT_Plane {
    float pos[3], _p1;
    float norm[3], _p2;
    float color[3], _p3;
};
struct alignas(16) RT_Light {
    int type; float _p0[3]; // Pad to 16 bytes
    float pos[3]; float _p1;
    float dir[3]; float _p2;
    float color[3]; float _p3;
    float intensity, range, cutoff, outerCutoff;
    float u[3], _p4;
    float v[3], _p5;
};

// FIX: Factory functions to fix aggregate initialization issues in Apple Clang / Xcode
RT_Sphere makeSphere(float x, float y, float z, float r, Vec3 col) {
    RT_Sphere s = {};
    s.pos[0] = x; s.pos[1] = y; s.pos[2] = z; s.rad = r;
    s.color[0] = col.x; s.color[1] = col.y; s.color[2] = col.z;
    return s;
}

RT_Box makeBox(float x, float y, float z, Vec3 ext, Vec3 col) {
    RT_Box b = {};
    b.pos[0] = x; b.pos[1] = y; b.pos[2] = z;
    b.ext[0] = ext.x; b.ext[1] = ext.y; b.ext[2] = ext.z;
    b.color[0] = col.x; b.color[1] = col.y; b.color[2] = col.z;
    return b;
}

RT_Plane makePlane(float x, float y, float z, Vec3 norm, Vec3 col) {
    RT_Plane p = {};
    p.pos[0] = x; p.pos[1] = y; p.pos[2] = z;
    p.norm[0] = norm.x; p.norm[1] = norm.y; p.norm[2] = norm.z;
    p.color[0] = col.x; p.color[1] = col.y; p.color[2] = col.z;
    return p;
}

RT_Light makeLight(int type, Vec3 pos, Vec3 dir, Vec3 col, float intensity, float range, float cutoff, float outerCutoff, Vec3 u, Vec3 v) {
    RT_Light l = {};
    l.type = type;
    l.pos[0] = pos.x; l.pos[1] = pos.y; l.pos[2] = pos.z;
    l.dir[0] = dir.x; l.dir[1] = dir.y; l.dir[2] = dir.z;
    l.color[0] = col.x; l.color[1] = col.y; l.color[2] = col.z;
    l.intensity = intensity; l.range = range; l.cutoff = cutoff; l.outerCutoff = outerCutoff;
    l.u[0] = u.x; l.u[1] = u.y; l.u[2] = u.z;
    l.v[0] = v.x; l.v[1] = v.y; l.v[2] = v.z;
    return l;
}

int wind_width = 1280;
int wind_height = 720;
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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(wind_width, wind_height, "Holodeck Raytraced Scenegraph", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    // Generate Dummy OBJ Cube File
    std::ofstream("cube.obj") << "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\n"
        "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
        "f 1 2 3\nf 1 3 4\nf 5 8 7\nf 5 7 6\n"
        "f 1 4 8\nf 1 8 5\nf 2 6 7\nf 2 7 3\n"
        "f 4 3 7\nf 4 7 8\nf 1 5 6\nf 1 6 2\n";

    // Setup Shaders
    GLuint rastProg = glCreateProgram();
    glAttachShader(rastProg, compileShader(GL_VERTEX_SHADER, rasterVS));
    glAttachShader(rastProg, compileShader(GL_FRAGMENT_SHADER, rasterFS));
    glLinkProgram(rastProg);

    GLuint rtProg = glCreateProgram();
    glAttachShader(rtProg, compileShader(GL_VERTEX_SHADER, rtVS));
    glAttachShader(rtProg, compileShader(GL_FRAGMENT_SHADER, rtFS));
    glLinkProgram(rtProg);

    // Setup Fullscreen Quad for Raytracer
    float quadVertices[] = { -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO); glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    // Create SSBOs
    GLuint ssboSpheres, ssboBoxes, ssboPlanes, ssboLights;
    glGenBuffers(1, &ssboSpheres); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSpheres);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256 * sizeof(RT_Sphere), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboSpheres);

    glGenBuffers(1, &ssboBoxes); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboBoxes);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256 * sizeof(RT_Box), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboBoxes);

    glGenBuffers(1, &ssboPlanes); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboPlanes);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256 * sizeof(RT_Plane), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboPlanes);

    glGenBuffers(1, &ssboLights); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLights);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 32 * sizeof(RT_Light), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboLights);

    // Initialize Scene Graph / World
    HolodeckGrid world(30.0f);
    std::vector<SceneNode*> allNodes;
    std::vector<RT_Light> lights;

    // Add Lights
    lights.push_back(makeLight(0, Vec3(0,0,0), Vec3(-0.5f, -1.0f, -0.5f), Vec3(1.0f, 0.9f, 0.8f), 1.5f, 0, 0, 0, Vec3(0,0,0), Vec3(0,0,0))); // Sun
    lights.push_back(makeLight(1, Vec3(10, 5, 10), Vec3(0,0,0), Vec3(1.0f, 0.2f, 0.2f), 15.0f, 20.0f, 0, 0, Vec3(0,0,0), Vec3(0,0,0))); // Point
    lights.push_back(makeLight(2, Vec3(-10, 15, -10), Vec3(0,0,0), Vec3(0.2f, 1.0f, 0.2f), 30.0f, 0, 0.9f, 0.8f, Vec3(0,0,0), Vec3(0,0,0))); // Spot
    lights.push_back(makeLight(3, Vec3(0, 15, 0), Vec3(0,0,0), Vec3(1.0f, 1.0f, 1.0f), 30.0f, 0, 0, 0, Vec3(5,0,0), Vec3(0,0,5))); // Area

    // Add Water
    SceneNode* water = new SceneNode();
    water->type = SceneNode::Type::WATER;
    water->position = Vec3(0, -5, 0);
    water->radius = 1000.0f;
    allNodes.push_back(water);
    world.insert(water);

    // Add Ground Plane
    SceneNode* ground = new SceneNode();
    ground->type = SceneNode::Type::PLANE;
    ground->position = Vec3(0, -6, 0);
    ground->normal = Vec3(0, 1, 0);
    ground->color = Vec3(0.3, 0.5, 0.3);
    ground->radius = 1000.0f;
    allNodes.push_back(ground);
    world.insert(ground);

    // Add Random Primitive Objects
    for (int i = 0; i < 30; ++i) {
        SceneNode* node = new SceneNode();
        node->type = (i % 2 == 0) ? SceneNode::Type::SPHERE : SceneNode::Type::BOX;
        node->position = Vec3(rand()%60 - 30, rand()%10 - 2, rand()%60 - 30);
        node->radius = 1.0f + (rand()%10)/10.0f;
        node->dimensions = Vec3(node->radius, node->radius, node->radius);
        node->color = Vec3((rand()%10)/10.0f, (rand()%10)/10.0f, (rand()%10)/10.0f);
        allNodes.push_back(node);
        world.insert(node);
    }

    // Add OBJ Mesh Node
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
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, GL_TRUE);

        // Camera Controls
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

        // Cull Nodes
        std::vector<SceneNode*> visibleNodes;
        world.collect(camPos, 150.0f, visibleNodes);

        std::vector<RT_Sphere> vSpheres;
        std::vector<RT_Box> vBoxes;
        std::vector<RT_Plane> vPlanes;
        bool hasWater = false;

        for (auto n : visibleNodes) {
            if (n->type == SceneNode::Type::SPHERE) {
                vSpheres.push_back(makeSphere(n->position.x, n->position.y, n->position.z, n->radius, n->color));
            } else if (n->type == SceneNode::Type::BOX) {
                vBoxes.push_back(makeBox(n->position.x, n->position.y, n->position.z, n->dimensions * 0.5f, n->color));
            } else if (n->type == SceneNode::Type::PLANE) {
                vPlanes.push_back(makePlane(n->position.x, n->position.y, n->position.z, n->normal, n->color));
            } else if (n->type == SceneNode::Type::WATER) {
                hasWater = true;
            }
        }

        // Update SSBOs
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSpheres);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vSpheres.size() * sizeof(RT_Sphere), vSpheres.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboBoxes);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vBoxes.size() * sizeof(RT_Box), vBoxes.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboPlanes);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vPlanes.size() * sizeof(RT_Plane), vPlanes.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLights);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, lights.size() * sizeof(RT_Light), lights.data());

        Mat4 view = lookAt(camPos, camPos + forward, Vec3(0,1,0));
        Mat4 proj = perspective(1.2f, (float)wind_width/wind_height, 0.1f, 500.0f);

        // ==========================================
        // PASS 1: Rasterization (OBJ Models)
        // ==========================================
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
        // PASS 2: Raytracing
        // ==========================================
        glUseProgram(rtProg);
        glUniform3f(glGetUniformLocation(rtProg, "uCamPos"), camPos.x, camPos.y, camPos.z);
        glUniform3f(glGetUniformLocation(rtProg, "uCamForward"), forward.x, forward.y, forward.z);
        glUniform3f(glGetUniformLocation(rtProg, "uCamRight"), right.x, right.y, right.z);
        glUniform3f(glGetUniformLocation(rtProg, "uCamUp"), up.x, up.y, up.z);
        glUniform1f(glGetUniformLocation(rtProg, "uScale"), tanf(1.2f / 2.0f));
        glUniform1f(glGetUniformLocation(rtProg, "uTime"), (float)glfwGetTime());
        glUniform3f(glGetUniformLocation(rtProg, "uFogColor"), 0.5f, 0.6f, 0.7f);
        glUniform1f(glGetUniformLocation(rtProg, "uFogDensity"), 0.015f);
        glUniformMatrix4fv(glGetUniformLocation(rtProg, "uView"), 1, GL_FALSE, view.ptr());
        glUniformMatrix4fv(glGetUniformLocation(rtProg, "uProj"), 1, GL_FALSE, proj.ptr());
        
        glUniform1i(glGetUniformLocation(rtProg, "numSpheres"), vSpheres.size());
        glUniform1i(glGetUniformLocation(rtProg, "numBoxes"), vBoxes.size());
        glUniform1i(glGetUniformLocation(rtProg, "numPlanes"), vPlanes.size());
        glUniform1i(glGetUniformLocation(rtProg, "numLights"), lights.size());
        glUniform1i(glGetUniformLocation(rtProg, "hasWater"), hasWater ? 1 : 0);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}