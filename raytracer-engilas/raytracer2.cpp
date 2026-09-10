// Raytracer (engilas) — single-file product
// Based on engilas/raytracing-opengl
//
// Controls: mouse look, WASD, Space/Ctrl up-down, Shift boost, Alt slow, Tab free cursor, Esc quit
// Linux:   g++ raytracer1.cpp -o raytracer -lglfw -lGLEW -lGL -lm
// Windows: see CMakeLists.txt (no pthread)

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <thread>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <unistd.h>

#define PI_F 3.14159265358979f

// ---------------------------------------------------------
// Math Structs & Helpers (replacing GLM)
// ---------------------------------------------------------
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3& b) const { return Vec3(x+b.x, y+b.y, z+b.z); }
    Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x, y-b.y, z-b.z); }
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s, y/s, z/s); } // FIXED: Added division operator
    Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
    Vec3& operator-=(const Vec3& b) { x-=b.x; y-=b.y; z-=b.z; return *this; }
    Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
};

struct Quat {
    float x, y, z, w;
    Quat() : x(0), y(0), z(0), w(1) {}
    Quat(float X, float Y, float Z, float W) : x(X), y(Y), z(Z), w(W) {}
};

Vec3 normalize(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-8f) return Vec3(0,0,0);
    return v / len;
}

Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

float radians(float deg) {
    return deg * PI_F / 180.0f;
}

Quat angleAxis(float angle, Vec3 axis) {
    float halfAngle = angle * 0.5f;
    float s = sinf(halfAngle);
    return Quat(axis.x * s, axis.y * s, axis.z * s, cosf(halfAngle));
}

Quat quatMult(Quat q1, Quat q2) {
    return Quat(
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    );
}

Quat quatFromEuler(float pitch, float yaw) {
    Quat qPitch = angleAxis(pitch, Vec3(1,0,0));
    Quat qYaw = angleAxis(yaw, Vec3(0,1,0));
    return quatMult(qYaw, qPitch);
}

Quat quatConj(Quat q) {
    return Quat(-q.x, -q.y, -q.z, q.w);
}

// Rotate a vector by a unit quaternion (same convention as the fragment shader).
Vec3 rotate(Quat q, Vec3 v) {
    Quat qv(v.x, v.y, v.z, 0.0f);
    Quat r = quatMult(quatMult(q, qv), quatConj(q));
    return Vec3(r.x, r.y, r.z);
}

// ---------------------------------------------------------
// Scene Structs (std140 matching the fragment shader)
// ---------------------------------------------------------
#pragma pack(push, 1) 

struct rt_material {
    float color[3]; float __p1;
    float absorb[3]; float diffuse;
    float reflect; float refract; int specular; float kd;
    float ks; float __padding[3];
};

struct rt_sphere {
    rt_material material;
    float obj[4];
    float quat_rotation[4];
    int textureNum; int hollow; float __padding[2];
};

struct rt_plane {
    rt_material material;
    float pos[3]; float __p1;
    float normal[3]; float __p2;
};

struct rt_box {
    rt_material mat;
    float quat_rotation[4];
    float pos[3]; float __p1;
    float form[3]; int textureNum;
};

struct rt_torus {
    rt_material mat;
    float quat_rotation[4];
    float pos[3]; float __p1;
    float form[2]; float __p2[2];
};

// FIXED: Added missing rt_ring struct
struct rt_ring {
    rt_material mat;
    float quat_rotation[4];
    float pos[3]; int textureNum;
    float r1, r2;
    float __p2[2];
};

struct rt_surface {
    rt_material mat;
    float quat_rotation[4];
    float xMin, yMin, zMin, __p0;
    float xMax, yMax, zMax, __p1;
    float pos[3]; float a;
    float b, c, d, e;
    float f; float __padding[3];
};

struct rt_light_direct {
    float direction[3]; float __p1;
    float color[3]; float intensity;
};

struct rt_light_point {
    float pos[4];
    float color[3]; float intensity;
    float linear_k; float quadratic_k; float __padding[2];
};

struct rt_scene {
    float quat_camera_rotation[4];
    float camera_pos[3]; float __p1;
    float bg_color[3]; int canvas_width;
    int canvas_height; int reflect_depth; float __padding[2];
};

#pragma pack(pop)

// ---------------------------------------------------------
// Helper functions to build the scene
// ---------------------------------------------------------
rt_material create_material(float r, float g, float b, int specular, float reflect, float refract, float diffuse) {
    rt_material m = {};
    m.color[0] = r; m.color[1] = g; m.color[2] = b;
    m.specular = specular;
    m.reflect = reflect;
    m.refract = refract;
    m.diffuse = diffuse;
    m.kd = 1.0f; m.ks = 1.0f;
    return m;
}

rt_sphere create_sphere(float x, float y, float z, float radius, rt_material mat, bool hollow = false) {
    rt_sphere s = {};
    s.obj[0] = x; s.obj[1] = y; s.obj[2] = z; s.obj[3] = radius;
    s.quat_rotation[3] = 1.0f; 
    s.material = mat;
    s.hollow = hollow ? 1 : 0;
    return s;
}

rt_plane create_plane(float x, float y, float z, float nx, float ny, float nz, rt_material mat) {
    rt_plane p = {};
    p.pos[0] = x; p.pos[1] = y; p.pos[2] = z;
    Vec3 n = normalize(Vec3(nx, ny, nz));
    p.normal[0] = n.x; p.normal[1] = n.y; p.normal[2] = n.z;
    p.material = mat;
    return p;
}

rt_box create_box(float x, float y, float z, float w, float h, float d, rt_material mat) {
    rt_box b = {};
    b.pos[0] = x; b.pos[1] = y; b.pos[2] = z;
    b.form[0] = w; b.form[1] = h; b.form[2] = d;
    b.mat = mat;
    b.quat_rotation[3] = 1.0f;
    return b;
}

rt_light_point create_point_light(float x, float y, float z, float r, float g, float b, float intensity) {
    rt_light_point l = {};
    l.pos[0] = x; l.pos[1] = y; l.pos[2] = z; l.pos[3] = 0.2f;
    l.color[0] = r; l.color[1] = g; l.color[2] = b;
    l.intensity = intensity;
    l.linear_k = 0.09f; l.quadratic_k = 0.032f;
    return l;
}

rt_light_direct create_direct_light(float dx, float dy, float dz, float r, float g, float b, float intensity) {
    rt_light_direct l = {};
    l.direction[0] = dx; l.direction[1] = dy; l.direction[2] = dz;
    l.color[0] = r; l.color[1] = g; l.color[2] = b;
    l.intensity = intensity;
    return l;
}

// ---------------------------------------------------------
// GLSL Shaders (Optimized)
// ---------------------------------------------------------
const char* vertexShaderSrc = R"GLSL(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;
out vec2 v_texCoord;
void main() {
    v_texCoord = aTexCoords;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

const char* fragmentShaderSrc = R"GLSL(
#version 330 core

#define FLT_MIN 1.175494351e-38
#define FLT_MAX 3.402823466e+38
#define PI_F 3.14159265358979f

#define TYPE_SPHERE 0
#define TYPE_PLANE 1
#define TYPE_SURFACE 2
#define TYPE_BOX 3
#define TYPE_TORUS 4
#define TYPE_RING 5
#define TYPE_POINT_LIGHT 6

#define SHADOW_ENABLED 1
#define TOTAL_INTERNAL_REFLECTION 1
#define DO_FRESNEL 1
#define PLANE_ONESIDE 1
#define REFLECT_REDUCE_ITERATION 1

struct rt_material { vec3 color; vec3 absorb; float diffuse; float reflection; float refraction; int specular; float kd; float ks; };
struct rt_sphere { rt_material mat; vec4 obj; vec4 quat_rotation; int textureNum; bool hollow; };
struct rt_plane { rt_material mat; vec3 pos; vec3 normal; };
struct rt_box { rt_material mat; vec4 quat_rotation; vec3 pos; vec3 form; int textureNum; };
struct rt_surface { rt_material mat; vec4 quat_rotation; vec3 v_min; vec3 v_max; vec3 pos; float a; float b; float c; float d; float e; float f; };
struct rt_torus { rt_material mat; vec4 quat_rotation; vec3 pos; vec2 form; };
struct rt_ring { rt_material mat; vec4 quat_rotation; vec3 pos; int textureNum; float r1; float r2; }; // FIXED: Added rt_ring declaration
struct rt_light_direct { vec3 direction; vec3 color; float intensity; };
struct rt_light_point { vec4 pos; vec3 color; float intensity; float linear_k; float quadratic_k; };
struct rt_scene { vec4 quat_camera_rotation; vec3 camera_pos; vec3 bg_color; int canvas_width; int canvas_height; int reflect_depth; };
struct hit_record { rt_material mat; vec3 normal; float bias_mult; float alpha; };

#define SPHERE_SIZE 3
#define PLANE_SIZE 1
#define SURFACE_SIZE 0
#define BOX_SIZE 2
#define TORUS_SIZE 0
#define RING_SIZE 0
#define LIGHT_DIRECT_SIZE 1
#define LIGHT_POINT_SIZE 1
#define AMBIENT_COLOR vec3(0.05, 0.05, 0.05)
#define SHADOW_AMBIENT vec3(0.2, 0.2, 0.2)
#define ITERATIONS 5

out vec4 FragColor;
const float maxDist = 1000000.0;
vec3 opt_normal;

layout( std140 ) uniform scene_buf { rt_scene scene; };
layout( std140 ) uniform spheres_buf { rt_sphere spheres[SPHERE_SIZE]; };
layout( std140 ) uniform planes_buf { rt_plane planes[1]; };
layout( std140 ) uniform surfaces_buf { rt_surface surfaces[1]; };
layout( std140 ) uniform boxes_buf { rt_box boxes[BOX_SIZE]; };
layout( std140 ) uniform toruses_buf { rt_torus toruses[1]; };
layout( std140 ) uniform rings_buf { rt_ring rings[1]; };
layout( std140 ) uniform lights_point_buf { rt_light_point lights_point[LIGHT_POINT_SIZE]; };
layout( std140 ) uniform lights_direct_buf { rt_light_direct lights_direct[LIGHT_DIRECT_SIZE]; };

vec4 quat_conj(vec4 q) { return vec4(-q.x, -q.y, -q.z, q.w); }
vec4 quat_mult(vec4 q1, vec4 q2) { 
    return vec4(
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z);
}
vec3 rotate(vec4 qr, vec3 v) { 
    vec4 qr_conj = quat_conj(qr);
    vec4 q_pos = vec4(v.xyz, 0);
    return quat_mult(quat_mult(qr, q_pos), qr_conj).xyz;
}

vec3 getRayDir() {
    vec3 result = vec3((gl_FragCoord.xy - vec2(scene.canvas_width, scene.canvas_height) / 2) / scene.canvas_height, 1);
    return normalize(rotate(scene.quat_camera_rotation, result));
}

bool intersectSphere(vec3 ro, vec3 rd, vec4 object, bool hollow, float tmin, out float t) {
    vec3 oc = ro - object.xyz;
    float b = dot(oc, rd); float c = dot(oc, oc) - object.w*object.w;
    float h = b*b - c; if(h<0.0) return false; float h_sqrt = sqrt(h);
    t = -b - h_sqrt; if(hollow && t < 0.0) t = -b + h_sqrt;
    return t > 0 && t < tmin;
}

bool intersectPlane(vec3 ro, vec3 rd, vec3 pos, vec3 nrm, float tmin, out float t) {
#ifdef PLANE_ONESIDE
    if (dot(rd, nrm) > 0.0) return false;
#endif
    float denom = dot(rd, nrm);
    if (abs(denom) < 1e-6) return false;
    t = dot(pos - ro, nrm) / denom;
    return t > 0.0 && t < tmin;
}

bool intersectBox(vec3 ro, vec3 rd, int num, float tmin, out float t) {
    rt_box box = boxes[num];
    vec3 rdd = rotate(box.quat_rotation, rd); vec3 roo = rotate(box.quat_rotation, ro - box.pos);
    vec3 m = 1.0/rdd; vec3 n = m*roo; vec3 k = abs(m)*box.form;
    vec3 t1 = -n - k; vec3 t2 = -n + k;
    float tN = max(max(t1.x, t1.y), t1.z); float tF = min(min(t2.x, t2.y), t2.z);
    if(tN > tF || tF < 0.0 || tN >= tmin) return false;
    vec3 nor = -sign(rdd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);
    t = tN; opt_normal = rotate(quat_conj(box.quat_rotation), nor);
    return true;
}

float calcInter(vec3 ro, vec3 rd, out int num, out int type) {
    float tmin = maxDist; float t;
    for(int i = 0; i < SPHERE_SIZE; i++) if(intersectSphere(ro, rd, spheres[i].obj, spheres[i].hollow, tmin, t)) { num = i; tmin = t; type = TYPE_SPHERE; }
    for(int i = 0; i < BOX_SIZE; i++) if(intersectBox(ro, rd, i, tmin, t)) { num = i; tmin = t; type = TYPE_BOX; }
    for(int i = 0; i < PLANE_SIZE; i++) if(intersectPlane(ro, rd, planes[i].pos, planes[i].normal, tmin, t)) { num = i; tmin = t; type = TYPE_PLANE; }
    for(int i = 0; i < LIGHT_POINT_SIZE; i++) if(intersectSphere(ro, rd, lights_point[i].pos, false, tmin, t)) { num = i; tmin = t; type = TYPE_POINT_LIGHT; }
    return tmin;
}

float inShadow(vec3 ro, vec3 rd, float dist) {
    float t; float shadow = 0;
    for(int i = 0; i < SPHERE_SIZE; i++) if(intersectSphere(ro, rd, spheres[i].obj, false, dist, t)) { shadow = 1; break; }
    if(shadow == 0) for(int i = 0; i < BOX_SIZE; i++) if(intersectBox(ro, rd, i, dist, t)) { shadow = 1; break; }
    if(shadow == 0) for(int i = 0; i < PLANE_SIZE; i++) if(intersectPlane(ro, rd, planes[i].pos, planes[i].normal, dist, t)) { shadow = 1; break; }
    return shadow;
}

void calcShade2(vec3 light_dir, vec3 light_color, float intensity, vec3 pt, vec3 rd, rt_material material, vec3 normal, bool doShadow, float dist, float distDiv, inout vec3 diffuse, inout vec3 specular) {
    light_dir = normalize(light_dir); float dp = clamp(dot(normal, light_dir), 0.0, 1.0); light_color *= dp;
    if(doShadow) light_color *= max(vec3(1 - inShadow(pt, light_dir, dist)), SHADOW_AMBIENT);
    diffuse += light_color * material.color * material.diffuse * intensity / distDiv;
    if(material.specular > 0) {
        vec3 reflection = reflect(light_dir, normal);
        float specDp = clamp(dot(rd, reflection), 0.0, 1.0);
        specular += light_color * pow(specDp, material.specular) * intensity / distDiv;
    }
}

vec3 calcShade(vec3 pt, vec3 rd, rt_material material, vec3 normal, bool doShadow) {
    vec3 diffuse = vec3(0), specular = vec3(0);
    vec3 pixelColor = AMBIENT_COLOR * material.color;
    for(int i = 0; i < LIGHT_POINT_SIZE; i++) {
        rt_light_point light = lights_point[i];
        vec3 light_dir = light.pos.xyz - pt; float dist = length(light_dir);
        float distDiv = 1 + light.linear_k * dist + light.quadratic_k * dist * dist;
        calcShade2(light_dir, light.color, light.intensity, pt, rd, material, normal, doShadow, dist, distDiv, diffuse, specular);
    }
    for(int i = 0; i < LIGHT_DIRECT_SIZE; i++) calcShade2(-lights_direct[i].direction, lights_direct[i].color, lights_direct[i].intensity, pt, rd, material, normal, doShadow, maxDist, 1, diffuse, specular);
    return pixelColor + diffuse * material.kd + specular * material.ks;
}

float getFresnel(vec3 normal, vec3 rd, float reflection) {
    float ndotv = clamp(dot(normal, -rd), 0.0, 1.0);
    return reflection + (1.0 - reflection) * pow(1.0 - ndotv, 5.0);
}

hit_record get_hit_info(vec3 ro, vec3 rd, vec3 pt, float t, int num, int type) {
    hit_record hr; hr.alpha = 1.0;
    if(type == TYPE_SPHERE) hr = hit_record(spheres[num].mat, normalize(pt - spheres[num].obj.xyz), 0, 1);
    if(type == TYPE_BOX) hr = hit_record(boxes[num].mat, opt_normal, 0, 1);
    if(type == TYPE_PLANE) hr = hit_record(planes[num].mat, planes[num].normal, 0, 1);
    hr.bias_mult = (9e-3 * length(pt - ro) + 35) / 35e3;
    return hr;
}

vec3 getReflectedColor(vec3 ro, vec3 rd) {
    vec3 color = vec3(0); int num, type; float t = calcInter(ro, rd, num, type);
    if(type == TYPE_POINT_LIGHT) return lights_point[num].color;
    if(t < maxDist) {
        vec3 pt = ro + rd * t; hit_record hr = get_hit_info(ro, rd, pt, t, num, type);
        ro = dot(rd, hr.normal) < 0 ? pt + hr.normal * hr.bias_mult : pt - hr.normal * hr.bias_mult;
        color = calcShade(ro, rd, hr.mat, hr.normal, true);
    }
    return color;
}

vec3 getSkyColor(vec3 rd) {
    float t = 0.5 * (rd.y + 1.0);
    return mix(vec3(0.7, 0.8, 0.9), vec3(0.2, 0.3, 0.5), t);
}

void main() {
    float reflectMultiplier, refractMultiplier, tm;
    vec3 mask = vec3(1.0), color = vec3(0.0);
    vec3 ro = vec3(scene.camera_pos), rd = getRayDir();
    float absorbDistance = 0.0; int type = 0, num; hit_record hr;
    
    for(int i = 0; i < ITERATIONS; i++) {
        tm = calcInter(ro, rd, num, type);
        if(tm < maxDist) {
            vec3 pt = ro + rd*tm; hr = get_hit_info(ro, rd, pt, tm, num, type);
            if(type == TYPE_POINT_LIGHT) { color += lights_point[num].color * mask; break; }
            rt_material mat = hr.mat; vec3 n = hr.normal;
            bool outside = dot(rd, n) < 0; n = outside ? n : -n;
            reflectMultiplier = getFresnel(n, rd, mat.reflection);
            refractMultiplier = 1 - reflectMultiplier;

            // Glass / dielectrics: continue the path through the surface (IOR in mat.refraction).
            if(mat.refraction > 1.0) {
                float eta = outside ? (1.0 / mat.refraction) : mat.refraction;
                vec3 refrDir = refract(rd, n, eta);
                color += calcShade(pt + n * hr.bias_mult, rd, mat, n, true) * reflectMultiplier * mask * 0.15;
                if (dot(refrDir, refrDir) > 0.0) {
                    ro = pt - n * hr.bias_mult;
                    rd = refrDir;
                    mask *= refractMultiplier;
                } else {
                    ro = pt + n * hr.bias_mult;
                    rd = reflect(rd, n);
                    mask *= reflectMultiplier;
                }
            } else if(mat.reflection > 0.0) {
                ro = pt + n * hr.bias_mult;
                color += calcShade(ro, rd, mat, n, true) * refractMultiplier * mask;
                rd = reflect(rd, n); mask *= reflectMultiplier;
            } else {
                color += calcShade(pt + n * hr.bias_mult, rd, mat, n, true) * mask * hr.alpha;
                break;
            }
        } else {
            color += getSkyColor(rd) * mask; break;
        }
    }
    FragColor = vec4(color, 1);
}
)GLSL";

// ---------------------------------------------------------
// OpenGL Utility & Globals
// ---------------------------------------------------------
GLuint shaderProgram;
GLuint sceneUbo = 0;
std::vector<rt_sphere> spheres;
std::vector<rt_plane> planes;
std::vector<rt_box> boxes;
std::vector<rt_light_point> lights_point;
std::vector<rt_light_direct> lights_direct;
rt_scene scene = {};

int wind_width = 1280;
int wind_height = 720;

float yaw = 0.0f, pitch = 0.0f;
Vec3 camera_pos(0, 0, -5);
bool cursorCaptured = true;
double lastX = 640, lastY = 360;
bool firstMouse = true;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    // Tab: release / recapture mouse without quitting
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        cursorCaptured = !cursorCaptured;
        glfwSetInputMode(window, GLFW_CURSOR, cursorCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        firstMouse = true;
    }
}

void framebufferSizeCallback(GLFWwindow*, int w, int h) {
    if (w < 1 || h < 1) return;
    wind_width = w;
    wind_height = h;
    scene.canvas_width = w;
    scene.canvas_height = h;
    glViewport(0, 0, w, h);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!cursorCaptured) return;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = (xpos - lastX) * 0.05f;
    float yoffset = (lastY - ypos) * 0.05f;
    lastX = xpos; lastY = ypos;
    yaw += xoffset; pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "Shader Compile Error: " << infoLog << std::endl;
    }
    return shader;
}

void init_buffers() {
    auto init_buffer = [](GLuint* ubo, const char* name, int bindingPoint, size_t size, void* data) {
        glGenBuffers(1, ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, *ubo);
        glBufferData(GL_UNIFORM_BUFFER, size, data, GL_DYNAMIC_DRAW);
        GLuint blockIndex = glGetUniformBlockIndex(shaderProgram, name);
        glUniformBlockBinding(shaderProgram, blockIndex, bindingPoint);
        glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, *ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    };
    GLuint sphereUbo, planeUbo, surfaceUbo, boxUbo, torusUbo, ringUbo, lightPointUbo, lightDirectUbo;
    init_buffer(&sceneUbo, "scene_buf", 0, sizeof(rt_scene), NULL);
    init_buffer(&sphereUbo, "spheres_buf", 1, sizeof(rt_sphere) * std::max(1, (int)spheres.size()), spheres.empty() ? NULL : spheres.data());
    init_buffer(&planeUbo, "planes_buf", 2, sizeof(rt_plane) * std::max(1, (int)planes.size()), planes.empty() ? NULL : planes.data());
    init_buffer(&surfaceUbo, "surfaces_buf", 3, sizeof(rt_surface), NULL);
    init_buffer(&boxUbo, "boxes_buf", 4, sizeof(rt_box) * std::max(1, (int)boxes.size()), boxes.empty() ? NULL : boxes.data());
    init_buffer(&torusUbo, "toruses_buf", 5, sizeof(rt_torus), NULL);
    init_buffer(&ringUbo, "rings_buf", 6, sizeof(rt_ring), NULL); // FIXED: Now safely uses rt_ring
    init_buffer(&lightPointUbo, "lights_point_buf", 7, sizeof(rt_light_point) * std::max(1, (int)lights_point.size()), lights_point.empty() ? NULL : lights_point.data());
    init_buffer(&lightDirectUbo, "lights_direct_buf", 8, sizeof(rt_light_direct) * std::max(1, (int)lights_direct.size()), lights_direct.empty() ? NULL : lights_direct.data());
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main() {
    if (!glfwInit()) return -1;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(wind_width, wind_height, "Raytracer", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    
    glfwMakeContextCurrent(window);
    //glfwSwapInterval(1); // vsync; drop the old 15 FPS sleep so WASD dt stays real
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    // Setup Scene
    scene.canvas_width = wind_width;
    scene.canvas_height = wind_height;
    scene.reflect_depth = 4;
    scene.quat_camera_rotation[3] = 1.0f;

    spheres.push_back(create_sphere(2, 0, 6, 1, create_material(0, 0, 1, 50, 0.35, 0.0f, 1.0f)));
    spheres.push_back(create_sphere(-1, 0, 6, 1, create_material(1, 0, 0, 100, 0.1, 1.125f, 1.0f), true));
    spheres.push_back(create_sphere(0.5, 2, 6, 1, create_material(0.1, 1, 0.1, 200, 0.8, 0.0f, 0.2f)));

    planes.push_back(create_plane(0, -10.0, 0, 0, 1, 0, create_material(0.55, 0.55, 0.6, 20, 0.08, 0.0f, 1.0f)));

    boxes.push_back(create_box(0, -1.2, 6, 10, 0.2, 5, create_material(1, 0.6, 0, 100, 0.05, 0.0f, 1.0f)));
    boxes.push_back(create_box(8, 1, 6, 1, 1, 1, create_material(0.8, 0.7, 0, 50, 0.0, 0.0f, 1.0f)));

    lights_point.push_back(create_point_light(3, 5, 0, 1, 1, 1, 25.5));
    lights_direct.push_back(create_direct_light(3, -1, 1, 1, 1, 1, 1.5));

    // Compile Shader
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs);
    glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram);
    {
        GLint linked = 0;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linked);
        if (!linked) {
            char infoLog[1024];
            glGetProgramInfoLog(shaderProgram, 1024, NULL, infoLog);
            std::cerr << "Shader Link Error: " << infoLog << std::endl;
            return -1;
        }
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    init_buffers();

    // Fullscreen Quad VAO
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    glUseProgram(shaderProgram);

    double lastFrameTime = glfwGetTime();
    double fpsTimer = 0;
    int frames = 0;
    const float cameraSpeed = 3.0f; // world units per second

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents(); // apply WASD / mouse before this frame's camera step

        double currentTime = glfwGetTime();
        // Must capture wall-clock dt (vsync included). Sleeping then stamping time made dt~0.
        double deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        if (deltaTime > 0.1) deltaTime = 0.1; // clamp hitch / first-frame spikes

        // Orientation shared with the GPU: shader rotates +Z rays by this quaternion.
        Quat q = quatFromEuler(radians(-pitch), radians(yaw));
        Vec3 front = normalize(rotate(q, Vec3(0, 0, 1)));
        Vec3 right = normalize(cross(front, Vec3(0, 1, 0)));
        if (right.x == 0.0f && right.y == 0.0f && right.z == 0.0f)
            right = Vec3(1, 0, 0);
        Vec3 up = normalize(cross(right, front));

        float speedMul = 1.0f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
            speedMul = 3.0f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS)
            speedMul = 0.25f;
        float speed = (float)deltaTime * cameraSpeed * speedMul;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera_pos += front * speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera_pos -= front * speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera_pos -= right * speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera_pos += right * speed;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera_pos += up * speed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) camera_pos -= up * speed;

        scene.camera_pos[0] = camera_pos.x; scene.camera_pos[1] = camera_pos.y; scene.camera_pos[2] = camera_pos.z;
        scene.quat_camera_rotation[0] = q.x; scene.quat_camera_rotation[1] = q.y; scene.quat_camera_rotation[2] = q.z; scene.quat_camera_rotation[3] = q.w;

        glBindBuffer(GL_UNIFORM_BUFFER, sceneUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_scene), &scene);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
usleep(30000);
        frames++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0) {
            std::string title = "Raytracer  " + std::to_string(frames) + " FPS";
            glfwSetWindowTitle(window, title.c_str());
            frames = 0;
            fpsTimer = 0;
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}