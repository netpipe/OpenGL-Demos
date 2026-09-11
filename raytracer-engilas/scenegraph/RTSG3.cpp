// Raytracer (engilas) — Scenegraph, Godrays, Fog, Water, OBJ Bump Mapping, Culling
// Controls: mouse look, WASD, Space/Ctrl up-down, Shift boost, Alt slow, Tab free cursor, Esc quit
// Linux:   g++ raytracer.cpp -o raytracer -lglfw -lGLEW -lGL -lm
// macOS:   g++ raytracer.cpp -o raytracer -lglfw -lGLEW -framework OpenGL -lm
// Windows: see CMakeLists.txt

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
#include <fstream>
#include <sstream>
#include <map>

#define PI_F 3.14159265358979f

#define MAX_SPHERES 10
#define MAX_BOXES 5
#define MAX_PLANES 2
#define MAX_WATERS 2
#define MAX_LIGHTS_POINT 3
#define MAX_LIGHTS_SPOT 2
#define MAX_LIGHTS_AREA 2
#define MAX_LIGHTS_DIRECT 2
#define MAX_MESH_TRIANGLES 256 // 256 * 192 = 49152 bytes; fits in a 64KB UBO

// ---------------------------------------------------------
// Math Structs & Helpers (No GLM)
// ---------------------------------------------------------
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
    Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
};

struct Vec2 {
    float x, y;
    Vec2() : x(0), y(0) {}
    Vec2(float X, float Y) : x(X), y(Y) {}
    Vec2 operator-(const Vec2& b) const { return Vec2(x-b.x, y-b.y); }
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

float radians(float deg) { return deg * PI_F / 180.0f; }

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

Quat quatConj(Quat q) { return Quat(-q.x, -q.y, -q.z, q.w); }

Vec3 quatRotate(Quat q, Vec3 v) {
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

struct rt_water {
    float pos[3]; float waveHeight;
    float normal[3]; float waveFreq;
    float color[3]; float flowSpeed;
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

struct rt_light_spot {
    float pos[4];
    float dir[4];
    float color[3]; float intensity;
    float cutoff; float outerCutoff; float __pad[2];
};

struct rt_light_area {
    float pos[4];
    float right[4];
    float up[4];
    float color[3]; float intensity;
};

struct rt_fog {
    float color[3]; float density;
    int enableGodrays; int godrayLightIndex; float __pad[2];
};

struct rt_triangle {
    float v0[4]; float v1[4]; float v2[4];
    float n0[4]; float n1[4]; float n2[4];
    float uv0[4]; float uv1[4]; float uv2[4];
    float t0[4]; float t1[4]; float t2[4];
};

struct rt_scene {
    float quat_camera_rotation[4];
    float camera_pos[3]; float __p1;
    float bg_color[3]; int canvas_width;
    int canvas_height; int reflect_depth; 
    int sphere_count; int box_count; int plane_count; int water_count;
    int light_point_count; int light_spot_count; int light_area_count; int light_direct_count;
    int mesh_count; float time; float __pad[2];
};

#pragma pack(pop)

// ---------------------------------------------------------
// OBJ Loader & Bump Mapping Support
// ---------------------------------------------------------
struct BumpVertex {
    Vec3 position; Vec3 normal; Vec2 texcoord; Vec3 tangent; Vec3 bitangent;
};

struct Mesh {
    std::vector<BumpVertex> vertices;
    Mesh() {}
};

class OBJModel {
public:
    std::vector<Mesh> meshes;

    bool parse(std::istream& file) {
        std::vector<Vec3> positions, normals;
        std::vector<Vec2> texcoords;
        Mesh* currentMesh = nullptr;
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix;
            if (prefix == "v") { Vec3 v; ss >> v.x >> v.y >> v.z; positions.push_back(v); }
            else if (prefix == "vn") { Vec3 n; ss >> n.x >> n.y >> n.z; normals.push_back(n); }
            else if (prefix == "vt") { Vec2 t; ss >> t.x >> t.y; texcoords.push_back(t); }
            else if (prefix == "f") {
                if (!currentMesh) { meshes.push_back(Mesh()); currentMesh = &meshes.back(); }
                std::string v[4]; int count = 0;
                while (count < 4 && ss >> v[count]) count++;
                for (int i = 1; i + 1 < count; ++i) {
                    parseVertex(v[0], positions, texcoords, normals, *currentMesh);
                    parseVertex(v[i], positions, texcoords, normals, *currentMesh);
                    parseVertex(v[i+1], positions, texcoords, normals, *currentMesh);
                }
            }
        }
        computeTangents();
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream file(filename.c_str());
        if (!file) return false;
        return parse(file);
    }

    bool loadFromMemory(const std::string& objData) {
        std::stringstream file(objData);
        return parse(file);
    }

private:
    void parseVertex(const std::string& token, const std::vector<Vec3>& positions,
                     const std::vector<Vec2>& texcoords, const std::vector<Vec3>& normals, Mesh& mesh) {
        BumpVertex vert;
        int vi = 0, ti = 0, ni = 0;
        size_t slash1 = token.find('/');
        if (slash1 == std::string::npos) { vi = std::stoi(token); } 
        else {
            vi = std::stoi(token.substr(0, slash1));
            size_t slash2 = token.find('/', slash1 + 1);
            if (slash2 == std::string::npos) {
                std::string tStr = token.substr(slash1 + 1);
                if (!tStr.empty()) ti = std::stoi(tStr);
            } else {
                std::string tStr = token.substr(slash1 + 1, slash2 - slash1 - 1);
                if (!tStr.empty()) ti = std::stoi(tStr);
                std::string nStr = token.substr(slash2 + 1);
                if (!nStr.empty()) ni = std::stoi(nStr);
            }
        }
        vert.position = (vi > 0 && vi <= (int)positions.size()) ? positions[vi - 1] : Vec3();
        vert.texcoord = (ti > 0 && ti <= (int)texcoords.size()) ? texcoords[ti - 1] : Vec2();
        vert.texcoord.y = 1.0f - vert.texcoord.y; // Flip Y
        vert.normal = (ni > 0 && ni <= (int)normals.size()) ? normals[ni - 1] : Vec3(0,0,1);
        vert.tangent = Vec3(0,0,0); vert.bitangent = Vec3(0,0,0);
        mesh.vertices.push_back(vert);
    }

    void computeTangents() {
        for (auto& mesh : meshes) {
            for (size_t i = 0; i + 2 < mesh.vertices.size(); i += 3) {
                BumpVertex& v0 = mesh.vertices[i];
                BumpVertex& v1 = mesh.vertices[i+1];
                BumpVertex& v2 = mesh.vertices[i+2];

                Vec3 edge1 = v1.position - v0.position;
                Vec3 edge2 = v2.position - v0.position;
                Vec2 deltaUV1 = v1.texcoord - v0.texcoord;
                Vec2 deltaUV2 = v2.texcoord - v0.texcoord;

                float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
                Vec3 tangent, bitangent;
                if (fabsf(denom) < 1e-8f) {
                    Vec3 n = normalize(v0.normal);
                    Vec3 helper = (fabsf(n.y) < 0.9f) ? Vec3(0,1,0) : Vec3(1,0,0);
                    tangent = normalize(cross(n, helper));
                    bitangent = normalize(cross(n, tangent));
                } else {
                    float f = 1.0f / denom;
                    tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
                    tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
                    tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
                    tangent = normalize(tangent);

                    bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
                    bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
                    bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
                    bitangent = normalize(bitangent);
                }

                v0.tangent = tangent; v0.bitangent = bitangent;
                v1.tangent = tangent; v1.bitangent = bitangent;
                v2.tangent = tangent; v2.bitangent = bitangent;
            }
        }
    }
};

// ---------------------------------------------------------
// Scenegraph Node System
// ---------------------------------------------------------
class SceneNode {
public:
    Vec3 pos; Quat rot; Vec3 scl; Vec3 tint;
    std::vector<SceneNode*> children;
    float cull_distance = 100.0f; // Distant Culling
    
    SceneNode() : pos(0,0,0), rot(0,0,0,1), scl(1,1,1), tint(1,1,1) {}
    virtual ~SceneNode() { for(auto c : children) delete c; }
    
    void setposition(float x, float y, float z) { pos = Vec3(x, y, z); }
    void rotate(float pitch, float yaw) { rot = quatMult(rot, quatFromEuler(radians(pitch), radians(yaw))); }
    void scale(float x, float y, float z) { scl = Vec3(x, y, z); }
    void colorize(float r, float g, float b) { tint = Vec3(r, g, b); }
    void addChild(SceneNode* child) { children.push_back(child); }
    
    Vec3 getGlobalPos(Vec3 pPos, Quat pRot, Vec3 pScl) {
        Vec3 scaledPos = Vec3(pos.x * pScl.x, pos.y * pScl.y, pos.z * pScl.z);
        Vec3 rotPos = quatRotate(pRot, scaledPos);
        return pPos + rotPos;
    }
    Quat getGlobalRot(Quat pRot) { return quatMult(pRot, rot); }
    Vec3 getGlobalScl(Vec3 pScl) { return Vec3(pScl.x * scl.x, pScl.y * scl.y, pScl.z * scl.z); }
    
    virtual void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                         std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                         std::vector<rt_light_area>&, std::vector<rt_light_direct>&,
                         std::vector<rt_triangle>&,
                         Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) {}
                         
    void traverse(std::vector<rt_sphere>& sph, std::vector<rt_box>& bx, std::vector<rt_plane>& pl, 
                  std::vector<rt_water>& wt, std::vector<rt_light_point>& lp, std::vector<rt_light_spot>& ls,
                  std::vector<rt_light_area>& la, std::vector<rt_light_direct>& ld, std::vector<rt_triangle>& tris,
                  Vec3 pPos, Quat pRot, Vec3 pScl, Vec3 pTint, Vec3 camPos) {
        
        Vec3 gPos = getGlobalPos(pPos, pRot, pScl);
        
        // DISTANT CULLING SYSTEM
        if (cull_distance > 0.0f) {
            float dx = gPos.x - camPos.x;
            float dy = gPos.y - camPos.y;
            float dz = gPos.z - camPos.z;
            float distSq = dx*dx + dy*dy + dz*dz;
            if (distSq > cull_distance * cull_distance) {
                return; // Skip this node and its children if out of range
            }
        }

        Quat gRot = getGlobalRot(pRot);
        Vec3 gScl = getGlobalScl(pScl);
        Vec3 gTint = Vec3(pTint.x * tint.x, pTint.y * tint.y, pTint.z * tint.z);
        
        flatten(sph, bx, pl, wt, lp, ls, la, ld, tris, gPos, gRot, gScl, gTint);
        for(auto c : children) {
            c->traverse(sph, bx, pl, wt, lp, ls, la, ld, tris, gPos, gRot, gScl, gTint, camPos);
        }
    }
};

class SphereNode : public SceneNode {
    float radius; rt_material mat;
public:
    SphereNode(float r, rt_material m) : radius(r), mat(m) {}
    void flatten(std::vector<rt_sphere>& sph, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>&, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        rt_sphere s = {};
        s.obj[0] = gPos.x; s.obj[1] = gPos.y; s.obj[2] = gPos.z; 
        s.obj[3] = radius * gScl.x;
        s.quat_rotation[0] = gRot.x; s.quat_rotation[1] = gRot.y; s.quat_rotation[2] = gRot.z; s.quat_rotation[3] = gRot.w;
        s.material = mat;
        s.material.color[0] *= gTint.x; s.material.color[1] *= gTint.y; s.material.color[2] *= gTint.z;
        if(sph.size() < MAX_SPHERES) sph.push_back(s);
    }
};

class BoxNode : public SceneNode {
    Vec3 dims; rt_material mat;
public:
    BoxNode(float w, float h, float d, rt_material m) : dims(w,h,d), mat(m) {}
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>& bx, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>&, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        rt_box b = {};
        b.pos[0] = gPos.x; b.pos[1] = gPos.y; b.pos[2] = gPos.z;
        b.form[0] = dims.x * gScl.x; b.form[1] = dims.y * gScl.y; b.form[2] = dims.z * gScl.z;
        b.quat_rotation[0] = gRot.x; b.quat_rotation[1] = gRot.y; b.quat_rotation[2] = gRot.z; b.quat_rotation[3] = gRot.w;
        b.mat = mat;
        b.mat.color[0] *= gTint.x; b.mat.color[1] *= gTint.y; b.mat.color[2] *= gTint.z;
        if(bx.size() < MAX_BOXES) bx.push_back(b);
    }
};

class WaterNode : public SceneNode {
public:
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>& wt, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>&, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        rt_water w = {};
        w.pos[0] = gPos.x; w.pos[1] = gPos.y; w.pos[2] = gPos.z;
        Vec3 n = normalize(quatRotate(gRot, Vec3(0,1,0)));
        w.normal[0] = n.x; w.normal[1] = n.y; w.normal[2] = n.z;
        w.color[0] = gTint.x; w.color[1] = gTint.y; w.color[2] = gTint.z;
        w.waveHeight = 0.2f; w.waveFreq = 2.0f; w.flowSpeed = 1.0f;
        if(wt.size() < MAX_WATERS) wt.push_back(w);
    }
};

class MeshNode : public SceneNode {
    OBJModel* model;
public:
    MeshNode(const std::string& objData) {
        model = new OBJModel();
        model->loadFromMemory(objData);
       // model->load("capsule.obj");
    }
    ~MeshNode() { delete model; }
    
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>&, std::vector<rt_triangle>& tris,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        if (!model) return;
        for (auto& mesh : model->meshes) {
            for (size_t i = 0; i + 2 < mesh.vertices.size(); i += 3) {
                if (tris.size() >= MAX_MESH_TRIANGLES) break;
                
                rt_triangle tri = {};
                BumpVertex& bv0 = mesh.vertices[i];
                BumpVertex& bv1 = mesh.vertices[i+1];
                BumpVertex& bv2 = mesh.vertices[i+2];
                
                auto transformPos = [&](Vec3 v) {
                    return quatRotate(gRot, Vec3(v.x * gScl.x, v.y * gScl.y, v.z * gScl.z)) + gPos;
                };
                auto transformDir = [&](Vec3 v) {
                    return normalize(quatRotate(gRot, v));
                };
                
                Vec3 p0 = transformPos(bv0.position); Vec3 p1 = transformPos(bv1.position); Vec3 p2 = transformPos(bv2.position);
                tri.v0[0]=p0.x; tri.v0[1]=p0.y; tri.v0[2]=p0.z;
                tri.v1[0]=p1.x; tri.v1[1]=p1.y; tri.v1[2]=p1.z;
                tri.v2[0]=p2.x; tri.v2[1]=p2.y; tri.v2[2]=p2.z;
                
                Vec3 n0 = transformDir(bv0.normal); Vec3 n1 = transformDir(bv1.normal); Vec3 n2 = transformDir(bv2.normal);
                tri.n0[0]=n0.x; tri.n0[1]=n0.y; tri.n0[2]=n0.z;
                tri.n1[0]=n1.x; tri.n1[1]=n1.y; tri.n1[2]=n1.z;
                tri.n2[0]=n2.x; tri.n2[1]=n2.y; tri.n2[2]=n2.z;
                
                tri.uv0[0]=bv0.texcoord.x; tri.uv0[1]=bv0.texcoord.y;
                tri.uv1[0]=bv1.texcoord.x; tri.uv1[1]=bv1.texcoord.y;
                tri.uv2[0]=bv2.texcoord.x; tri.uv2[1]=bv2.texcoord.y;
                
                Vec3 t0 = transformDir(bv0.tangent); Vec3 t1 = transformDir(bv1.tangent); Vec3 t2 = transformDir(bv2.tangent);
                tri.t0[0]=t0.x; tri.t0[1]=t0.y; tri.t0[2]=t0.z;
                tri.t1[0]=t1.x; tri.t1[1]=t1.y; tri.t1[2]=t1.z;
                tri.t2[0]=t2.x; tri.t2[1]=t2.y; tri.t2[2]=t2.z;
                
               // Vec3 b0 = transformDir(bv0.bitangent); Vec3 b1 = transformDir(bv1.bitangent); Vec3 b2 = transformDir(bv2.bitangent);
              //  tri.b0[0]=b0.x; tri.b0[1]=b0.y; tri.b0[2]=b0.z;
              //  tri.b1[0]=b1.x; tri.b1[1]=b1.y; tri.b1[2]=b1.z;
              //  tri.b2[0]=b2.x; tri.b2[1]=b2.y; tri.b2[2]=b2.z;
                
                tris.push_back(tri);
            }
        }
    }
};

class PointLightNode : public SceneNode {
    Vec3 col; float intens;
public:
    PointLightNode(Vec3 c, float i) : col(c), intens(i) {}
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>& lp, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>&, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        rt_light_point l = {};
        l.pos[0] = gPos.x; l.pos[1] = gPos.y; l.pos[2] = gPos.z; l.pos[3] = 0.2f;
        l.color[0] = col.x * gTint.x; l.color[1] = col.y * gTint.y; l.color[2] = col.z * gTint.z;
        l.intensity = intens; l.linear_k = 0.09f; l.quadratic_k = 0.032f;
        if(lp.size() < MAX_LIGHTS_POINT) lp.push_back(l);
    }
};

class SpotLightNode : public SceneNode {
    Vec3 col; float intens, cut, outCut;
public:
    SpotLightNode(Vec3 c, float i, float ct, float oct) : col(c), intens(i), cut(ct), outCut(oct) {}
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>& ls,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>&, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        rt_light_spot l = {};
        l.pos[0] = gPos.x; l.pos[1] = gPos.y; l.pos[2] = gPos.z;
        Vec3 d = quatRotate(gRot, Vec3(0,0,-1));
        l.dir[0] = d.x; l.dir[1] = d.y; l.dir[2] = d.z;
        l.color[0] = col.x * gTint.x; l.color[1] = col.y * gTint.y; l.color[2] = col.z * gTint.z;
        l.intensity = intens; l.cutoff = cut; l.outerCutoff = outCut;
        if(ls.size() < MAX_LIGHTS_SPOT) ls.push_back(l);
    }
};

class AreaLightNode : public SceneNode {
    Vec3 col; float intens, w, h;
public:
    AreaLightNode(Vec3 c, float i, float width, float height) : col(c), intens(i), w(width), h(height) {}
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>& la, std::vector<rt_light_direct>&, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        rt_light_area l = {};
        l.pos[0] = gPos.x; l.pos[1] = gPos.y; l.pos[2] = gPos.z;
        Vec3 r = quatRotate(gRot, Vec3(w*0.5f, 0, 0));
        Vec3 u = quatRotate(gRot, Vec3(0, h*0.5f, 0));
        l.right[0] = r.x; l.right[1] = r.y; l.right[2] = r.z;
        l.up[0] = u.x; l.up[1] = u.y; l.up[2] = u.z;
        l.color[0] = col.x * gTint.x; l.color[1] = col.y * gTint.y; l.color[2] = col.z * gTint.z;
        l.intensity = intens;
        if(la.size() < MAX_LIGHTS_AREA) la.push_back(l);
    }
};

class SunLightNode : public SceneNode {
    Vec3 col; float intens;
public:
    SunLightNode(Vec3 c, float i) : col(c), intens(i) {}
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>& ld, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        rt_light_direct l = {};
        Vec3 d = quatRotate(gRot, Vec3(0,-1,0));
        l.direction[0] = d.x; l.direction[1] = d.y; l.direction[2] = d.z;
        l.color[0] = col.x * gTint.x; l.color[1] = col.y * gTint.y; l.color[2] = col.z * gTint.z;
        l.intensity = intens;
        if(ld.size() < MAX_LIGHTS_DIRECT) ld.push_back(l);
    }
};

rt_fog global_fog = {};
class FogNode : public SceneNode {
public:
    float density; bool godrays; int lightIdx; Vec3 col;
    FogNode(float d, bool gr, int idx, Vec3 c) : density(d), godrays(gr), lightIdx(idx), col(c) {}
    void flatten(std::vector<rt_sphere>&, std::vector<rt_box>&, std::vector<rt_plane>&, 
                 std::vector<rt_water>&, std::vector<rt_light_point>&, std::vector<rt_light_spot>&,
                 std::vector<rt_light_area>&, std::vector<rt_light_direct>&, std::vector<rt_triangle>&,
                 Vec3 gPos, Quat gRot, Vec3 gScl, Vec3 gTint) override {
        global_fog.density = density;
        global_fog.enableGodrays = godrays ? 1 : 0;
        global_fog.godrayLightIndex = lightIdx;
        global_fog.color[0] = col.x; global_fog.color[1] = col.y; global_fog.color[2] = col.z;
    }
};

// ---------------------------------------------------------
// Helper functions
// ---------------------------------------------------------
rt_material create_material(float r, float g, float b, int specular, float reflect, float refract, float diffuse) {
    rt_material m = {};
    m.color[0] = r; m.color[1] = g; m.color[2] = b;
    m.specular = specular; m.reflect = reflect; m.refract = refract; m.diffuse = diffuse;
    m.kd = 1.0f; m.ks = 1.0f;
    return m;
}

// ---------------------------------------------------------
// GLSL Shaders
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
#define TYPE_BOX 3
#define TYPE_POINT_LIGHT 6
#define TYPE_WATER 7
#define TYPE_MESH 8

#define SHADOW_ENABLED 1
#define PLANE_ONESIDE 1

struct rt_material { vec3 color; vec3 absorb; float diffuse; float reflection; float refraction; int specular; float kd; float ks; };
struct rt_sphere { rt_material mat; vec4 obj; vec4 quat_rotation; int textureNum; bool hollow; };
struct rt_plane { rt_material mat; vec3 pos; vec3 normal; };
struct rt_box { rt_material mat; vec4 quat_rotation; vec3 pos; vec3 form; int textureNum; };
struct rt_light_direct { vec3 direction; vec3 color; float intensity; };
struct rt_light_point { vec4 pos; vec3 color; float intensity; float linear_k; float quadratic_k; };
struct rt_light_spot { vec4 pos; vec4 dir; vec3 color; float intensity; float cutoff; float outerCutoff; vec2 __pad; };
struct rt_light_area { vec4 pos; vec4 right; vec4 up; vec3 color; float intensity; };
struct rt_water { vec3 pos; float waveHeight; vec3 normal; float waveFreq; vec3 color; float flowSpeed; };
struct rt_triangle { vec4 v0, v1, v2; vec4 n0, n1, n2; vec4 uv0, uv1, uv2; vec4 t0, t1, t2; };
struct rt_fog { vec3 color; float density; int enableGodrays; int godrayLightIndex; vec2 __pad; };
struct rt_scene { vec4 quat_camera_rotation; vec3 camera_pos; vec3 bg_color; int canvas_width; int canvas_height; int reflect_depth; int sphere_count; int box_count; int plane_count; int water_count; int light_point_count; int light_spot_count; int light_area_count; int light_direct_count; int mesh_count; float time; };
struct hit_record { rt_material mat; vec3 normal; float bias_mult; float alpha; };

#define AMBIENT_COLOR vec3(0.05, 0.05, 0.05)
#define SHADOW_AMBIENT vec3(0.2, 0.2, 0.2)
#define ITERATIONS 2

out vec4 FragColor;
const float maxDist = 10000.0;
vec3 opt_normal;

layout( std140 ) uniform scene_buf { rt_scene scene; };
layout( std140 ) uniform spheres_buf { rt_sphere spheres[10]; };
layout( std140 ) uniform planes_buf { rt_plane planes[2]; };
layout( std140 ) uniform boxes_buf { rt_box boxes[5]; };
layout( std140 ) uniform waters_buf { rt_water waters[2]; };
layout( std140 ) uniform lights_point_buf { rt_light_point lights_point[3]; };
layout( std140 ) uniform lights_spot_buf { rt_light_spot lights_spot[2]; };
layout( std140 ) uniform lights_area_buf { rt_light_area lights_area[2]; };
layout( std140 ) uniform lights_direct_buf { rt_light_direct lights_direct[2]; };
layout( std140 ) uniform mesh_buf { rt_triangle mesh_tris[256]; };
layout( std140 ) uniform fog_buf { rt_fog fog; };

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

bool intersectWater(vec3 ro, vec3 rd, vec3 pos, vec3 nrm, float tmin, out float t) {
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

// Moller-Trumbore Triangle Intersection
bool intersectTriangle(vec3 ro, vec3 rd, int triIndex, float tmin, out float t, out vec3 bar, out vec3 normal) {
    rt_triangle tri = mesh_tris[triIndex];
    vec3 edge1 = tri.v1.xyz - tri.v0.xyz;
    vec3 edge2 = tri.v2.xyz - tri.v0.xyz;
    vec3 h = cross(rd, edge2);
    float a = dot(edge1, h);
    if (a > -0.00001 && a < 0.00001) return false;
    float f = 1.0 / a;
    vec3 s = ro - tri.v0.xyz;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    vec3 q = cross(s, edge1);
    float v = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) return false;
    t = f * dot(edge2, q);
    if (t > 0.0001 && t < tmin) {
        bar = vec3(1.0 - u - v, u, v);
        normal = normalize(tri.n0.xyz * bar.x + tri.n1.xyz * bar.y + tri.n2.xyz * bar.z);
        return true;
    }
    return false;
}

float calcInter(vec3 ro, vec3 rd, out int num, out int type, out int outTriIndex, out vec3 outBar) {
    float tmin = maxDist; float t; vec3 bar, normal;
    for(int i = 0; i < scene.sphere_count; i++) if(intersectSphere(ro, rd, spheres[i].obj, spheres[i].hollow, tmin, t)) { num = i; tmin = t; type = TYPE_SPHERE; }
    for(int i = 0; i < scene.box_count; i++) if(intersectBox(ro, rd, i, tmin, t)) { num = i; tmin = t; type = TYPE_BOX; }
    for(int i = 0; i < scene.plane_count; i++) if(intersectPlane(ro, rd, planes[i].pos, planes[i].normal, tmin, t)) { num = i; tmin = t; type = TYPE_PLANE; }
    for(int i = 0; i < scene.water_count; i++) if(intersectWater(ro, rd, waters[i].pos, waters[i].normal, tmin, t)) { num = i; tmin = t; type = TYPE_WATER; }
    for(int i = 0; i < scene.mesh_count; i++) {
        if(intersectTriangle(ro, rd, i, tmin, t, bar, normal)) {
            num = i; tmin = t; type = TYPE_MESH; outTriIndex = i; outBar = bar;
        }
    }
    for(int i = 0; i < scene.light_point_count; i++) if(intersectSphere(ro, rd, lights_point[i].pos, false, tmin, t)) { num = i; tmin = t; type = TYPE_POINT_LIGHT; }
    return tmin;
}

float inShadow(vec3 ro, vec3 rd, float dist) {
    float t; float shadow = 0; vec3 bar, normal;
    for(int i = 0; i < scene.sphere_count; i++) if(intersectSphere(ro, rd, spheres[i].obj, false, dist, t)) { shadow = 1; break; }
    if(shadow == 0) for(int i = 0; i < scene.box_count; i++) if(intersectBox(ro, rd, i, dist, t)) { shadow = 1; break; }
    if(shadow == 0) for(int i = 0; i < scene.plane_count; i++) if(intersectPlane(ro, rd, planes[i].pos, planes[i].normal, dist, t)) { shadow = 1; break; }
    if(shadow == 0) for(int i = 0; i < scene.mesh_count; i++) if(intersectTriangle(ro, rd, i, dist, t, bar, normal)) { shadow = 1; break; }
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
    for(int i = 0; i < scene.light_point_count; i++) {
        rt_light_point light = lights_point[i];
        vec3 light_dir = light.pos.xyz - pt; float dist = length(light_dir);
        float distDiv = 1 + light.linear_k * dist + light.quadratic_k * dist * dist;
        calcShade2(light_dir, light.color, light.intensity, pt, rd, material, normal, doShadow, dist, distDiv, diffuse, specular);
    }
    for(int i = 0; i < scene.light_spot_count; i++) {
        rt_light_spot light = lights_spot[i];
        vec3 light_dir = light.pos.xyz - pt; float dist = length(light_dir);
        light_dir /= dist;
        float theta = dot(light_dir, normalize(-light.dir.xyz));
        float epsilon = light.cutoff - light.outerCutoff;
        float intensity = clamp((theta - light.outerCutoff) / epsilon, 0.0, 1.0);
        if(intensity > 0.0) {
            float distDiv = 1.0 + 0.09 * dist + 0.032 * dist * dist;
            calcShade2(light_dir, light.color, light.intensity * intensity, pt, rd, material, normal, doShadow, dist, distDiv, diffuse, specular);
        }
    }
    for(int i = 0; i < scene.light_area_count; i++) {
        rt_light_area light = lights_area[i];
        vec3 light_dir = light.pos.xyz - pt; float dist = length(light_dir);
        float distDiv = 1.0 + 0.09 * dist + 0.032 * dist * dist;
        calcShade2(light_dir, light.color, light.intensity, pt, rd, material, normal, doShadow, dist, distDiv, diffuse, specular);
    }
    for(int i = 0; i < scene.light_direct_count; i++) calcShade2(-lights_direct[i].direction, lights_direct[i].color, lights_direct[i].intensity, pt, rd, material, normal, doShadow, maxDist, 1, diffuse, specular);
    return pixelColor + diffuse * material.kd + specular * material.ks;
}

float getFresnel(vec3 normal, vec3 rd, float reflection) {
    float ndotv = clamp(dot(normal, -rd), 0.0, 1.0);
    return reflection + (1.0 - reflection) * pow(1.0 - ndotv, 5.0);
}

hit_record get_hit_info(vec3 ro, vec3 rd, vec3 pt, float t, int num, int type, int triIndex, vec3 bar) {
    rt_triangle tri;
    if (type == TYPE_MESH) tri = mesh_tris[triIndex];
    hit_record hr; hr.alpha = 1.0;
    
    if(type == TYPE_SPHERE) hr = hit_record(spheres[num].mat, normalize(pt - spheres[num].obj.xyz), 0, 1);
    if(type == TYPE_BOX) hr = hit_record(boxes[num].mat, opt_normal, 0, 1);
    if(type == TYPE_PLANE) hr = hit_record(planes[num].mat, planes[num].normal, 0, 1);
    
    if(type == TYPE_WATER) {
        // FIX: Manually construct the water material since rt_water has no .mat field
        rt_material waterMat = rt_material(
            waters[num].color,  // color
            vec3(0.0),          // absorb
            0.15,               // diffuse
            0.6,                // reflection
            1.33,               // refraction / IOR water
            200,                // specular exponent
            1.0,                // kd
            1.0                 // ks
        );
        hr = hit_record(waterMat, waters[num].normal, 0, 1);
        
        vec3 p = pt;
        float wave = sin(p.x * waters[num].waveFreq + scene.time * waters[num].flowSpeed) * 
                     cos(p.z * waters[num].waveFreq + scene.time * waters[num].flowSpeed);
        vec3 perturb = vec3(wave, 0.0, wave) * waters[num].waveHeight;
        hr.normal = normalize(hr.normal + perturb);
        hr.mat.reflection = 0.6;
        hr.mat.refraction = 1.33;
        hr.mat.color = waters[num].color;
    }
    
       if(type == TYPE_MESH) {
        vec3 n = tri.n0.xyz * bar.x + tri.n1.xyz * bar.y + tri.n2.xyz * bar.z;
        vec2 uv = tri.uv0.xy * bar.x + tri.uv1.xy * bar.y + tri.uv2.xy * bar.z;
        vec3 T = tri.t0.xyz * bar.x + tri.t1.xyz * bar.y + tri.t2.xyz * bar.z;

        n = normalize(n);
        T = normalize(T - dot(T, n) * n);  // Gram-Schmidt orthogonalize
        vec3 B = cross(n, T);              // Reconstruct bitangent

        mat3 TBN = mat3(T, B, n);

        // Procedural Bump Map (Checkerboard)
        float checker = mod(floor(uv.x * 10.0) + floor(uv.y * 10.0), 2.0);
        vec3 bumpN = vec3(0.0, 0.0, 1.0);
        if (checker > 0.5) bumpN = vec3(0.371, 0.0, 0.928); // pre-normalized

        vec3 finalNormal = normalize(TBN * bumpN);
        rt_material m = rt_material(vec3(0.8, 0.8, 0.8), vec3(0.0), 0.8, 0.1, 1.0, 50, 1.0, 1.0);
        hr = hit_record(m, finalNormal, 0.0, 1.0);
    }
    
    hr.bias_mult = (9e-3 * length(pt - ro) + 35) / 35e3;
    return hr;
}

vec3 applyFog(vec3 ro, vec3 rd, float dist, vec3 hitColor) {
    if(fog.density <= 0.0 || dist >= maxDist) return hitColor;
    float t = 0.0; float stepSize = dist / 8.0;
    vec3 inscatter = vec3(0.0); float transmittance = 1.0;
    for(int i = 0; i < 8; i++) {
        t += stepSize; vec3 p = ro + rd * t;
        transmittance *= exp(-fog.density * stepSize);
        if(fog.enableGodrays == 1 && fog.godrayLightIndex < scene.light_direct_count) {
            vec3 sunDir = normalize(-lights_direct[fog.godrayLightIndex].direction);
            float shadow = inShadow(p, sunDir, maxDist);
            inscatter += (1.0 - shadow) * lights_direct[fog.godrayLightIndex].color * fog.density * stepSize * transmittance;
        } else {
            inscatter += fog.color * fog.density * stepSize * transmittance;
        }
    }
    return hitColor * transmittance + inscatter;
}

vec3 getSkyColor(vec3 rd) {
    float t = 0.5 * (rd.y + 1.0);
    return mix(vec3(0.7, 0.8, 0.9), vec3(0.2, 0.3, 0.5), t);
}

void main() {
    float reflectMultiplier, refractMultiplier, tm;
    vec3 mask = vec3(1.0), color = vec3(0.0);
    vec3 ro = vec3(scene.camera_pos), rd = getRayDir();
    int type = 0, num, tri_hit; hit_record hr; vec3 bar_hit;
    
    for(int i = 0; i < ITERATIONS; i++) {
        tm = calcInter(ro, rd, num, type, tri_hit, bar_hit);
        if(tm < maxDist) {
            vec3 pt = ro + rd*tm; hr = get_hit_info(ro, rd, pt, tm, num, type, tri_hit, bar_hit);
            if(type == TYPE_POINT_LIGHT) { color += lights_point[num].color * mask; break; }
            rt_material mat = hr.mat; vec3 n = hr.normal;
            bool outside = dot(rd, n) < 0; n = outside ? n : -n;
            reflectMultiplier = getFresnel(n, rd, mat.reflection);
            refractMultiplier = 1 - reflectMultiplier;

            vec3 localColor = calcShade(pt + n * hr.bias_mult, rd, mat, n, true);
            localColor = applyFog(ro, rd, tm, localColor);

            if(mat.refraction > 1.0) {
                float eta = outside ? (1.0 / mat.refraction) : mat.refraction;
                vec3 refrDir = refract(rd, n, eta);
                color += localColor * reflectMultiplier * mask * 0.15;
                if (dot(refrDir, refrDir) > 0.0) { ro = pt - n * hr.bias_mult; rd = refrDir; mask *= refractMultiplier; }
                else { ro = pt + n * hr.bias_mult; rd = reflect(rd, n); mask *= reflectMultiplier; }
            } else if(mat.reflection > 0.0) {
                ro = pt + n * hr.bias_mult; color += localColor * refractMultiplier * mask;
                rd = reflect(rd, n); mask *= reflectMultiplier;
            } else { color += localColor * mask * hr.alpha; break; }
        } else {
            vec3 skyColor = getSkyColor(rd);
            skyColor = applyFog(ro, rd, maxDist, skyColor);
            color += skyColor * mask; break;
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
GLuint sphereUbo = 0, boxUbo = 0, planeUbo = 0, waterUbo = 0;
GLuint lightPointUbo = 0, lightSpotUbo = 0, lightAreaUbo = 0, lightDirectUbo = 0;
GLuint meshUbo = 0, fogUbo = 0;

std::vector<rt_sphere> spheres;
std::vector<rt_plane> planes;
std::vector<rt_box> boxes;
std::vector<rt_water> waters;
std::vector<rt_light_point> lights_point;
std::vector<rt_light_spot> lights_spot;
std::vector<rt_light_area> lights_area;
std::vector<rt_light_direct> lights_direct;
std::vector<rt_triangle> mesh_triangles;
rt_scene scene = {};

int wind_width = 800;
int wind_height = 600;

float yaw = 0.0f, pitch = 0.0f;
Vec3 camera_pos(0, 0, -5);
bool cursorCaptured = true;
double lastX = 640, lastY = 360;
bool firstMouse = true;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        cursorCaptured = !cursorCaptured;
        glfwSetInputMode(window, GLFW_CURSOR, cursorCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        firstMouse = true;
    }
}

void framebufferSizeCallback(GLFWwindow*, int w, int h) {
    if (w < 1 || h < 1) return;
    wind_width = w; wind_height = h;
    scene.canvas_width = w; scene.canvas_height = h;
    glViewport(0, 0, w, h);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!cursorCaptured) return;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = (xpos - lastX) * 0.05f; float yoffset = (lastY - ypos) * 0.05f;
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
    auto init_buffer = [&](GLuint* ubo, const char* name, int bindingPoint, size_t size, void* data) {
        glGenBuffers(1, ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, *ubo);
        glBufferData(GL_UNIFORM_BUFFER, size, data, GL_DYNAMIC_DRAW);
        GLuint blockIndex = glGetUniformBlockIndex(shaderProgram, name);
        glUniformBlockBinding(shaderProgram, blockIndex, bindingPoint);
        glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, *ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    };
    
    init_buffer(&sceneUbo, "scene_buf", 0, sizeof(rt_scene), NULL);
    init_buffer(&sphereUbo, "spheres_buf", 1, sizeof(rt_sphere) * MAX_SPHERES, NULL);
    init_buffer(&planeUbo, "planes_buf", 2, sizeof(rt_plane) * MAX_PLANES, NULL);
    init_buffer(&boxUbo, "boxes_buf", 3, sizeof(rt_box) * MAX_BOXES, NULL);
    init_buffer(&waterUbo, "waters_buf", 4, sizeof(rt_water) * MAX_WATERS, NULL);
    init_buffer(&lightPointUbo, "lights_point_buf", 5, sizeof(rt_light_point) * MAX_LIGHTS_POINT, NULL);
    init_buffer(&lightSpotUbo, "lights_spot_buf", 6, sizeof(rt_light_spot) * MAX_LIGHTS_SPOT, NULL);
    init_buffer(&lightAreaUbo, "lights_area_buf", 7, sizeof(rt_light_area) * MAX_LIGHTS_AREA, NULL);
    init_buffer(&lightDirectUbo, "lights_direct_buf", 8, sizeof(rt_light_direct) * MAX_LIGHTS_DIRECT, NULL);
    init_buffer(&meshUbo, "mesh_buf", 9, sizeof(rt_triangle) * MAX_MESH_TRIANGLES, NULL);
    init_buffer(&fogUbo, "fog_buf", 10, sizeof(rt_fog), NULL);
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main() {
    if (!glfwInit()) return -1;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(wind_width, wind_height, "Raytracer Scenegraph", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    scene.canvas_width = wind_width;
    scene.canvas_height = wind_height;
    scene.reflect_depth = 2;
    scene.quat_camera_rotation[3] = 1.0f;

    SceneNode* root = new SceneNode();
    
    SunLightNode* sun = new SunLightNode(Vec3(1, 0.9, 0.8), 1.5);
    sun->rotate(-45, 0);
    root->addChild(sun);

    FogNode* fog = new FogNode(0.03f, true, 0, Vec3(1, 0.9, 0.8));
    root->addChild(fog);

    WaterNode* water = new WaterNode();
    water->setposition(0, -2, 6);
    water->scale(1, 1, 1);
    water->colorize(0.1, 0.3, 0.6);
    root->addChild(water);

    AreaLightNode* areaLight = new AreaLightNode(Vec3(1, 0.5, 0.2), 5.0, 4.0, 2.0);
    areaLight->setposition(-3, 3, 6);
    areaLight->rotate(0, 45);
    root->addChild(areaLight);

    SpotLightNode* spotLight = new SpotLightNode(Vec3(0.2, 1, 0.5), 10.0, cos(radians(12.5)), cos(radians(17.5)));
    spotLight->setposition(4, 3, 6);
    spotLight->rotate(45, 0);
    root->addChild(spotLight);

    PointLightNode* pointLight = new PointLightNode(Vec3(1, 0, 0), 5.0);
    pointLight->setposition(0, 2, 6);
    root->addChild(pointLight);

    SphereNode* sph1 = new SphereNode(1.0, create_material(0, 0, 1, 50, 0.35, 0.0f, 1.0f));
    sph1->setposition(2, 0, 6);
    root->addChild(sph1);

    SphereNode* sph2 = new SphereNode(1.0, create_material(1, 0, 0, 100, 0.1, 1.125f, 1.0f));
    sph2->setposition(-1, 0, 6);
    root->addChild(sph2);

    BoxNode* bx1 = new BoxNode(1, 1, 1, create_material(0.8, 0.7, 0, 50, 0.0, 0.0f, 1.0f));
    bx1->setposition(2, -1, 8);
    bx1->scale(2, 2, 2);
    root->addChild(bx1);

    // Built-in OBJ Cube to demonstrate Bump Mapping
    std::string cubeObj = R"(
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0
vn  0.0  0.0  1.0
vn  0.0  0.0 -1.0
vn  0.0 -1.0  0.0
vn  0.0  1.0  0.0
vn -1.0  0.0  0.0
vn  1.0  0.0  0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1/1 2/2/1 3/3/1 4/4/1
f 6/1/2 5/2/2 8/3/2 7/4/2
f 5/1/3 6/2/3 2/3/3 1/4/3
f 4/1/4 3/2/4 7/3/4 8/4/4
f 5/1/5 1/2/5 4/3/5 8/4/5
f 2/1/6 6/2/6 7/3/6 3/4/6
)";
    MeshNode* mesh1 = new MeshNode(cubeObj);
    mesh1->setposition(8, 0, 6);
    mesh1->scale(1.5, 1.5, 1.5);
    mesh1->cull_distance = 10.0f; // Example: If you move >50 units away, the cube is dropped from UBO
    root->addChild(mesh1);

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
    glDeleteShader(vs); glDeleteShader(fs);

    init_buffers();

    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f, -1.0f, -1.0f,  0.0f, 0.0f,  1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,  1.0f, -1.0f,  1.0f, 0.0f,  1.0f,  1.0f,  1.0f, 1.0f
    };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    glUseProgram(shaderProgram);

    double lastFrameTime = glfwGetTime();
    double fpsTimer = 0;
    int frames = 0;
    const float cameraSpeed = 3.0f; 

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents(); 

        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        if (deltaTime > 0.1) deltaTime = 0.1; 

        Quat q = quatFromEuler(radians(-pitch), radians(yaw));
        Vec3 front = normalize(quatRotate(q, Vec3(0, 0, 1)));
        Vec3 right = normalize(cross(front, Vec3(0, 1, 0)));
        if (right.x == 0.0f && right.y == 0.0f && right.z == 0.0f) right = Vec3(1, 0, 0);
        Vec3 up = normalize(cross(right, front));

        float speedMul = 1.0f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) speedMul = 3.0f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS) speedMul = 0.25f;
        float speed = (float)deltaTime * cameraSpeed * speedMul;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera_pos += front * speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera_pos -= front * speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera_pos -= right * speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera_pos += right * speed;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera_pos += up * speed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) camera_pos -= up * speed;

        scene.camera_pos[0] = camera_pos.x; scene.camera_pos[1] = camera_pos.y; scene.camera_pos[2] = camera_pos.z;
        scene.quat_camera_rotation[0] = q.x; scene.quat_camera_rotation[1] = q.y; scene.quat_camera_rotation[2] = q.z; scene.quat_camera_rotation[3] = q.w;

        float t = (float)glfwGetTime();

        sph1->setposition(2.0f + 2.0f * sinf(t), 0, 6.0f + 2.0f * cosf(t));
        sph2->scale(1.0f + 0.3f * sinf(t * 3.0f), 1.0f + 0.3f * sinf(t * 3.0f), 1.0f + 0.3f * sinf(t * 3.0f));
        bx1->rotate(0, 45.0f * (float)deltaTime);
        mesh1->rotate(0, 30.0f * (float)deltaTime); // Rotate Mesh to show bump mapping
        water->setposition(0, -2.0f + sinf(t) * 0.5f, 6);

        spheres.clear(); boxes.clear(); planes.clear(); waters.clear();
        lights_point.clear(); lights_spot.clear(); lights_area.clear(); lights_direct.clear();
        mesh_triangles.clear();
        
        Vec3 pPos(0,0,0); Quat pRot(0,0,0,1); Vec3 pScl(1,1,1); Vec3 pTint(1,1,1);
        root->traverse(spheres, boxes, planes, waters, lights_point, lights_spot, lights_area, lights_direct, mesh_triangles, pPos, pRot, pScl, pTint, camera_pos);

        scene.sphere_count = spheres.size();
        scene.box_count = boxes.size();
        scene.plane_count = planes.size();
        scene.water_count = waters.size();
        scene.light_point_count = lights_point.size();
        scene.light_spot_count = lights_spot.size();
        scene.light_area_count = lights_area.size();
        scene.light_direct_count = lights_direct.size();
        scene.mesh_count = mesh_triangles.size();
        scene.time = t;

        glBindBuffer(GL_UNIFORM_BUFFER, sceneUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_scene), &scene);

        glBindBuffer(GL_UNIFORM_BUFFER, sphereUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_sphere) * spheres.size(), spheres.data());

        glBindBuffer(GL_UNIFORM_BUFFER, boxUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_box) * boxes.size(), boxes.data());

        glBindBuffer(GL_UNIFORM_BUFFER, waterUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_water) * waters.size(), waters.data());

        glBindBuffer(GL_UNIFORM_BUFFER, lightPointUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_light_point) * lights_point.size(), lights_point.data());

        glBindBuffer(GL_UNIFORM_BUFFER, lightSpotUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_light_spot) * lights_spot.size(), lights_spot.data());

        glBindBuffer(GL_UNIFORM_BUFFER, lightAreaUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_light_area) * lights_area.size(), lights_area.data());

        glBindBuffer(GL_UNIFORM_BUFFER, lightDirectUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_light_direct) * lights_direct.size(), lights_direct.data());

        glBindBuffer(GL_UNIFORM_BUFFER, meshUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_triangle) * mesh_triangles.size(), mesh_triangles.data());

        glBindBuffer(GL_UNIFORM_BUFFER, fogUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(rt_fog), &global_fog);

        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        usleep(50000);
        frames++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0) {
            std::string title = "Raytracer Scenegraph  " + std::to_string(frames) + " FPS";
            glfwSetWindowTitle(window, title.c_str());
            frames = 0;
            fpsTimer = 0;
        }
    }

    delete root;
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}