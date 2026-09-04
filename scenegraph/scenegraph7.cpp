// ============================================================================
// Advanced Scenegraph Demo: Deferred Rendering Version
// ============================================================================
// Compile (Windows): g++ main.cpp -o demo -lglfw3 -lglew32 -lopengl32
// Compile (Linux):   g++ main.cpp -o demo -lglfw -lGLEW -lGL
//
// Features:
// 1. Deferred Rendering Pipeline (G-Buffer + Light Pass)
// 2. stb_image integration for universal texture loading (PNG, JPG, etc.)
// 3. Hardware Mesh Instancing (glDrawArraysInstanced)
// 4. Per-Instance Frustum Culling
// 5. Advanced Material System (Diffuse, Specular, Shininess)
// 6. Scene Node Color Tinting (multiplies with texture)
// 7. Comprehensive Light Nodes: Sun, Point, Spot, and 3 types of Area Lights!
// 8. FLASHLIGHT DEMO: Camera-attached spotlight toggled with 'F' (FIXED!)
//
// FIXES APPLIED:
// - Corrected the camera view matrix math. The original code used an orbiting 
//   transform (trans * rotY * rotX) which caused the flashlight to drift in 
//   view-space as you moved. It is now a standard First-Person matrix.
// - Completely refactored to a Deferred Renderer using an MRT (Multiple Render 
//   Targets) G-Buffer to handle lighting efficiently on a full-screen quad.
//
// NOTE: Ensure stb_image.h is in the same directory as this file.
// ============================================================================

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cstdio>
#include <cstring>
#include <algorithm> // For std::min, std::remove

// ==========================================
// GLOBAL STATE 
// ==========================================
GLuint g_gBufferShader = 0;
GLuint g_lightPassShader = 0;
GLuint g_activeShader = 0; // Tracks the active shader program during rendering

GLFWwindow* g_window = nullptr;
float g_cameraPosition[3] = {0.0f, 5.0f, 25.0f}; 
float g_cameraRotation[3] = {-15.0f, 0.0f, 0.0f}; 
float g_speed = 0.15f;

// --- FLASHLIGHT MOD: Global pointer to our flashlight ---
class LightNode;
LightNode* g_flashlight = nullptr;

// --- G-BUFFER & DEFERRED STATE ---
GLuint g_gBufferFBO;
GLuint g_positionTexture, g_normalTexture, g_albedoTexture, g_specShinTexture;
GLuint g_depthTexture;
GLuint g_quadVAO, g_quadVBO;

// ==========================================
// 1. MATH UTILITIES
// ==========================================

struct Vec2 {
    float x, y;
    Vec2() : x(0), y(0) {}
    Vec2(float X, float Y) : x(X), y(Y) {}
};

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
    
    float length() const { return sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const {
        float l = length();
        if (l > 0) return Vec3(x/l, y/l, z/l);
        return Vec3();
    }
    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const {
        return Vec3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x);
    }
};

struct Mat4 {
    float m[16];
    Mat4() { identity(); }
    void identity() {
        memset(m, 0, sizeof(float) * 16);
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }
    
    static Mat4 perspective(float fov, float aspect, float near, float far) {
        Mat4 r;
        float f = 1.0f / tan(fov * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (far + near) / (near - far);
        r.m[11] = -1.0f;
        r.m[14] = (2.0f * far * near) / (near - far);
        r.m[15] = 0.0f;
        return r;
    }

    static Mat4 translate(float x, float y, float z) {
        Mat4 r;
        r.m[12] = x; r.m[13] = y; r.m[14] = z;
        return r;
    }

    static Mat4 rotateX(float angle) {
        Mat4 r; float c = cos(angle), s = sin(angle);
        r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c;
        return r;
    }

    static Mat4 rotateY(float angle) {
        Mat4 r; float c = cos(angle), s = sin(angle);
        r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c;
        return r;
    }

    static Mat4 rotateZ(float angle) {
        Mat4 r; float c = cos(angle), s = sin(angle);
        r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c;
        return r;
    }

    static Mat4 scale(float x, float y, float z) {
        Mat4 r;
        r.m[0] = x; r.m[5] = y; r.m[10] = z;
        return r;
    }

    Mat4 operator*(const Mat4& b) const {
        Mat4 res;
        for(int r=0; r<4; ++r) {
            for(int c=0; c<4; ++c) {
                res.m[c*4+r] = 0;
                for(int k=0; k<4; ++k) {
                    res.m[c*4+r] += m[k*4+r] * b.m[c*4+k];
                }
            }
        }
        return res;
    }

    Vec3 transformPoint(const Vec3& p) const {
        return Vec3(
            m[0]*p.x + m[4]*p.y + m[8]*p.z + m[12],
            m[1]*p.x + m[5]*p.y + m[9]*p.z + m[13],
            m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14]
        );
    }
};

struct BoundingBox {
    Vec3 min;
    Vec3 max;
    BoundingBox() : min(1e10f, 1e10f, 1e10f), max(-1e10f, -1e10f, -1e10f) {}
    
    void expand(const Vec3& p) {
        if (p.x < min.x) min.x = p.x;
        if (p.y < min.y) min.y = p.y;
        if (p.z < min.z) min.z = p.z;
        if (p.x > max.x) max.x = p.x;
        if (p.y > max.y) max.y = p.y;
        if (p.z > max.z) max.z = p.z;
    }

    BoundingBox transform(const Mat4& m) const {
        BoundingBox result;
        Vec3 corners[8] = {
            Vec3(min.x, min.y, min.z), Vec3(max.x, min.y, min.z),
            Vec3(min.x, max.y, min.z), Vec3(max.x, max.y, min.z),
            Vec3(min.x, min.y, max.z), Vec3(max.x, min.y, max.z),
            Vec3(min.x, max.y, max.z), Vec3(max.x, max.y, max.z)
        };
        for(int i=0; i<8; ++i) {
            result.expand(m.transformPoint(corners[i]));
        }
        return result;
    }
};

struct Plane {
    float a, b, c, d;
    float distance(const Vec3& p) const { return a * p.x + b * p.y + c * p.z + d; }
    void normalize() {
        float len = sqrt(a*a + b*b + c*c);
        if (len > 0) { a/=len; b/=len; c/=len; d/=len; }
    }
};

struct Frustum {
    Plane planes[6];
    void extract(const Mat4& vp) {
        planes[0] = {vp.m[3] - vp.m[0], vp.m[7] - vp.m[4], vp.m[11] - vp.m[8], vp.m[15] - vp.m[12]};
        planes[1] = {vp.m[3] + vp.m[0], vp.m[7] + vp.m[4], vp.m[11] + vp.m[8], vp.m[15] + vp.m[12]};
        planes[2] = {vp.m[3] + vp.m[1], vp.m[7] + vp.m[5], vp.m[11] + vp.m[9], vp.m[15] + vp.m[13]};
        planes[3] = {vp.m[3] - vp.m[1], vp.m[7] - vp.m[5], vp.m[11] - vp.m[9], vp.m[15] - vp.m[13]};
        planes[4] = {vp.m[3] - vp.m[2], vp.m[7] - vp.m[6], vp.m[11] - vp.m[10], vp.m[15] - vp.m[14]};
        planes[5] = {vp.m[3] + vp.m[2], vp.m[7] + vp.m[6], vp.m[11] + vp.m[10], vp.m[15] + vp.m[14]};
        for(int i=0; i<6; ++i) planes[i].normalize();
    }

    bool intersects(const BoundingBox& aabb) const {
        for (int i=0; i<6; ++i) {
            Vec3 pVertex;
            pVertex.x = (planes[i].a >= 0.0f) ? aabb.max.x : aabb.min.x;
            pVertex.y = (planes[i].b >= 0.0f) ? aabb.max.y : aabb.min.y;
            pVertex.z = (planes[i].c >= 0.0f) ? aabb.max.z : aabb.min.z;
            if (planes[i].distance(pVertex) < 0.0f) return false;
        }
        return true;
    }
};

void getNormalMatrix(const Mat4& model, float* out9) {
    float a00 = model.m[0], a01 = model.m[1], a02 = model.m[2];
    float a10 = model.m[4], a11 = model.m[5], a12 = model.m[6];
    float a20 = model.m[8], a21 = model.m[9], a22 = model.m[10];

    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;

    float det = a00 * b01 + a01 * b11 + a02 * b21;
    if (det == 0) det = 1e-6f;
    det = 1.0f / det;

    out9[0] = b01 * det; out9[1] = (-a22 * a01 + a02 * a21) * det; out9[2] = (a12 * a01 - a02 * a11) * det;
    out9[3] = b11 * det; out9[4] = (a22 * a00 - a02 * a20) * det; out9[5] = (-a12 * a00 + a02 * a10) * det;
    out9[6] = b21 * det; out9[7] = (-a21 * a00 + a01 * a20) * det; out9[8] = (a11 * a00 - a01 * a10) * det;
}

// ==========================================
// 2. UNIVERSAL TEXTURE LOADER (stb_image)
// ==========================================
GLuint loadTexture(const std::string& filename) {
    GLuint texID = 0;
    glGenTextures(1, &texID);
    
    int width, height, nrChannels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrChannels, 0);
    
    if (!data) {
        std::cout << "Failed to load texture: " << filename << ". Generating checkerboard fallback.\n";
        unsigned char checker[64*64*4];
        for(int y=0; y<64; ++y) {
            for(int x=0; x<64; ++x) {
                bool c = ((x/8 + y/8) % 2) == 0;
                checker[(y*64+x)*4+0] = c ? 255 : 150;
                checker[(y*64+x)*4+1] = c ? 255 : 150;
                checker[(y*64+x)*4+2] = c ? 255 : 150;
                checker[(y*64+x)*4+3] = 255;
            }
        }
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, checker);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        return texID;
    }

    GLenum format = GL_RGB;
    if (nrChannels == 1)      format = GL_RED;
    else if (nrChannels == 3) format = GL_RGB;
    else if (nrChannels == 4) format = GL_RGBA;

    glBindTexture(GL_TEXTURE_2D, texID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
    std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ", " << nrChannels << " channels)\n";
    return texID;
}

// ==========================================
// 3. OBJ / MTL LOADER
// ==========================================
struct OBJMaterial {
    std::string name;
    Vec3 ambient, diffuse, specular;
    float shininess;
    std::string diffuseMap;
    GLuint textureID;

    OBJMaterial() : shininess(0.0f), textureID(0) {
        ambient   = Vec3(0.2f, 0.2f, 0.2f);
        diffuse   = Vec3(0.8f, 0.8f, 0.8f);
        specular  = Vec3(0.0f, 0.0f, 0.0f);
    }
    
    void loadTexture() {
        if (!diffuseMap.empty() && textureID == 0) {
            textureID = ::loadTexture(diffuseMap);
        }
    }
};

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 texcoord;
};

struct Mesh {
    std::vector<Vertex> vertices;
    OBJMaterial* material;
    Mesh() : material(0) {}
};

class OBJModel {
public:
    std::vector<Mesh> meshes;
    std::map<std::string, OBJMaterial> materials;

    bool load(const std::string& filename) {
        std::ifstream file(filename.c_str());
        if (!file) return false;

        std::vector<Vec3> positions;
        std::vector<Vec3> normals;
        std::vector<Vec2> texcoords;

        Mesh* currentMesh = 0;
        std::string line;

        size_t pos = filename.find_last_of("/");
        std::string basePath = (pos != std::string::npos) ? filename.substr(0, pos + 1) : "";

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix;

            if (prefix == "mtllib") {
                std::string mtl;
                ss >> mtl;
                loadMTL(basePath + mtl);
            }
            else if (prefix == "usemtl") {
                std::string name;
                ss >> name;
                meshes.push_back(Mesh());
                currentMesh = &meshes.back();
                currentMesh->material = &materials[name];
            }
            else if (prefix == "v") {
                Vec3 v; ss >> v.x >> v.y >> v.z;
                positions.push_back(v);
            }
            else if (prefix == "vn") {
                Vec3 n; ss >> n.x >> n.y >> n.z;
                normals.push_back(n);
            }
            else if (prefix == "vt") {
                Vec2 t; ss >> t.x >> t.y;
                texcoords.push_back(t);
            }
            else if (prefix == "f" && currentMesh) {
                std::string v[4];
                int count = 0;
                while (ss >> v[count]) count++;
                for (int i = 1; i + 1 < count; ++i) {
                    parseVertex(v[0], positions, texcoords, normals, *currentMesh);
                    parseVertex(v[i], positions, texcoords, normals, *currentMesh);
                    parseVertex(v[i+1], positions, texcoords, normals, *currentMesh);
                }
            }
        }
        return true;
    }

private:
    void loadMTL(const std::string& filename) {
        std::ifstream file(filename.c_str());
        if (!file) return;

        OBJMaterial* current = 0;
        std::string line;

        size_t pos = filename.find_last_of("/");
        std::string basePath = (pos != std::string::npos) ? filename.substr(0, pos + 1) : "";

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix;

            if (prefix == "newmtl") {
                std::string name;
                ss >> name;
                materials[name] = OBJMaterial();
                materials[name].name = name;
                current = &materials[name];
            }
            else if (prefix == "Ka" && current) ss >> current->ambient.x >> current->ambient.y >> current->ambient.z;
            else if (prefix == "Kd" && current) ss >> current->diffuse.x >> current->diffuse.y >> current->diffuse.z;
            else if (prefix == "Ks" && current) ss >> current->specular.x >> current->specular.y >> current->specular.z;
            else if (prefix == "Ns" && current) ss >> current->shininess;
            else if (prefix == "map_Kd" && current) {
                ss >> current->diffuseMap;
                current->diffuseMap = basePath + current->diffuseMap;
            }
        }
    }

    void parseVertex(const std::string& token, const std::vector<Vec3>& positions,
                     const std::vector<Vec2>& texcoords, const std::vector<Vec3>& normals,
                     Mesh& mesh) {
        Vertex vert;
        int vi = 0, ti = 0, ni = 0;
        
        size_t slash1 = token.find('/');
        if (slash1 == std::string::npos) {
            vi = std::stoi(token);
        } else {
            vi = std::stoi(token.substr(0, slash1));
            size_t slash2 = token.find('/', slash1 + 1);
            if (slash2 == std::string::npos) {
                ti = std::stoi(token.substr(slash1 + 1));
            } else {
                std::string vtStr = token.substr(slash1 + 1, slash2 - slash1 - 1);
                if (!vtStr.empty()) ti = std::stoi(vtStr);
                std::string vnStr = token.substr(slash2 + 1);
                if (!vnStr.empty()) ni = std::stoi(vnStr);
            }
        }

        vert.position = (vi > 0 && vi <= (int)positions.size()) ? positions[vi - 1] : Vec3();
        vert.texcoord = (ti > 0 && ti <= (int)texcoords.size()) ? texcoords[ti - 1] : Vec2();
        vert.texcoord.y = 1.0f - vert.texcoord.y;
        vert.normal   = (ni > 0 && ni <= (int)normals.size()) ? normals[ni - 1] : Vec3(0,0,1);
        mesh.vertices.push_back(vert);
    }
};

// ==========================================
// 4. MATERIALS & LIGHTING
// ==========================================
struct Material {
    Vec3 ambient = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 diffuse = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 specular = Vec3(1.0f, 1.0f, 1.0f);
    float shininess = 32.0f;
};

struct DirLight {
    Vec3 direction = Vec3(-0.5f, -1.0f, -0.3f);
    Vec3 ambient = Vec3(0.2f, 0.2f, 0.2f);
    Vec3 diffuse = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 specular = Vec3(1.0f, 1.0f, 1.0f);
};

DirLight g_dirLight;

// ==========================================
// 5. SCENEGRAPH & RENDERABLE COMPONENTS
// ==========================================

struct GLMesh {
    GLuint vao, vbo;
    int vertexCount;
    BoundingBox localAABB;
    
    GLMesh() : vao(0), vbo(0), vertexCount(0) {}
    ~GLMesh() {
        if (vao) glDeleteVertexArrays(1, &vao);
        if (vbo) glDeleteBuffers(1, &vbo);
    }
    
    void upload(const std::vector<Vertex>& verts, const OBJMaterial* mat) {
        for (const auto& v : verts) localAABB.expand(v.position);
        vertexCount = verts.size();

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        glBindVertexArray(0);
    }

    void setupTextureAndMaterial(const Material& mat, GLuint texID, bool hasTex) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hasTex ? texID : 0);
        
        GLint loc;
        loc = glGetUniformLocation(g_activeShader, "uTexture"); if(loc != -1) glUniform1i(loc, 0);
        loc = glGetUniformLocation(g_activeShader, "uHasTexture"); if(loc != -1) glUniform1i(loc, hasTex ? 1 : 0);
        
        loc = glGetUniformLocation(g_activeShader, "uMaterial.ambient"); if(loc != -1) glUniform3f(loc, mat.ambient.x, mat.ambient.y, mat.ambient.z);
        loc = glGetUniformLocation(g_activeShader, "uMaterial.diffuse"); if(loc != -1) glUniform3f(loc, mat.diffuse.x, mat.diffuse.y, mat.diffuse.z);
        loc = glGetUniformLocation(g_activeShader, "uMaterial.specular"); if(loc != -1) glUniform3f(loc, mat.specular.x, mat.specular.y, mat.specular.z);
        loc = glGetUniformLocation(g_activeShader, "uMaterial.shininess"); if(loc != -1) glUniform1f(loc, mat.shininess);
    }

    void draw(const Material& mat, GLuint texID, bool hasTex) {
        if (vao) {
            glBindVertexArray(vao);
            setupTextureAndMaterial(mat, texID, hasTex);
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
            glBindVertexArray(0);
        }
    }
    
    void drawInstanced(const Material& mat, GLuint texID, bool hasTex, int count) {
        if (vao) {
            glBindVertexArray(vao);
            setupTextureAndMaterial(mat, texID, hasTex);
            glDrawArraysInstanced(GL_TRIANGLES, 0, vertexCount, count);
            glBindVertexArray(0);
        }
    }
};

class Renderable {
public:
    std::vector<GLMesh*> meshes;
    std::vector<GLuint> textures;
    std::vector<bool> hasTextures;
    BoundingBox localAABB;
    
    ~Renderable() {
        for (auto* m : meshes) delete m;
    }
    
    void addMesh(GLMesh* m, GLuint tex, bool hasTex) {
        meshes.push_back(m);
        textures.push_back(tex);
        hasTextures.push_back(hasTex);
        localAABB.expand(m->localAABB.min);
        localAABB.expand(m->localAABB.max);
    }
    
    void draw(const Material& mat) {
        for (size_t i = 0; i < meshes.size(); ++i) {
            meshes[i]->draw(mat, textures[i], hasTextures[i]);
        }
    }
    
    void drawInstanced(const Material& mat, int count) {
        for (size_t i = 0; i < meshes.size(); ++i) {
            meshes[i]->drawInstanced(mat, textures[i], hasTextures[i], count);
        }
    }
};

void generateBox(std::vector<Vertex>& verts) {
    float h = 0.5f;
    auto addQuad = [&](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 n) {
        verts.push_back({p0, n, Vec2(0,0)});
        verts.push_back({p1, n, Vec2(1,0)});
        verts.push_back({p2, n, Vec2(1,1)});
        verts.push_back({p0, n, Vec2(0,0)});
        verts.push_back({p2, n, Vec2(1,1)});
        verts.push_back({p3, n, Vec2(0,1)});
    };
    addQuad(Vec3(-h,-h, h), Vec3( h,-h, h), Vec3( h, h, h), Vec3(-h, h, h), Vec3( 0, 0, 1));
    addQuad(Vec3( h,-h,-h), Vec3(-h,-h,-h), Vec3(-h, h,-h), Vec3( h, h,-h), Vec3( 0, 0,-1));
    addQuad(Vec3(-h, h, h), Vec3( h, h, h), Vec3( h, h,-h), Vec3(-h, h,-h), Vec3( 0, 1, 0));
    addQuad(Vec3(-h,-h,-h), Vec3( h,-h,-h), Vec3( h,-h, h), Vec3(-h,-h, h), Vec3( 0,-1, 0));
    addQuad(Vec3( h,-h, h), Vec3( h,-h,-h), Vec3( h, h,-h), Vec3( h, h, h), Vec3( 1, 0, 0));
    addQuad(Vec3(-h,-h,-h), Vec3(-h,-h, h), Vec3(-h, h, h), Vec3(-h, h,-h), Vec3(-1, 0, 0));
}

void generatePlane(std::vector<Vertex>& verts) {
    float h = 0.5f;
    Vec3 n(0, 0, -1);
    verts.push_back({Vec3(-h, -h, 0), n, Vec2(0,0)});
    verts.push_back({Vec3( h, -h, 0), n, Vec2(1,0)});
    verts.push_back({Vec3( h,  h, 0), n, Vec2(1,1)});
    
    verts.push_back({Vec3(-h, -h, 0), n, Vec2(0,0)});
    verts.push_back({Vec3( h,  h, 0), n, Vec2(1,1)});
    verts.push_back({Vec3(-h,  h, 0), n, Vec2(0,1)});
}

void loadModelIntoRenderable(Renderable* r, const std::string& path) {
    OBJModel model;
    if (model.load(path) && !model.meshes.empty()) {
        for (auto& mesh : model.meshes) {
            if (mesh.material) mesh.material->loadTexture();
            GLMesh* glmesh = new GLMesh();
            glmesh->upload(mesh.vertices, mesh.material);
            
            GLuint tex = 0; bool hasTex = false;
            if (mesh.material && mesh.material->textureID != 0) {
                tex = mesh.material->textureID; hasTex = true;
            }
            r->addMesh(glmesh, tex, hasTex);
        }
        std::cout << "Loaded OBJ: " << path << " (" << model.meshes.size() << " meshes)\n";
    } else {
        std::cout << "Failed to load OBJ: " << path << ". Using fallback box.\n";
        std::vector<Vertex> verts;
        generateBox(verts);
        GLMesh* boxMesh = new GLMesh();
        boxMesh->upload(verts, nullptr);
        GLuint tex = loadTexture("fallback_checkerboard");
        r->addMesh(boxMesh, tex, true);
    }
}

// ==========================================
// 6. SCENE NODES
// ==========================================

class SceneNode {
public:
    SceneNode* parent = nullptr;
    std::vector<SceneNode*> children;

    Vec3 position, rotation, scale;
    Renderable* renderable = nullptr;
    bool ownsRenderable = false;

    Mat4 localMatrix, worldMatrix;
    BoundingBox worldAABB;
    bool visible = true;
    
    Vec3 color = Vec3(1.0f, 1.0f, 1.0f);
    Material material;
    
    bool isLightMesh = false;
    Vec3 lightColor = Vec3(1,1,1);

    SceneNode() { scale = Vec3(1, 1, 1); }
    virtual ~SceneNode() {
        for (auto* c : children) delete c;
        if (ownsRenderable && renderable) delete renderable;
    }

    void setRenderable(Renderable* r, bool own = false) { renderable = r; ownsRenderable = own; }
    void addChild(SceneNode* child) { child->parent = this; children.push_back(child); }

    void updateLocalMatrix() {
        Mat4 trans = Mat4::translate(position.x, position.y, position.z);
        Mat4 rotX = Mat4::rotateX(rotation.x * 3.14159265f / 180.0f);
        Mat4 rotY = Mat4::rotateY(rotation.y * 3.14159265f / 180.0f);
        Mat4 rotZ = Mat4::rotateZ(rotation.z * 3.14159265f / 180.0f);
        Mat4 scl = Mat4::scale(scale.x, scale.y, scale.z);
        localMatrix = trans * rotY * rotX * rotZ * scl;
    }

    virtual void updateWorldTransform(const Mat4& parentWorld) {
        updateLocalMatrix();
        worldMatrix = parentWorld * localMatrix;
        for (auto* child : children) child->updateWorldTransform(worldMatrix);

        if (renderable) {
            worldAABB = renderable->localAABB.transform(worldMatrix);
        } else {
            worldAABB = BoundingBox(); 
            for(auto* child : children) {
                worldAABB.expand(child->worldAABB.min);
                worldAABB.expand(child->worldAABB.max);
            }
        }
    }

    void draw(const Frustum& frustum, int& cullCount, int& drawCount) {
        if (!visible) return;

        if (renderable || !children.empty()) {
            if (!frustum.intersects(worldAABB)) {
                cullCount += countNodes();
                return; 
            }
        }

        if (renderable) {
            GLint loc;
            loc = glGetUniformLocation(g_activeShader, "uIsInstanced"); if(loc != -1) glUniform1i(loc, 0);
            loc = glGetUniformLocation(g_activeShader, "uModel"); if(loc != -1) glUniformMatrix4fv(loc, 1, GL_FALSE, worldMatrix.m);
            
            float normalMat[9];
            getNormalMatrix(worldMatrix, normalMat);
            loc = glGetUniformLocation(g_activeShader, "uNormalMatrix"); if(loc != -1) glUniformMatrix3fv(loc, 1, GL_FALSE, normalMat);

            loc = glGetUniformLocation(g_activeShader, "uInstanceColors[0]"); if(loc != -1) glUniform3fv(loc, 1, (float*)&color);
            
            loc = glGetUniformLocation(g_activeShader, "uIsLightMesh"); if(loc != -1) glUniform1i(loc, isLightMesh ? 1 : 0);
            loc = glGetUniformLocation(g_activeShader, "uLightColor"); if(loc != -1) glUniform3f(loc, lightColor.x, lightColor.y, lightColor.z);

            renderable->draw(material);
            drawCount++;
        }

        for (auto* child : children) child->draw(frustum, cullCount, drawCount);
    }
    
    int countNodes() {
        int c = 1;
        for (auto* child : children) c += child->countNodes();
        return c;
    }
};

struct InstanceData {
    Mat4 matrix;
    Vec3 color;
};

class InstancedSceneNode {
public:
    Renderable* renderable;
    std::vector<InstanceData> instances;
    Material material;

    bool isLightMesh = false;
    Vec3 lightColor = Vec3(1,1,1);

    void draw(const Frustum& frustum, int& cullCount, int& drawCount) {
        std::vector<InstanceData> visible;
        for (const auto& inst : instances) {
            BoundingBox worldAABB = renderable->localAABB.transform(inst.matrix);
            if (frustum.intersects(worldAABB)) {
                visible.push_back(inst);
            } else {
                cullCount++;
            }
        }

        if (visible.empty()) return;

        std::vector<float> matrices;
        std::vector<float> colors;
        matrices.reserve(visible.size() * 16);
        colors.reserve(visible.size() * 3);

        for (const auto& inst : visible) {
            for(int i=0; i<16; ++i) matrices.push_back(inst.matrix.m[i]);
            colors.push_back(inst.color.x);
            colors.push_back(inst.color.y);
            colors.push_back(inst.color.z);
        }

        GLint locMats = glGetUniformLocation(g_activeShader, "uInstanceMatrices");
        GLint locCols = glGetUniformLocation(g_activeShader, "uInstanceColors");
        GLint loc;
        
        loc = glGetUniformLocation(g_activeShader, "uIsInstanced"); if(loc != -1) glUniform1i(loc, 1);
        loc = glGetUniformLocation(g_activeShader, "uIsLightMesh"); if(loc != -1) glUniform1i(loc, isLightMesh ? 1 : 0);
        loc = glGetUniformLocation(g_activeShader, "uLightColor"); if(loc != -1) glUniform3f(loc, lightColor.x, lightColor.y, lightColor.z);
        
        if(locMats != -1) glUniformMatrix4fv(locMats, visible.size(), GL_FALSE, matrices.data());
        if(locCols != -1) glUniform3fv(locCols, visible.size(), colors.data());

        renderable->drawInstanced(material, visible.size());
        drawCount += visible.size();
    }
};

// ==========================================
// 7. LIGHT SYSTEM NODES
// ==========================================

enum LightType {
    LIGHT_SUN,
    LIGHT_POINT,
    LIGHT_SPOT,
    LIGHT_AREA_SPHERE,
    LIGHT_AREA_RECT,
    LIGHT_AREA_GRID
};

struct LightData {
    LightType type;
    Vec3 position;
    Vec3 direction;
    Vec3 right;
    Vec3 up;
    Vec3 color;
    float intensity;
    float radius;
    float width, height;
    float cutOff, outerCutOff;
    float constant, linear, quadratic;
};

class LightNode;

std::vector<LightNode*> g_lightNodes;

class LightNode : public SceneNode {
public:
    LightData data;
    SceneNode* visualNode = nullptr;

    LightNode(LightType t) {
        data.type = t;
        data.color = Vec3(1, 1, 1);
        data.intensity = 20.0f;
        data.radius = 1.0f;
        data.width = 2.0f;
        data.height = 2.0f;
        data.cutOff = cos(15.0f * 3.14159265f / 180.0f);
        data.outerCutOff = cos(20.0f * 3.14159265f / 180.0f);
        data.constant = 1.0f;
        data.linear = 0.09f;
        data.quadratic = 0.032f;
        g_lightNodes.push_back(this);
    }
    
    ~LightNode() {
        g_lightNodes.erase(std::remove(g_lightNodes.begin(), g_lightNodes.end(), this), g_lightNodes.end());
    }

    void setupVisuals(Renderable* cube, Renderable* plane) {
        visualNode = new SceneNode();
        visualNode->isLightMesh = true;
        visualNode->ownsRenderable = false;
        
        if (data.type == LIGHT_POINT || data.type == LIGHT_SUN) {
            visualNode->setRenderable(cube);
            visualNode->scale = Vec3(0.3f, 0.3f, 0.3f);
        } else if (data.type == LIGHT_SPOT) {
            visualNode->setRenderable(cube);
            visualNode->scale = Vec3(0.4f, 0.4f, 0.6f); 
        } else if (data.type == LIGHT_AREA_RECT || data.type == LIGHT_AREA_GRID) {
            visualNode->setRenderable(plane);
            visualNode->scale = Vec3(data.width, data.height, 1.0f);
        } else if (data.type == LIGHT_AREA_SPHERE) {
            visualNode->setRenderable(cube);
            visualNode->scale = Vec3(data.width, data.width, data.width);
        }
        this->addChild(visualNode);
    }
    
    void updateWorldTransform(const Mat4& parentWorld) override {
        SceneNode::updateWorldTransform(parentWorld);
        
        data.position = worldMatrix.transformPoint(Vec3(0,0,0));
        
        Vec3 fwdLocal(0, 0, -1);
        Vec3 rightLocal(1, 0, 0);
        Vec3 upLocal(0, 1, 0);

        Vec3 fwdWorld = worldMatrix.transformPoint(fwdLocal) - data.position;
        Vec3 rightWorld = worldMatrix.transformPoint(rightLocal) - data.position;
        Vec3 upWorld = worldMatrix.transformPoint(upLocal) - data.position;

        data.direction = fwdWorld.normalized();
        data.right = rightWorld.normalized();
        data.up = upWorld.normalized();

        if(visualNode) {
            visualNode->lightColor = data.color * (data.intensity * 0.1f); 
        }
    }
};

// ==========================================
// 8. SHADERS (DEFERRED RENDERING VERSION)
// ==========================================

const char* gBufferVertexSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoord;

    uniform mat4 uView, uProj;
    uniform mat3 uNormalMatrix;
    uniform mat4 uModel; 
    
    uniform bool uIsInstanced;
    uniform mat4 uInstanceMatrices[128];
    uniform vec3 uInstanceColors[128];

    out vec3 vFragPos;
    out vec3 vNormal;
    out vec2 vTexCoord;
    out vec3 vColor;

    void main() {
        mat4 modelMat;
        mat3 normMat;
        
        if (uIsInstanced) {
            modelMat = uInstanceMatrices[gl_InstanceID];
            normMat = mat3(modelMat); 
            vColor = uInstanceColors[gl_InstanceID];
        } else {
            modelMat = uModel;
            normMat = uNormalMatrix;
            vColor = uInstanceColors[0]; 
        }

        vec4 worldPos = modelMat * vec4(aPos, 1.0);
        vFragPos = worldPos.xyz;
        vNormal = normMat * aNormal;
        vTexCoord = aTexCoord;
        
        gl_Position = uProj * uView * worldPos;
    }
)";

const char* gBufferFragmentSource = R"(
    #version 330 core
    in vec3 vFragPos;
    in vec3 vNormal;
    in vec2 vTexCoord;
    in vec3 vColor;

    layout (location = 0) out vec3 gPosition;
    layout (location = 1) out vec3 gNormal;
    layout (location = 2) out vec4 gAlbedo; // RGB: Albedo, A: Flag (1.0 = normal, 0.0 = light mesh)
    layout (location = 3) out vec4 gSpecShin; // RGB: Specular, A: Shininess

    uniform bool uIsLightMesh;
    uniform vec3 uLightColor;

    struct Material {
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
    };
    uniform Material uMaterial;
    uniform sampler2D uTexture;
    uniform bool uHasTexture;

    void main() {
        gPosition = vFragPos;
        gNormal = normalize(vNormal);
        
        if (uIsLightMesh) {
            gAlbedo = vec4(uLightColor, 0.0); // A = 0.0 indicates this is a light mesh
            gSpecShin = vec4(0.0);
            return;
        }

        vec3 texColor = vec3(1.0);
        if (uHasTexture) texColor = texture(uTexture, vTexCoord).rgb;
        
        gAlbedo = vec4(vColor * uMaterial.diffuse * texColor, 1.0); // A = 1.0 indicates normal geometry
        gSpecShin = vec4(uMaterial.specular, uMaterial.shininess / 255.0);
    }
)";

const char* lightPassVertexSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;
    out vec2 vTexCoord;
    void main() {
        vTexCoord = aTexCoords;
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); 
    }
)";

const char* lightPassFragmentSource = R"(
    #version 330 core
    in vec2 vTexCoord;
    out vec4 FragColor;

    uniform sampler2D gPosition;
    uniform sampler2D gNormal;
    uniform sampler2D gAlbedo;
    uniform sampler2D gSpecShin;

    uniform vec3 uViewPos;

    struct DirLight { vec3 direction; vec3 ambient; vec3 diffuse; vec3 specular; };
    uniform DirLight uDirLight;

    struct PointLight { 
        vec3 position; vec3 color; float intensity; 
        float constant; float linear; float quadratic; 
    };
    uniform PointLight uPointLights[8];
    uniform int uNumPointLights;

    struct SpotLight { 
        vec3 position; vec3 direction; vec3 color; float intensity; 
        float cutOff; float outerCutOff; float constant; float linear; float quadratic; 
    };
    uniform SpotLight uSpotLights[4];
    uniform int uNumSpotLights;

    struct AreaLight { 
        vec3 position; vec3 direction; vec3 right; vec3 up; vec3 color; 
        float intensity; float width; float height; int type; 
    };
    uniform AreaLight uAreaLights[4];
    uniform int uNumAreaLights;

    vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec3 baseColor, vec3 specColor, float shininess) {
        vec3 lightDir = normalize(-light.direction);
        vec3 ambient = light.ambient * baseColor;
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = light.diffuse * diff * baseColor;
        vec3 reflectDir = reflect(-lightDir, normal);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = light.specular * spec * specColor;
        return (ambient + diffuse + specular);
    }

    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 baseColor, vec3 specColor, float shininess) {
        vec3 lightDir = light.position - fragPos;
        float distance = length(lightDir);
        lightDir = normalize(lightDir);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = light.color * diff * baseColor * attenuation * light.intensity;
        vec3 reflectDir = reflect(-lightDir, normal);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = light.color * spec * specColor * attenuation * light.intensity;
        return (diffuse + specular);
    }

    vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 baseColor, vec3 specColor, float shininess) {
        vec3 lightDir = light.position - fragPos;
        float distance = length(lightDir);
        lightDir = normalize(lightDir);
        float theta = dot(lightDir, normalize(-light.direction)); 
        float epsilon = light.cutOff - light.outerCutOff;
        float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
        attenuation *= intensity;
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = light.color * diff * baseColor * attenuation * light.intensity;
        vec3 reflectDir = reflect(-lightDir, normal);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = light.color * spec * specColor * attenuation * light.intensity;
        return (diffuse + specular);
    }

    vec3 CalcSphereAreaLight(AreaLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 baseColor, vec3 specColor, float shininess) {
        vec3 L = light.position - fragPos;
        float distance = max(length(L), 0.1); 
        vec3 lightDir = normalize(L);
        float diff = max(dot(normal, lightDir), 0.0);
        if (diff <= 0.0) return vec3(0.0);
        float attenuation = 1.0 / (distance * distance + 0.01);
        vec3 diffuse = light.color * diff * baseColor * attenuation * light.intensity;
        float radius = light.width * 0.5;
        vec3 R = reflect(-viewDir, normal);
        vec3 centerToRay = dot(L, R) * R - L;
        float distToCenter = length(centerToRay);
        vec3 closestPoint = L + centerToRay * clamp(radius / max(distToCenter, 0.001), 0.0, 1.0);
        vec3 specularVec = normalize(closestPoint);
        float spec = pow(max(dot(R, specularVec), 0.0), shininess);
        vec3 specular = light.color * spec * specColor * attenuation * light.intensity;
        return diffuse + specular;
    }

    vec3 CalcRectAreaLight(AreaLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 baseColor, vec3 specColor, float shininess) {
        vec3 L = light.position - fragPos;
        float distance = max(length(L), 0.1);
        vec3 lightDir = normalize(L);
        float diff = max(dot(normal, lightDir), 0.0);
        if (diff <= 0.0) return vec3(0.0);
        float attenuation = 1.0 / (distance * distance + 0.01);
        vec3 diffuse = light.color * diff * baseColor * attenuation * light.intensity;
        vec3 R = reflect(-viewDir, normal);
        vec3 closestPoint = light.position; 
        float denom = dot(R, light.direction);
        if (abs(denom) > 0.0001) {
            float t = dot(light.position - fragPos, light.direction) / denom;
            if (t > 0.0) {
                vec3 intersectPt = fragPos + R * t;
                vec3 localPt = intersectPt - light.position;
                float x = clamp(dot(localPt, light.right), -light.width * 0.5, light.width * 0.5);
                float y = clamp(dot(localPt, light.up), -light.height * 0.5, light.height * 0.5);
                closestPoint = light.position + light.right * x + light.up * y;
            }
        }
        vec3 L_prime = normalize(closestPoint - fragPos);
        vec3 R_prime = reflect(-L_prime, normal);
        float spec = pow(max(dot(viewDir, R_prime), 0.0), shininess);
        vec3 specular = light.color * spec * specColor * attenuation * light.intensity;
        return diffuse + specular;
    }

    void main() {
        vec4 albedoFlag = texture(gAlbedo, vTexCoord);
        // Alpha == 0.0 means this fragment is a light mesh (visual representation of a light)
        if (albedoFlag.a < 0.5) {
            FragColor = vec4(albedoFlag.rgb, 1.0);
            return;
        }

        vec3 fragPos = texture(gPosition, vTexCoord).rgb;
        vec3 normal = texture(gNormal, vTexCoord).rgb;
        vec3 albedo = albedoFlag.rgb;
        vec4 specShin = texture(gSpecShin, vTexCoord);
        vec3 specColor = specShin.rgb;
        float shininess = max(specShin.a * 255.0, 1.0);

        vec3 viewDir = normalize(uViewPos - fragPos);
        vec3 result = albedo * 0.05; 

        result += CalcDirLight(uDirLight, normal, viewDir, albedo, specColor, shininess);

        for(int i = 0; i < uNumPointLights; i++)
            result += CalcPointLight(uPointLights[i], normal, fragPos, viewDir, albedo, specColor, shininess);

        for(int i = 0; i < uNumSpotLights; i++)
            result += CalcSpotLight(uSpotLights[i], normal, fragPos, viewDir, albedo, specColor, shininess);

        for(int i = 0; i < uNumAreaLights; i++) {
            if (uAreaLights[i].type == 0) 
                result += CalcSphereAreaLight(uAreaLights[i], normal, fragPos, viewDir, albedo, specColor, shininess);
            else if (uAreaLights[i].type == 1) 
                result += CalcRectAreaLight(uAreaLights[i], normal, fragPos, viewDir, albedo, specColor, shininess);
        }

        FragColor = vec4(result, 1.0);
    }
)";

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
    }
    return shader;
}

void setupShaders() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, gBufferVertexSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, gBufferFragmentSource);
    g_gBufferShader = glCreateProgram();
    glAttachShader(g_gBufferShader, vs);
    glAttachShader(g_gBufferShader, fs);
    glLinkProgram(g_gBufferShader);
    glDeleteShader(vs);
    glDeleteShader(fs);

    vs = compileShader(GL_VERTEX_SHADER, lightPassVertexSource);
    fs = compileShader(GL_FRAGMENT_SHADER, lightPassFragmentSource);
    g_lightPassShader = glCreateProgram();
    glAttachShader(g_lightPassShader, vs);
    glAttachShader(g_lightPassShader, fs);
    glLinkProgram(g_lightPassShader);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

void setupGBuffer() {
    glGenFramebuffers(1, &g_gBufferFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, g_gBufferFBO);

    // Position texture (RGB32F)
    glGenTextures(1, &g_positionTexture);
    glBindTexture(GL_TEXTURE_2D, g_positionTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 800, 600, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_positionTexture, 0);

    // Normal texture (RGB16F)
    glGenTextures(1, &g_normalTexture);
    glBindTexture(GL_TEXTURE_2D, g_normalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 800, 600, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g_normalTexture, 0);

    // Albedo + Flag texture (RGBA8)
    glGenTextures(1, &g_albedoTexture);
    glBindTexture(GL_TEXTURE_2D, g_albedoTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, g_albedoTexture, 0);

    // Specular + Shininess texture (RGBA8)
    glGenTextures(1, &g_specShinTexture);
    glBindTexture(GL_TEXTURE_2D, g_specShinTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, g_specShinTexture, 0);

    GLuint attachments[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
    glDrawBuffers(4, attachments);

    // Depth buffer
    glGenRenderbuffers(1, &g_depthTexture);
    glBindRenderbuffer(GL_RENDERBUFFER, g_depthTexture);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 800, 600);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_depthTexture);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
        
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void setupQuad() {
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &g_quadVAO);
    glGenBuffers(1, &g_quadVBO);
    glBindVertexArray(g_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
}

// ==========================================
// 9. CORE ENGINE LOOP
// ==========================================

std::vector<InstancedSceneNode*> g_instancedNodes;

void drawScene(SceneNode* root) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Mat4 proj = Mat4::perspective(3.14159265f / 3.0f, 800.0f / 600.0f, 0.1f, 200.0f);
    float pitchRad = g_cameraRotation[0] * 3.14159265f / 180.0f;
    float yawRad = g_cameraRotation[1] * 3.14159265f / 180.0f;

    Mat4 rotX = Mat4::rotateX(-pitchRad);
    Mat4 rotY = Mat4::rotateY(-yawRad);
    Mat4 trans = Mat4::translate(-g_cameraPosition[0], -g_cameraPosition[1], -g_cameraPosition[2]);
    
    // FIX: Standard first-person camera view matrix.
    // The original code had (trans * rotY * rotX) which incorrectly rotated around the world origin
    // instead of the camera origin. This caused the flashlight to drift when moving.
    Mat4 view = rotX * rotY * trans; 
    Mat4 vp = proj * view;

    Frustum frustum;
    frustum.extract(vp);

    Mat4 identity;
    root->updateWorldTransform(identity);

    std::vector<LightData> pts;
    std::vector<LightData> spots;
    std::vector<LightData> areas;

    for (auto* ln : g_lightNodes) {
        if (!ln->visible) continue;

        if (ln->data.type == LIGHT_SUN) {
            g_dirLight.direction = ln->data.direction;
            g_dirLight.diffuse = ln->data.color * ln->data.intensity;
            g_dirLight.specular = ln->data.color * ln->data.intensity;
            g_dirLight.ambient = ln->data.color * 0.05f;
        }
        else if (ln->data.type == LIGHT_POINT) {
            pts.push_back(ln->data);
        }
        else if (ln->data.type == LIGHT_SPOT) {
            spots.push_back(ln->data);
        }
        else if (ln->data.type == LIGHT_AREA_GRID) {
            float halfW = ln->data.width * 0.5f;
            float halfH = ln->data.height * 0.5f;
            Vec3 offsets[4] = {
                Vec3(-halfW, -halfH, 0), Vec3(halfW, -halfH, 0),
                Vec3(-halfW,  halfH, 0), Vec3(halfW,  halfH, 0)
            };
            for(int i=0; i<4; ++i) {
                LightData ld = ln->data;
                ld.type = LIGHT_POINT; 
                ld.intensity = ln->data.intensity;
                ld.position = ln->worldMatrix.transformPoint(offsets[i]);
                pts.push_back(ld);
            }
        }
        else {
            areas.push_back(ln->data);
        }
    }

    // ==========================================
    // 1. GEOMETRY PASS (G-Buffer)
    // ==========================================
    glBindFramebuffer(GL_FRAMEBUFFER, g_gBufferFBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    g_activeShader = g_gBufferShader;
    glUseProgram(g_activeShader);

    GLint loc;
    loc = glGetUniformLocation(g_activeShader, "uView"); if(loc != -1) glUniformMatrix4fv(loc, 1, GL_FALSE, view.m);
    loc = glGetUniformLocation(g_activeShader, "uProj"); if(loc != -1) glUniformMatrix4fv(loc, 1, GL_FALSE, proj.m);

    int cullCount = 0, drawCount = 0;
    root->draw(frustum, cullCount, drawCount);

    for (auto* instNode : g_instancedNodes) {
        instNode->draw(frustum, cullCount, drawCount);
    }

    // ==========================================
    // 2. LIGHTING PASS
    // ==========================================
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    g_activeShader = g_lightPassShader;
    glUseProgram(g_activeShader);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, g_positionTexture);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, g_normalTexture);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, g_albedoTexture);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, g_specShinTexture);
    
    glUniform1i(glGetUniformLocation(g_activeShader, "gPosition"), 0);
    glUniform1i(glGetUniformLocation(g_activeShader, "gNormal"), 1);
    glUniform1i(glGetUniformLocation(g_activeShader, "gAlbedo"), 2);
    glUniform1i(glGetUniformLocation(g_activeShader, "gSpecShin"), 3);
    
    glUniform3f(glGetUniformLocation(g_activeShader, "uViewPos"), g_cameraPosition[0], g_cameraPosition[1], g_cameraPosition[2]);

    // Upload Global Light
    glUniform3f(glGetUniformLocation(g_activeShader, "uDirLight.direction"), g_dirLight.direction.x, g_dirLight.direction.y, g_dirLight.direction.z);
    glUniform3f(glGetUniformLocation(g_activeShader, "uDirLight.ambient"), g_dirLight.ambient.x, g_dirLight.ambient.y, g_dirLight.ambient.z);
    glUniform3f(glGetUniformLocation(g_activeShader, "uDirLight.diffuse"), g_dirLight.diffuse.x, g_dirLight.diffuse.y, g_dirLight.diffuse.z);
    glUniform3f(glGetUniformLocation(g_activeShader, "uDirLight.specular"), g_dirLight.specular.x, g_dirLight.specular.y, g_dirLight.specular.z);

    // Upload Point Lights
    int pCount = std::min((int)pts.size(), 8);
    glUniform1i(glGetUniformLocation(g_activeShader, "uNumPointLights"), pCount);
    for(int i=0; i<pCount; ++i) {
        std::string base = "uPointLights[" + std::to_string(i) + "].";
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"position").c_str()), pts[i].position.x, pts[i].position.y, pts[i].position.z);
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"color").c_str()), pts[i].color.x, pts[i].color.y, pts[i].color.z);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"intensity").c_str()), pts[i].intensity);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"constant").c_str()), pts[i].constant);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"linear").c_str()), pts[i].linear);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"quadratic").c_str()), pts[i].quadratic);
    }

    // Upload Spot Lights
    int sCount = std::min((int)spots.size(), 4);
    glUniform1i(glGetUniformLocation(g_activeShader, "uNumSpotLights"), sCount);
    for(int i=0; i<sCount; ++i) {
        std::string base = "uSpotLights[" + std::to_string(i) + "].";
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"position").c_str()), spots[i].position.x, spots[i].position.y, spots[i].position.z);
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"direction").c_str()), spots[i].direction.x, spots[i].direction.y, spots[i].direction.z);
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"color").c_str()), spots[i].color.x, spots[i].color.y, spots[i].color.z);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"intensity").c_str()), spots[i].intensity);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"cutOff").c_str()), spots[i].cutOff);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"outerCutOff").c_str()), spots[i].outerCutOff);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"constant").c_str()), spots[i].constant);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"linear").c_str()), spots[i].linear);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"quadratic").c_str()), spots[i].quadratic);
    }

    // Upload Area Lights
    int aCount = std::min((int)areas.size(), 4);
    glUniform1i(glGetUniformLocation(g_activeShader, "uNumAreaLights"), aCount);
    for(int i=0; i<aCount; ++i) {
        std::string base = "uAreaLights[" + std::to_string(i) + "].";
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"position").c_str()), areas[i].position.x, areas[i].position.y, areas[i].position.z);
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"direction").c_str()), areas[i].direction.x, areas[i].direction.y, areas[i].direction.z);
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"right").c_str()), areas[i].right.x, areas[i].right.y, areas[i].right.z);
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"up").c_str()), areas[i].up.x, areas[i].up.y, areas[i].up.z);
        glUniform3f(glGetUniformLocation(g_activeShader, (base+"color").c_str()), areas[i].color.x, areas[i].color.y, areas[i].color.z);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"intensity").c_str()), areas[i].intensity);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"width").c_str()), areas[i].width);
        glUniform1f(glGetUniformLocation(g_activeShader, (base+"height").c_str()), areas[i].height);
        glUniform1i(glGetUniformLocation(g_activeShader, (base+"type").c_str()), areas[i].type == LIGHT_AREA_SPHERE ? 0 : 1);
    }

    // Draw Screen Quad
    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    std::string title = "Deferred Scenegraph | Drawn: " + std::to_string(drawCount) + " | Culled: " + std::to_string(cullCount) + " | Press 'F' for Flashlight";
    glfwSetWindowTitle(g_window, title.c_str());
    glfwSwapBuffers(g_window);
}

void handleInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    static double lastX = 800.0 / 2.0, lastY = 600.0 / 2.0;
    double currentX, currentY;
    glfwGetCursorPos(window, &currentX, &currentY);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        g_cameraRotation[1] += (currentX - lastX) * 0.2f; 
        g_cameraRotation[0] += (lastY - currentY) * 0.2f; 
        if (g_cameraRotation[0] > 89.0f) g_cameraRotation[0] = 89.0f;
        if (g_cameraRotation[0] < -89.0f) g_cameraRotation[0] = -89.0f;
    }
    lastX = currentX; lastY = currentY;

    float yawRad = g_cameraRotation[1] * 3.14159265f / 180.0f;

    static bool fWasPressed = false;
    bool fIsPressed = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
    if (g_flashlight && fIsPressed && !fWasPressed) {
        g_flashlight->visible = !g_flashlight->visible;
        std::cout << "Flashlight turned " << (g_flashlight->visible ? "ON" : "OFF") << "\n";
    }
    fWasPressed = fIsPressed;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { g_cameraPosition[0] -= sin(yawRad) * g_speed; g_cameraPosition[2] -= cos(yawRad) * g_speed; }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { g_cameraPosition[0] += sin(yawRad) * g_speed; g_cameraPosition[2] += cos(yawRad) * g_speed; }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { g_cameraPosition[0] -= cos(yawRad) * g_speed; g_cameraPosition[2] += sin(yawRad) * g_speed; }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { g_cameraPosition[0] += cos(yawRad) * g_speed; g_cameraPosition[2] -= sin(yawRad) * g_speed; }
}

int main() {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); 
    #endif

    g_window = glfwCreateWindow(800, 600, "Deferred Scenegraph & Lighting", nullptr, nullptr);
    if (!g_window) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(g_window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
    glfwSwapInterval(1);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

    setupShaders();
    setupGBuffer();
    setupQuad();

    Renderable* boxRenderable = new Renderable();
    {
        std::vector<Vertex> boxVerts;
        generateBox(boxVerts);
        GLMesh* boxMesh = new GLMesh();
        boxMesh->upload(boxVerts, nullptr);
        GLuint boxTex = loadTexture("fallback_checkerboard");
        boxRenderable->addMesh(boxMesh, boxTex, true);
    }

    Renderable* objRenderable = new Renderable();
    loadModelIntoRenderable(objRenderable, "capsule.obj");

    Renderable* glowCube = new Renderable();
    {
        std::vector<Vertex> boxVerts;
        generateBox(boxVerts);
        GLMesh* boxMesh = new GLMesh();
        boxMesh->upload(boxVerts, nullptr);
        glowCube->addMesh(boxMesh, 0, false);
    }

    Renderable* glowPlane = new Renderable();
    {
        std::vector<Vertex> planeVerts;
        generatePlane(planeVerts);
        GLMesh* planeMesh = new GLMesh();
        planeMesh->upload(planeVerts, nullptr);
        glowPlane->addMesh(planeMesh, 0, false);
    }

    SceneNode* root = new SceneNode();

    LightNode* sun = new LightNode(LIGHT_SUN);
    sun->rotation = Vec3(-45, 30, 0);
    sun->data.color = Vec3(0.8f, 0.9f, 1.0f);
    sun->data.intensity = 0.05f;
    sun->setupVisuals(glowCube, glowPlane);
    root->addChild(sun);

    LightNode* pointLight = new LightNode(LIGHT_POINT);
    pointLight->position = Vec3(-5, 3, -5);
    pointLight->data.color = Vec3(1, 0.2, 0.2);
    pointLight->data.intensity = 50.0f;
    pointLight->setupVisuals(glowCube, glowPlane);
    root->addChild(pointLight);

    LightNode* spotLight = new LightNode(LIGHT_SPOT);
    spotLight->position = Vec3(5, 6, -2);
    spotLight->rotation = Vec3(-45, 0, 0);
    spotLight->data.color = Vec3(0.2, 1, 0.2);
    spotLight->data.intensity = 80.0f;
    spotLight->setupVisuals(glowCube, glowPlane);
    root->addChild(spotLight);

    LightNode* areaSphere = new LightNode(LIGHT_AREA_SPHERE);
    areaSphere->position = Vec3(-8, 4, 5);
    areaSphere->data.color = Vec3(0.2, 0.4, 1.0);
    areaSphere->data.intensity = 80.0f;
    areaSphere->data.width = 3.0f; 
    areaSphere->setupVisuals(glowCube, glowPlane);
    root->addChild(areaSphere);

    LightNode* areaRect = new LightNode(LIGHT_AREA_RECT);
    areaRect->position = Vec3(8, 4, 5);
    areaRect->rotation = Vec3(-30, -20, 0);
    areaRect->data.color = Vec3(1.0, 0.8, 0.2);
    areaRect->data.intensity = 100.0f;
    areaRect->data.width = 6.0f;
    areaRect->data.height = 2.0f;
    areaRect->setupVisuals(glowCube, glowPlane);
    root->addChild(areaRect);

    LightNode* areaGrid = new LightNode(LIGHT_AREA_GRID);
    areaGrid->position = Vec3(0, 6, 10);
    areaGrid->rotation = Vec3(-45, 0, 0);
    areaGrid->data.color = Vec3(1.0, 0.2, 1.0);
    areaGrid->data.intensity = 25.0f; 
    areaGrid->data.width = 5.0f;
    areaGrid->data.height = 5.0f;
    areaGrid->setupVisuals(glowCube, glowPlane);
    root->addChild(areaGrid);

    g_flashlight = new LightNode(LIGHT_SPOT);
    g_flashlight->data.color = Vec3(1.0f, 1.0f, 0.9f);
    g_flashlight->data.intensity = 150.0f;
    g_flashlight->data.cutOff = cos(12.5f * 3.14159265f / 180.0f);
    g_flashlight->data.outerCutOff = cos(15.0f * 3.14159265f / 180.0f);
    g_flashlight->data.constant = 1.0f;
    g_flashlight->data.linear = 0.09f;
    g_flashlight->data.quadratic = 0.032f;
    g_flashlight->setupVisuals(glowCube, glowPlane);
    if(g_flashlight->visualNode) {
        g_flashlight->visualNode->position = Vec3(0.0f, -0.3f, -1.0f); 
        g_flashlight->visualNode->scale = Vec3(0.1f, 0.1f, 0.3f);
    }
    root->addChild(g_flashlight);

    SceneNode* spinGroup = new SceneNode();
    spinGroup->position = Vec3(0, 0, -5);
    root->addChild(spinGroup);

    SceneNode* objNode = new SceneNode();
    objNode->position = Vec3(0, 2, 0);
    objNode->color = Vec3(0.1f, 1.0f, 0.2f); 
    objNode->material.diffuse = Vec3(1.0f, 1.0f, 1.0f);
    objNode->material.specular = Vec3(0.8f, 0.8f, 0.8f);
    objNode->material.shininess = 64.0f;
    objNode->setRenderable(objRenderable);
    spinGroup->addChild(objNode);

    InstancedSceneNode* grid = new InstancedSceneNode();
    grid->renderable = boxRenderable;
    grid->material.diffuse = Vec3(1.0f, 1.0f, 1.0f);
    grid->material.specular = Vec3(0.5f, 0.5f, 0.5f);
    grid->material.shininess = 32.0f;

    int gridSizeX = 12, gridSizeZ = 10; 
    for(int x=0; x<gridSizeX; ++x) {
        for(int z=0; z<gridSizeZ; ++z) {
            InstanceData inst;
            float px = (x - gridSizeX/2) * 2.5f + 15.0f;
            float pz = (z - gridSizeZ/2) * 2.5f - 20.0f;
            float py = sin(px * 0.3f) * cos(pz * 0.3f) * 3.0f; 
            
            inst.matrix = Mat4::translate(px, py, pz) * Mat4::scale(1.0f, 1.0f + abs(py) * 0.5f, 1.0f);
            
            float t = (py + 3.0f) / 6.0f;
            inst.color = Vec3(0.2f + t * 0.4f, 0.8f, 0.2f); 
            
            grid->instances.push_back(inst);
        }
    }
    g_instancedNodes.push_back(grid);

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(g_window)) {
        double currentTime = glfwGetTime();
        float dt = currentTime - lastTime;
        lastTime = currentTime;

        handleInput(g_window);

        if (g_flashlight) {
            g_flashlight->position = Vec3(g_cameraPosition[0], g_cameraPosition[1], g_cameraPosition[2]);
            g_flashlight->rotation = Vec3(g_cameraRotation[0], g_cameraRotation[1], 0.0f);
        }
        
        spinGroup->rotation.y += 20.0f * dt;
        objNode->rotation.x += 10.0f * dt;
        objNode->position.y = 2.0f + sin(currentTime * 2.0f) * 0.5f; 

        drawScene(root);
        glfwPollEvents();
    }

    delete root;
    delete boxRenderable;
    delete objRenderable;
    delete glowCube;
    delete glowPlane;
    for(auto* in : g_instancedNodes) delete in;
    
    glfwTerminate();
    return 0;
}