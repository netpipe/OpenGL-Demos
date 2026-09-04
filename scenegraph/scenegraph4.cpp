// ============================================================================
// Advanced Scenegraph Demo: Instancing, Materials, and Lighting
// ============================================================================
// Compile (Windows): g++ main.cpp -o demo -lglfw3 -lglew32 -lopengl32
// Compile (Linux):   g++ main.cpp -o demo -lglfw -lGLEW -lGL
//
// Features:
// 1. stb_image integration for universal texture loading (PNG, JPG, etc.)
// 2. Hardware Mesh Instancing (glDrawArraysInstanced)
// 3. Per-Instance Frustum Culling
// 4. Advanced Material System (Diffuse, Specular, Shininess)
// 5. Global Directional Light
// 6. Scene Node Color Tinting (multiplies with texture)
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

// ==========================================
// GLOBAL STATE 
// ==========================================
GLuint g_shaderProgram = 0;
GLFWwindow* g_window = nullptr;
float g_cameraPosition[3] = {0.0f, 5.0f, 25.0f}; 
float g_cameraRotation[3] = {-15.0f, 0.0f, 0.0f}; 
float g_speed = 0.15f;

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

        size_t pos = filename.find_last_of("/\\");
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

        size_t pos = filename.find_last_of("/\\");
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
        
        glUniform1i(glGetUniformLocation(g_shaderProgram, "uTexture"), 0);
        glUniform1i(glGetUniformLocation(g_shaderProgram, "uHasTexture"), hasTex ? 1 : 0);
        
        glUniform3f(glGetUniformLocation(g_shaderProgram, "uMaterial.ambient"), mat.ambient.x, mat.ambient.y, mat.ambient.z);
        glUniform3f(glGetUniformLocation(g_shaderProgram, "uMaterial.diffuse"), mat.diffuse.x, mat.diffuse.y, mat.diffuse.z);
        glUniform3f(glGetUniformLocation(g_shaderProgram, "uMaterial.specular"), mat.specular.x, mat.specular.y, mat.specular.z);
        glUniform1f(glGetUniformLocation(g_shaderProgram, "uMaterial.shininess"), mat.shininess);
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
    
    // Custom properties for the advanced system
    Vec3 color = Vec3(1.0f, 1.0f, 1.0f); // Tints the texture
    Material material;

    SceneNode() { scale = Vec3(1, 1, 1); }
    ~SceneNode() {
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

    void updateWorldTransform(const Mat4& parentWorld) {
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
            glUniform1i(glGetUniformLocation(g_shaderProgram, "uIsInstanced"), 0);
            glUniformMatrix4fv(glGetUniformLocation(g_shaderProgram, "uModel"), 1, GL_FALSE, worldMatrix.m);
            
            float normalMat[9];
            getNormalMatrix(worldMatrix, normalMat);
            glUniformMatrix3fv(glGetUniformLocation(g_shaderProgram, "uNormalMatrix"), 1, GL_FALSE, normalMat);

            // Pass node color to index 0 of the color array so the shader uses it
            glUniform3fv(glGetUniformLocation(g_shaderProgram, "uInstanceColors[0]"), 1, (float*)&color);

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

        // Prepare uniform data
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

        GLint locMats = glGetUniformLocation(g_shaderProgram, "uInstanceMatrices");
        GLint locCols = glGetUniformLocation(g_shaderProgram, "uInstanceColors");
        
        glUniform1i(glGetUniformLocation(g_shaderProgram, "uIsInstanced"), 1);
        glUniformMatrix4fv(locMats, visible.size(), GL_FALSE, matrices.data());
        glUniformMatrix4fv(locMats, visible.size(), GL_FALSE, matrices.data());
        glUniform3fv(locCols, visible.size(), colors.data()); // Removed GL_FALSE

        renderable->drawInstanced(material, visible.size());
        drawCount += visible.size();
    }
};

// ==========================================
// 7. SHADERS
// ==========================================

const char* vertexShaderSource = R"(
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

    out vec3 vNormal, vFragPos, vColor;
    out vec2 vTexCoord;

    void main() {
        mat4 modelMat;
        mat3 normMat;
        
        if (uIsInstanced) {
            modelMat = uInstanceMatrices[gl_InstanceID];
            normMat = mat3(modelMat); // Assumes uniform scale for instancing
            vColor = uInstanceColors[gl_InstanceID];
        } else {
            modelMat = uModel;
            normMat = uNormalMatrix;
            vColor = uInstanceColors[0]; // Base color passed via index 0
        }

        vec4 worldPos = modelMat * vec4(aPos, 1.0);
        vFragPos = worldPos.xyz;
        vNormal = normMat * aNormal;
        vTexCoord = aTexCoord;
        
        gl_Position = uProj * uView * worldPos;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 vNormal, vFragPos, vColor;
    in vec2 vTexCoord;
    out vec4 FragColor;

    struct Material {
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
    };
    uniform Material uMaterial;
    uniform sampler2D uTexture;
    uniform bool uHasTexture;

    struct DirLight {
        vec3 direction;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
    };
    uniform DirLight uDirLight;
    uniform vec3 uViewPos;

    void main() {
        vec3 norm = normalize(vNormal);
        vec3 lightDir = normalize(-uDirLight.direction);
        
        // Tinting: Node Color (vColor) multiplies Material/Texture
        vec3 texColor = vec3(1.0);
        if (uHasTexture) texColor = texture(uTexture, vTexCoord).rgb;
        
        vec3 baseColor = vColor * uMaterial.diffuse * texColor;
        
        // Ambient
        vec3 ambient = uDirLight.ambient * baseColor;
        
        // Diffuse
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = uDirLight.diffuse * diff * baseColor;
        
        // Specular
        vec3 viewDir = normalize(uViewPos - vFragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), uMaterial.shininess);
        vec3 specular = uDirLight.specular * spec * uMaterial.specular;
        
        FragColor = vec4(ambient + diffuse + specular, 1.0);
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
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    g_shaderProgram = glCreateProgram();
    glAttachShader(g_shaderProgram, vs);
    glAttachShader(g_shaderProgram, fs);
    glLinkProgram(g_shaderProgram);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

// ==========================================
// 8. CORE ENGINE LOOP
// ==========================================

std::vector<InstancedSceneNode*> g_instancedNodes;

void drawScene(SceneNode* root) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(g_shaderProgram);

    Mat4 proj = Mat4::perspective(3.14159265f / 3.0f, 800.0f / 600.0f, 0.1f, 200.0f);
    float pitchRad = g_cameraRotation[0] * 3.14159265f / 180.0f;
    float yawRad = g_cameraRotation[1] * 3.14159265f / 180.0f;

    Mat4 rotX = Mat4::rotateX(-pitchRad);
    Mat4 rotY = Mat4::rotateY(-yawRad);
    Mat4 trans = Mat4::translate(-g_cameraPosition[0], -g_cameraPosition[1], -g_cameraPosition[2]);
    Mat4 view = trans * rotY * rotX;
    Mat4 vp = proj * view;

    Frustum frustum;
    frustum.extract(vp);

    glUniformMatrix4fv(glGetUniformLocation(g_shaderProgram, "uView"), 1, GL_FALSE, view.m);
    glUniformMatrix4fv(glGetUniformLocation(g_shaderProgram, "uProj"), 1, GL_FALSE, proj.m);
    
    // Upload Light
    glUniform3f(glGetUniformLocation(g_shaderProgram, "uDirLight.direction"), g_dirLight.direction.x, g_dirLight.direction.y, g_dirLight.direction.z);
    glUniform3f(glGetUniformLocation(g_shaderProgram, "uDirLight.ambient"), g_dirLight.ambient.x, g_dirLight.ambient.y, g_dirLight.ambient.z);
    glUniform3f(glGetUniformLocation(g_shaderProgram, "uDirLight.diffuse"), g_dirLight.diffuse.x, g_dirLight.diffuse.y, g_dirLight.diffuse.z);
    glUniform3f(glGetUniformLocation(g_shaderProgram, "uDirLight.specular"), g_dirLight.specular.x, g_dirLight.specular.y, g_dirLight.specular.z);
    
    glUniform3f(glGetUniformLocation(g_shaderProgram, "uViewPos"), g_cameraPosition[0], g_cameraPosition[1], g_cameraPosition[2]);

    Mat4 identity;
    root->updateWorldTransform(identity);

    int cullCount = 0, drawCount = 0;
    
    // Draw regular nodes
    root->draw(frustum, cullCount, drawCount);

    // Draw instanced nodes
    for (auto* instNode : g_instancedNodes) {
        instNode->draw(frustum, cullCount, drawCount);
    }

    std::string title = "Advanced Scenegraph | Drawn: " + std::to_string(drawCount) + " | Culled: " + std::to_string(cullCount);
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

    g_window = glfwCreateWindow(800, 600, "Advanced Scenegraph Demo", nullptr, nullptr);
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
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);

    setupShaders();

    // Setup Global Light
    g_dirLight.direction = Vec3(-0.5f, -1.0f, -0.3f);
    g_dirLight.ambient = Vec3(0.3f, 0.3f, 0.35f);
    g_dirLight.diffuse = Vec3(1.0f, 1.0f, 1.0f);
    g_dirLight.specular = Vec3(1.0f, 1.0f, 1.0f);

    // --- Create Renderables ---
    Renderable* boxRenderable = new Renderable();
    std::vector<Vertex> boxVerts;
    generateBox(boxVerts);
    GLMesh* boxMesh = new GLMesh();
    boxMesh->upload(boxVerts, nullptr);
    GLuint boxTex = loadTexture("fallback_checkerboard");
    boxRenderable->addMesh(boxMesh, boxTex, true);

    Renderable* objRenderable = new Renderable();
    loadModelIntoRenderable(objRenderable, "capsule.obj");

    // --- Build Scenegraph ---
    SceneNode* root = new SceneNode();

    // 1. Animated OBJ Model (Tinted Green!)
    SceneNode* spinGroup = new SceneNode();
    spinGroup->position = Vec3(0, 0, -5);
    root->addChild(spinGroup);

    SceneNode* objNode = new SceneNode();
    objNode->position = Vec3(0, 2, 0); // Floating slightly
    objNode->color = Vec3(0.1f, 1.0f, 0.2f); // Bright Green Tint!
    objNode->material.diffuse = Vec3(1.0f, 1.0f, 1.0f); // Let texture dominate
    objNode->material.specular = Vec3(0.8f, 0.8f, 0.8f); // Shiny
    objNode->material.shininess = 64.0f;
    objNode->setRenderable(objRenderable);
    spinGroup->addChild(objNode);

    // 2. Instanced Terrain Grid (120 Boxes)
    InstancedSceneNode* grid = new InstancedSceneNode();
    grid->renderable = boxRenderable;
    grid->material.diffuse = Vec3(1.0f, 1.0f, 1.0f);
    grid->material.specular = Vec3(0.5f, 0.5f, 0.5f);
    grid->material.shininess = 32.0f;

    int gridSizeX = 12, gridSizeZ = 10; // 120 instances
    for(int x=0; x<gridSizeX; ++x) {
        for(int z=0; z<gridSizeZ; ++z) {
            InstanceData inst;
            float px = (x - gridSizeX/2) * 2.5f + 15.0f;
            float pz = (z - gridSizeZ/2) * 2.5f - 20.0f;
            float py = sin(px * 0.3f) * cos(pz * 0.3f) * 3.0f; // Wavey terrain!
            
            inst.matrix = Mat4::translate(px, py, pz) * Mat4::scale(1.0f, 1.0f + abs(py) * 0.5f, 1.0f);
            
            // Color gradient based on height
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
        
        spinGroup->rotation.y += 20.0f * dt;
        objNode->rotation.x += 10.0f * dt;
        objNode->position.y = 2.0f + sin(currentTime * 2.0f) * 0.5f; // Bobbing

        drawScene(root);
        glfwPollEvents();
    }

    delete root;
    delete boxRenderable;
    delete objRenderable;
    for(auto* in : g_instancedNodes) delete in;
    
    glfwTerminate();
    return 0;
}