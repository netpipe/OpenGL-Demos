// ============================================================================
// Scenegraph & Culling Demo with Integrated OBJ Loader
// ============================================================================
// To compile (Windows MinGW/MSVC):
//   g++ main.cpp -o demo -lglfw3 -lglew32 -lopengl32
//   cl main.cpp /link glfw3.lib glew32.lib opengl32.lib
//
// To compile (Linux):
//   g++ main.cpp -o demo -lglfw -lGLEW -lGL
//
// This demo integrates `simple_obj_mtl_loader.h` with a modern OpenGL 6DOF 
// camera architecture. It implements a hierarchical scenegraph with custom 
// nodes and frustum culling to optimize rendering.
// ============================================================================

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
        // Extracting 6 planes from a column-major View-Projection matrix (Gribb-Hartmann)
        planes[0] = {vp.m[3] - vp.m[0], vp.m[7] - vp.m[4], vp.m[11] - vp.m[8], vp.m[15] - vp.m[12]}; // Right
        planes[1] = {vp.m[3] + vp.m[0], vp.m[7] + vp.m[4], vp.m[11] + vp.m[8], vp.m[15] + vp.m[12]}; // Left
        planes[2] = {vp.m[3] + vp.m[1], vp.m[7] + vp.m[5], vp.m[11] + vp.m[9], vp.m[15] + vp.m[13]}; // Bottom
        planes[3] = {vp.m[3] - vp.m[1], vp.m[7] - vp.m[5], vp.m[11] - vp.m[9], vp.m[15] - vp.m[13]}; // Top
        planes[4] = {vp.m[3] - vp.m[2], vp.m[7] - vp.m[6], vp.m[11] - vp.m[10], vp.m[15] - vp.m[14]}; // Far
        planes[5] = {vp.m[3] + vp.m[2], vp.m[7] + vp.m[6], vp.m[11] + vp.m[10], vp.m[15] + vp.m[14]}; // Near

        for(int i=0; i<6; ++i) planes[i].normalize();
    }

    bool intersects(const BoundingBox& aabb) const {
        for (int i=0; i<6; ++i) {
            // Find the positive vertex (furthest along the plane normal)
            Vec3 pVertex;
            pVertex.x = (planes[i].a >= 0.0f) ? aabb.max.x : aabb.min.x;
            pVertex.y = (planes[i].b >= 0.0f) ? aabb.max.y : aabb.min.y;
            pVertex.z = (planes[i].c >= 0.0f) ? aabb.max.z : aabb.min.z;

            if (planes[i].distance(pVertex) < 0.0f) return false;
        }
        return true;
    }
};

// Calculates inverse transpose of upper 3x3 matrix for correct normal lighting transformations
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
// 2. EMBEDDED OBJ / MTL LOADER
// ==========================================
struct OBJMaterial {
    std::string name;
    Vec3 ambient, diffuse, specular;
    float shininess;
    std::string diffuseMap;
    unsigned int textureID;

    OBJMaterial() {
        ambient   = Vec3(0.2f, 0.2f, 0.2f);
        diffuse   = Vec3(0.8f, 0.8f, 0.8f);
        specular  = Vec3(0.0f, 0.0f, 0.0f);
        shininess = 0.0f;
        textureID = 0;
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
        std::sscanf(token.c_str(), "%d/%d/%d", &vi, &ti, &ni);
        vert.position = positions[vi - 1];
        vert.texcoord = (ti > 0) ? texcoords[ti - 1] : Vec2();
        vert.texcoord.y = 1.0f - vert.texcoord.y;
        vert.normal   = (ni > 0) ? normals[ni - 1] : Vec3(0,0,1);
        mesh.vertices.push_back(vert);
    }
};

// ==========================================
// 3. SCENEGRAPH & RENDERABLE COMPONENTS
// ==========================================

class Renderable {
public:
    GLuint vao, vbo, ebo;
    int indexCount;
    BoundingBox localAABB;
    
    Renderable() : vao(0), vbo(0), ebo(0), indexCount(0) {}
    ~Renderable() {
        if (vao) glDeleteVertexArrays(1, &vao);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (ebo) glDeleteBuffers(1, &ebo);
    }
    
    void upload(const std::vector<Vertex>& verts, const std::vector<unsigned int>& indices) {
        for (const auto& v : verts) localAABB.expand(v.position);
        indexCount = indices.size();

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        glBindVertexArray(0);
    }

    void draw() {
        if (vao) {
            glBindVertexArray(vao);
            glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }
};

void generateBox(std::vector<Vertex>& verts, std::vector<unsigned int>& indices) {
    float h = 0.5f;
    verts.push_back({Vec3(-h, -h, h), Vec3(0,0,1), Vec2(0,0)});
    verts.push_back({Vec3( h, -h, h), Vec3(0,0,1), Vec2(1,0)});
    verts.push_back({Vec3( h,  h, h), Vec3(0,0,1), Vec2(1,1)});
    verts.push_back({Vec3(-h,  h, h), Vec3(0,0,1), Vec2(0,1)});
    verts.push_back({Vec3( h, -h,-h), Vec3(0,0,-1), Vec2(0,0)});
    verts.push_back({Vec3(-h, -h,-h), Vec3(0,0,-1), Vec2(1,0)});
    verts.push_back({Vec3(-h,  h,-h), Vec3(0,0,-1), Vec2(1,1)});
    verts.push_back({Vec3( h,  h,-h), Vec3(0,0,-1), Vec2(0,1)});
    verts.push_back({Vec3(-h,  h, h), Vec3(0,1,0), Vec2(0,0)});
    verts.push_back({Vec3( h,  h, h), Vec3(0,1,0), Vec2(1,0)});
    verts.push_back({Vec3( h,  h,-h), Vec3(0,1,0), Vec2(1,1)});
    verts.push_back({Vec3(-h,  h,-h), Vec3(0,1,0), Vec2(0,1)});
    verts.push_back({Vec3(-h, -h,-h), Vec3(0,-1,0), Vec2(0,0)});
    verts.push_back({Vec3( h, -h,-h), Vec3(0,-1,0), Vec2(1,0)});
    verts.push_back({Vec3( h, -h, h), Vec3(0,-1,0), Vec2(1,1)});
    verts.push_back({Vec3(-h, -h, h), Vec3(0,-1,0), Vec2(0,1)});
    verts.push_back({Vec3( h, -h, h), Vec3(1,0,0), Vec2(0,0)});
    verts.push_back({Vec3( h, -h,-h), Vec3(1,0,0), Vec2(1,0)});
    verts.push_back({Vec3( h,  h,-h), Vec3(1,0,0), Vec2(1,1)});
    verts.push_back({Vec3( h,  h, h), Vec3(1,0,0), Vec2(0,1)});
    verts.push_back({Vec3(-h, -h,-h), Vec3(-1,0,0), Vec2(0,0)});
    verts.push_back({Vec3(-h, -h, h), Vec3(-1,0,0), Vec2(1,0)});
    verts.push_back({Vec3(-h,  h, h), Vec3(-1,0,0), Vec2(1,1)});
    verts.push_back({Vec3(-h,  h,-h), Vec3(-1,0,0), Vec2(0,1)});

    for (int i=0; i<6; ++i) {
        int off = i * 4;
        indices.push_back(off + 0); indices.push_back(off + 1); indices.push_back(off + 2);
        indices.push_back(off + 2); indices.push_back(off + 3); indices.push_back(off + 0);
    }
}

void loadModelIntoRenderable(Renderable* r, const std::string& path) {
    OBJModel model;
    if (model.load(path) && !model.meshes.empty()) {
        std::vector<Vertex> allVerts;
        std::vector<unsigned int> allIndices;
        for (const auto& mesh : model.meshes) {
            size_t offset = allVerts.size();
            allVerts.insert(allVerts.end(), mesh.vertices.begin(), mesh.vertices.end());
            for (size_t i = 0; i < mesh.vertices.size(); ++i) {
                allIndices.push_back(offset + i);
            }
        }
        r->upload(allVerts, allIndices);
        std::cout << "Loaded OBJ: " << path << " (" << allVerts.size() << " vertices)\n";
    } else {
        std::cout << "Failed to load OBJ: " << path << ". Using fallback box.\n";
        std::vector<Vertex> verts;
        std::vector<unsigned int> indices;
        generateBox(verts, indices);
        r->upload(verts, indices);
    }
}

GLuint g_shaderProgram = 0;

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
    Vec3 color = Vec3(0.8f, 0.8f, 0.8f);

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
        
        // Update children FIRST to ensure bottom-up AABB calculation works on the first frame
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

    void draw(const Frustum& frustum, const Mat4& view, const Mat4& proj, int& cullCount, int& drawCount) {
        if (!visible) return;

        // Hierarchical Frustum Culling
        if (renderable || !children.empty()) {
            if (!frustum.intersects(worldAABB)) {
                cullCount += countNodes();
                return; 
            }
        }

        if (renderable) {
            glUniformMatrix4fv(glGetUniformLocation(g_shaderProgram, "uModel"), 1, GL_FALSE, worldMatrix.m);
            
            float normalMat[9];
            getNormalMatrix(worldMatrix, normalMat);
            glUniformMatrix3fv(glGetUniformLocation(g_shaderProgram, "uNormalMatrix"), 1, GL_FALSE, normalMat);

            glUniform3f(glGetUniformLocation(g_shaderProgram, "uBaseColor"), color.x, color.y, color.z);

            renderable->draw();
            drawCount++;
        }

        for (auto* child : children) child->draw(frustum, view, proj, cullCount, drawCount);
    }
    
    int countNodes() {
        int c = 1;
        for (auto* child : children) c += child->countNodes();
        return c;
    }
};

// ==========================================
// 4. GLOBAL VARIABLES & SHADERS
// ==========================================

GLFWwindow* g_window = nullptr;
float g_cameraPosition[3] = {0.0f, 3.0f, 15.0f}; 
float g_cameraRotation[3] = {0.0f, 0.0f, 0.0f}; 
float g_speed = 0.15f;

const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoord;

    uniform mat4 uModel, uView, uProj;
    uniform mat3 uNormalMatrix;

    out vec3 vNormal, vFragPos;

    void main() {
        vec4 worldPos = uModel * vec4(aPos, 1.0);
        vFragPos = worldPos.xyz;
        vNormal = uNormalMatrix * aNormal;
        gl_Position = uProj * uView * worldPos;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 vNormal, vFragPos;
    out vec4 FragColor;

    uniform vec3 uLightDir, uBaseColor, uViewPos;

    void main() {
        vec3 norm = normalize(vNormal);
        vec3 lightDir = normalize(uLightDir);
        float diff = max(dot(norm, -lightDir), 0.0);
        vec3 ambient = 0.2 * uBaseColor;
        vec3 diffuse = diff * uBaseColor;
        
        vec3 viewDir = normalize(uViewPos - vFragPos);
        vec3 reflectDir = reflect(lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = vec3(0.3) * spec;
        
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
    glUniform3f(glGetUniformLocation(g_shaderProgram, "uLightDir"), -0.5f, -1.0f, -0.3f);
    glUniform3f(glGetUniformLocation(g_shaderProgram, "uViewPos"), g_cameraPosition[0], g_cameraPosition[1], g_cameraPosition[2]);

    Mat4 identity;
    root->updateWorldTransform(identity);

    int cullCount = 0, drawCount = 0;
    root->draw(frustum, view, proj, cullCount, drawCount);

    std::string title = "Scenegraph Demo | Drawn: " + std::to_string(drawCount) + " | Culled: " + std::to_string(cullCount);
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

    g_window = glfwCreateWindow(800, 600, "Scenegraph Demo", nullptr, nullptr);
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

    // Allocate shared renderable geometry
    Renderable* boxRenderable = new Renderable();
    std::vector<Vertex> boxVerts; std::vector<unsigned int> boxIndices;
    generateBox(boxVerts, boxIndices);
    boxRenderable->upload(boxVerts, boxIndices);

    Renderable* objRenderable = new Renderable();
    loadModelIntoRenderable(objRenderable, "capsule.obj");

    SceneNode* root = new SceneNode();

    // 1. Create a spinning object group
    SceneNode* spinGroup = new SceneNode();
    spinGroup->position = Vec3(0, 0, -5);
    root->addChild(spinGroup);

    SceneNode* objNode = new SceneNode();
    objNode->position = Vec3(3, 0, 0);
   // objNode->color = Vec3(0.2f, 0.8f, 0.2f);
    objNode->setRenderable(objRenderable);
    spinGroup->addChild(objNode);

    // 2. Create a large grid of nodes to show off hierarchical culling
    SceneNode* gridGroup = new SceneNode();
    gridGroup->position = Vec3(15, 0, -20);
    root->addChild(gridGroup);

    int gridSize = 20; // 10x10 = 100 child nodes
    for(int x=0; x<gridSize; ++x) {
        for(int z=0; z<gridSize; ++z) {
            SceneNode* boxNode = new SceneNode();
            boxNode->position = Vec3((x - gridSize/2) * 2.0f, 0, (z - gridSize/2) * 2.0f);
            boxNode->setRenderable(boxRenderable); // Shared renderable
            boxNode->color = Vec3(0.8f, 0.4f, 0.1f);
            gridGroup->addChild(boxNode);
        }
    }

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(g_window)) {
        double currentTime = glfwGetTime();
        float dt = currentTime - lastTime;
        lastTime = currentTime;

        handleInput(g_window);
        
        // Animate spinning group to show local transformations
        spinGroup->rotation.y += 30.0f * dt;
        objNode->rotation.x += 60.0f * dt;

        drawScene(root);
        glfwPollEvents();
    }

    delete root;
    delete boxRenderable;
    delete objRenderable;
    glfwTerminate();
    return 0;
}