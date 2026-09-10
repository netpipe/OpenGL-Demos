#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <memory>
#include <algorithm>
#include <chrono>
#include <queue>
#include <random>
#include <array>

#define PI 3.14159265358979323846f

// ============================================================================
// 1. MATH PRIMITIVES
// ============================================================================
struct Vec2 { float x, y; Vec2(float x=0, float y=0):x(x),y(y){} };
struct Vec3 { 
    float x, y, z; 
    Vec3(float x=0, float y=0, float z=0):x(x),y(y),z(z){} 
    Vec3 operator+(const Vec3& o) const {return {x+o.x, y+o.y, z+o.z};} 
    Vec3 operator-(const Vec3& o) const {return {x-o.x, y-o.y, z-o.z};} 
    Vec3 operator*(float s) const {return {x*s, y*s, z*s};} 
    float dot(const Vec3& o) const {return x*o.x + y*o.y + z*o.z;} 
    Vec3 cross(const Vec3& o) const {return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};} 
    float length() const {return std::sqrt(x*x + y*y + z*z);} 
    Vec3 normalized() const {float l=length(); return l>0?Vec3(x/l, y/l, z/l):Vec3();} 
    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};
struct Vec4 { float x, y, z, w; Vec4(float x=0,float y=0,float z=0,float w=0):x(x),y(y),z(z),w(w){} };

struct Mat4 { 
    float m[16]; 
    Mat4() { for(int i=0;i<16;++i) m[i]=0; m[0]=m[5]=m[10]=m[15]=1; } 
    static Mat4 identity() { return Mat4(); }
    static Mat4 translate(const Vec3& t) { Mat4 r; r.m[12]=t.x; r.m[13]=t.y; r.m[14]=t.z; return r; }
    static Mat4 perspective(float fov, float aspect, float near, float far) {
        Mat4 r; float f = 1.0f / std::tan(fov/2);
        r.m[0]=f/aspect; r.m[5]=f; r.m[10]=(far+near)/(near-far); r.m[11]=-1; r.m[14]=(2*far*near)/(near-far); r.m[15]=0; return r;
    }
    Mat4 operator*(const Mat4& o) const {
        Mat4 r; for(int i=0;i<4;++i) for(int j=0;j<4;++j) { r.m[j*4+i]=0; for(int k=0;k<4;++k) r.m[j*4+i]+=m[k*4+i]*o.m[j*4+k]; } return r;
    }
    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = (center - eye).normalized();
    Vec3 s = f.cross(up).normalized();
    Vec3 u = s.cross(f);
    
    Mat4 r = Mat4::identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8]  = s.z;
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    
    r.m[12] = -s.dot(eye);
    r.m[13] = -u.dot(eye);
    r.m[14] = f.dot(eye);
    return r;
}
};

struct AABB {
    Vec3 min, max;
    AABB():min(1e30f,1e30f,1e30f),max(-1e30f,-1e30f,-1e30f){}
    AABB(const Vec3& mn, const Vec3& mx):min(mn),max(mx){}
    void expand(const Vec3& p) { min.x=std::min(min.x,p.x); min.y=std::min(min.y,p.y); min.z=std::min(min.z,p.z); max.x=std::max(max.x,p.x); max.y=std::max(max.y,p.y); max.z=std::max(max.z,p.z); }
};

// ============================================================================
// 2. SCENE GRAPH 
// ============================================================================
class SceneNode {
public:
    std::string name;
    Mat4 localMatrix, worldMatrix;
    AABB localBounds, worldBounds;
    SceneNode* parent = nullptr;
    std::vector<SceneNode*> children;
    bool visible = true;
    
    virtual ~SceneNode() { for(auto* c : children) delete c; }
    
    void addChild(SceneNode* child) {
        child->parent = this;
        children.push_back(child);
    }
    
    virtual void updateTransform(const Mat4& parentWorld) {
        worldMatrix = parentWorld * localMatrix;
        worldBounds = localBounds; 
        for (auto* c : children) c->updateTransform(worldMatrix);
    }
};

class LightNode : public SceneNode {
public:
    enum Type { SUN, POINT, SPOT, AREA_RECT, AREA_SPHERE } type;
    Vec3 color; float intensity;
    LightNode(Type t) : type(t) { color = Vec3(1,1,1); intensity = 1.0f; }
};

// ============================================================================
// 3. MESH & GENERATION
// ============================================================================
struct Vertex { Vec3 pos, normal; Vec2 uv; };

class GLMesh {
public:
    GLuint vao, vbo;
    std::vector<Vertex> verts;
    
    ~GLMesh() { glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &vbo); }
    
    void upload() {
        glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3*sizeof(float)));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(6*sizeof(float)));
        glBindVertexArray(0);
    }
    void draw() { glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, verts.size()); glBindVertexArray(0); }
};

void generateCube(GLMesh& mesh, float size) {
    float h = size * 0.5f;
    float vertices[] = {
        // Front
        -h, -h,  h,  0, 0, 1, 0,0,   h, -h,  h,  0, 0, 1, 1,0,   h,  h,  h,  0, 0, 1, 1,1,  -h,  h,  h,  0, 0, 1, 0,1,
        // Back
         h, -h, -h,  0, 0,-1, 0,0,  -h, -h, -h,  0, 0,-1, 1,0,  -h,  h, -h,  0, 0,-1, 1,1,   h,  h, -h,  0, 0,-1, 0,1,
        // Top
        -h,  h,  h,  0, 1, 0, 0,0,   h,  h,  h,  0, 1, 0, 1,0,   h,  h, -h,  0, 1, 0, 1,1,  -h,  h, -h,  0, 1, 0, 0,1,
        // Bottom
        -h, -h, -h,  0,-1, 0, 0,0,   h, -h, -h,  0,-1, 0, 1,0,   h, -h,  h,  0,-1, 0, 1,1,  -h, -h,  h,  0,-1, 0, 0,1,
        // Right
         h, -h,  h,  1, 0, 0, 0,0,   h, -h, -h,  1, 0, 0, 1,0,   h,  h, -h,  1, 0, 0, 1,1,   h,  h,  h,  1, 0, 0, 0,1,
        // Left
        -h, -h, -h, -1, 0, 0, 0,0,  -h, -h,  h, -1, 0, 0, 1,0,  -h,  h,  h, -1, 0, 0, 1,1,  -h,  h, -h, -1, 0, 0, 0,1
    };
    unsigned int indices[] = {
        0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11, 12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
    };
    for(int i=0; i<36; ++i) {
        int idx = indices[i];
        Vertex v;
        v.pos = {vertices[idx*8+0], vertices[idx*8+1], vertices[idx*8+2]};
        v.normal = {vertices[idx*8+3], vertices[idx*8+4], vertices[idx*8+5]};
        v.uv = {vertices[idx*8+6], vertices[idx*8+7]};
        mesh.verts.push_back(v);
    }
    mesh.upload();
}

void generatePlane(GLMesh& mesh, float size) {
    float h = size * 0.5f;
    Vertex v; v.normal = {0,1,0};
    v.pos = {-h, 0, -h}; v.uv = {0,0}; mesh.verts.push_back(v);
    v.pos = {h, 0, -h}; v.uv = {1,0}; mesh.verts.push_back(v);
    v.pos = {h, 0, h}; v.uv = {1,1}; mesh.verts.push_back(v);
    v.pos = {-h, 0, -h}; v.uv = {0,0}; mesh.verts.push_back(v);
    v.pos = {h, 0, h}; v.uv = {1,1}; mesh.verts.push_back(v);
    v.pos = {-h, 0, h}; v.uv = {0,1}; mesh.verts.push_back(v);
    mesh.upload();
}

// ============================================================================
// 4. DEFERRED SHADING PIPELINE (Shaders)
// ============================================================================

const char* gBufferVert = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uM, uV, uP;
out vec3 vWP; out vec3 vN; out vec2 vUV;
void main() {
    vec4 wp = uM * vec4(aPos, 1.0);
    vWP = wp.xyz; 
    vN = mat3(transpose(inverse(uM))) * aNormal;
    vUV = aUV; 
    gl_Position = uP * uV * wp;
}
)";

const char* gBufferFrag = R"(
#version 330 core
in vec3 vWP; in vec3 vN; in vec2 vUV;
layout(location=0) out vec4 gPos;
layout(location=1) out vec4 gNorm;
layout(location=2) out vec4 gAlb;
uniform vec3 uColor; 
void main() {
    // Alpha channel = 1.0 means "Valid Geometry". (0.0 means empty sky)
    gPos = vec4(vWP, 1.0); 
    gNorm = vec4(normalize(vN), 1.0);
    gAlb = vec4(uColor, 1.0);
}
)";

const char* lightPassVert = R"(
#version 330 core
layout(location=0) in vec2 aPos; out vec2 vUV;
void main() { vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0, 1); }
)";

const char* lightPassFrag = R"(
#version 330 core
in vec2 vUV; out vec4 FragColor;
uniform sampler2D gPos, gNorm, gAlb;
uniform vec3 uCamPos, uSunDir, uSunColor;

// Godrays / Volumetric Fog
vec3 calcVolumetric(vec3 ro, vec3 rd, float maxD) {
    float steps = 16.0; 
    float stepSize = maxD / steps; 
    vec3 accum = vec3(0.0);
    for(float i=1.0; i<=steps; i++) {
        vec3 pos = ro + rd * (i * stepSize);
        float heightFog = exp(-max(pos.y - 2.0, 0.0) * 0.1);
        float shadow = max(dot(rd, -uSunDir), 0.0); 
        accum += uSunColor * shadow * heightFog * 0.05 * stepSize;
    }
    return accum;
}

void main() {
    vec4 pd = texture(gPos, vUV);
    
    // If Alpha is 0.0, this is the background sky
    if(pd.a < 0.5) {
        FragColor = vec4(0.5, 0.7, 1.0, 1.0); // Sky blue
        return;
    }
    
    vec3 wp = pd.xyz, wn = normalize(texture(gNorm, vUV).xyz), alb = texture(gAlb, vUV).rgb;
    vec3 Lo = alb * 0.1; // Ambient
    
    // Sun Lighting
    float diff = max(dot(wn, -uSunDir), 0.0);
    Lo += alb * uSunColor * diff;
    
    vec3 viewDir = normalize(uCamPos - wp);
    float dist = length(uCamPos - wp);
    
    // Add Volumetric Fog / Godrays
    Lo += calcVolumetric(uCamPos, -viewDir, dist); 
    
    FragColor = vec4(Lo, 1.0);
}
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type); 
    glShaderSource(s, 1, &src, NULL); 
    glCompileShader(s); 
    GLint success;
    glGetShaderiv(s, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(s, 512, NULL, log);
        std::cerr << "Shader Error: " << log << std::endl;
    }
    return s;
}

// ============================================================================
// 5. MAIN APPLICATION
// ============================================================================
int main() {
    if (!glfwInit()) return -1;

    // Request OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Holodeck Engine (Fixed Deferred)", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;
glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Hides and captures mouse
    int w, h; glfwGetFramebufferSize(window, &w, &h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Setup G-Buffer
    GLuint g_fbo, g_pos, g_norm, g_alb, g_depth;
    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    
    // Changed to RGBA16F to support Alpha channel for Validity Flag
    glGenTextures(1, &g_pos); glBindTexture(GL_TEXTURE_2D, g_pos);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_pos, 0);
    
    glGenTextures(1, &g_norm); glBindTexture(GL_TEXTURE_2D, g_norm);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g_norm, 0);
    
    glGenTextures(1, &g_alb); glBindTexture(GL_TEXTURE_2D, g_alb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, g_alb, 0);
    
    glGenRenderbuffers(1, &g_depth); glBindRenderbuffer(GL_RENDERBUFFER, g_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_depth);
    
    GLuint attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, attachments);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Compile Shaders
    GLuint gShader = glCreateProgram();
    glAttachShader(gShader, compileShader(GL_VERTEX_SHADER, gBufferVert));
    glAttachShader(gShader, compileShader(GL_FRAGMENT_SHADER, gBufferFrag));
    glLinkProgram(gShader);

    GLuint lShader = glCreateProgram();
    glAttachShader(lShader, compileShader(GL_VERTEX_SHADER, lightPassVert));
    glAttachShader(lShader, compileShader(GL_FRAGMENT_SHADER, lightPassFrag));
    glLinkProgram(lShader);

    // Setup Geometry
    GLMesh cubeMesh; generateCube(cubeMesh, 2.0f);
    GLMesh planeMesh; generatePlane(planeMesh, 50.0f);

    SceneNode root;
    LightNode* sun = new LightNode(LightNode::SUN); 
    sun->color = Vec3(1.0f, 0.9f, 0.8f); sun->intensity = 2.0f; 
    root.addChild(sun);

    SceneNode* cubeNode = new SceneNode();
    cubeNode->localMatrix = Mat4::translate({0, 1.5f, -5});
    root.addChild(cubeNode);

    SceneNode* floorNode = new SceneNode();
    root.addChild(floorNode);

    // Setup Fullscreen Quad (Created ONCE outside loop)
    float quad[] = { -1, -1,  1, -1,  1, 1,  -1, -1,  1, 1,  -1, 1 };
    GLuint qVao, qVbo;
    glGenVertexArrays(1, &qVao); glGenBuffers(1, &qVbo);
    glBindVertexArray(qVao);
    glBindBuffer(GL_ARRAY_BUFFER, qVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glBindVertexArray(0);

Vec3 camPos = {0, 2, 8};
float camYaw = -90.0f;   // Start facing -Z
float camPitch = 0.0f;
double lastX = 640, lastY = 360;
bool firstMouse = true;
    glEnable(GL_DEPTH_TEST);
    
    while (!glfwWindowShouldClose(window)) {
        root.updateTransform(Mat4::identity());
        
        // 1. Geometry Pass
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(gShader);
        
// --- FPS CAMERA CONTROLS ---
double xpos, ypos;
glfwGetCursorPos(window, &xpos, &ypos);

if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
float xoffset = xpos - lastX;
float yoffset = lastY - ypos; // Reversed since Y coordinates go from bottom to top
lastX = xpos; lastY = ypos;

float sensitivity = 0.1f;
camYaw += xoffset * sensitivity;
camPitch += yoffset * sensitivity;

// Clamp pitch to prevent screen flip
if (camPitch > 89.0f) camPitch = 89.0f;
if (camPitch < -89.0f) camPitch = -89.0f;

// Calculate new Front vector
Vec3 camFront;
camFront.x = cos(camYaw * PI / 180.0f) * cos(camPitch * PI / 180.0f);
camFront.y = sin(camPitch * PI / 180.0f);
camFront.z = sin(camYaw * PI / 180.0f) * cos(camPitch * PI / 180.0f);
camFront = camFront.normalized();

// WASD Movement
float speed = 0.15f;
Vec3 camRight = camFront.cross({0, 1, 0}).normalized();

if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camPos = camPos + camFront * speed;
if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camPos = camPos - camFront * speed;
if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camPos = camPos + camRight * speed;
if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camPos = camPos - camRight * speed;

// Update View Matrix using lookAt
Mat4 view = Mat4::lookAt(camPos, camPos + camFront, {0, 1, 0});
Mat4 proj = Mat4::perspective(PI / 3.0f, (float)w / h, 0.1f, 100.0f);
// ---------------------------
        
        glUniformMatrix4fv(glGetUniformLocation(gShader, "uV"), 1, GL_FALSE, view.m);
        glUniformMatrix4fv(glGetUniformLocation(gShader, "uP"), 1, GL_FALSE, proj.m);
        
        glUniformMatrix4fv(glGetUniformLocation(gShader, "uM"), 1, GL_FALSE, cubeNode->worldMatrix.m);
        glUniform3f(glGetUniformLocation(gShader, "uColor"), 0.8f, 0.2f, 0.2f);
        cubeMesh.draw();
        
        glUniformMatrix4fv(glGetUniformLocation(gShader, "uM"), 1, GL_FALSE, floorNode->worldMatrix.m);
        glUniform3f(glGetUniformLocation(gShader, "uColor"), 0.4f, 0.4f, 0.4f);
        planeMesh.draw();
        
        // 2. Lighting Pass (Godrays, Fog, Lights)
        glDisable(GL_DEPTH_TEST); // Disable for 2D screen pass
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(lShader);
        
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, g_pos);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, g_norm);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, g_alb);
        
        glUniform1i(glGetUniformLocation(lShader, "gPos"), 0);
        glUniform1i(glGetUniformLocation(lShader, "gNorm"), 1);
        glUniform1i(glGetUniformLocation(lShader, "gAlb"), 2);
        
        glUniform3f(glGetUniformLocation(lShader, "uCamPos"), camPos.x, camPos.y, camPos.z);
        glUniform3f(glGetUniformLocation(lShader, "uSunDir"), 0.5f, -0.8f, 0.3f);
        glUniform3f(glGetUniformLocation(lShader, "uSunColor"), sun->color.x, sun->color.y, sun->color.z);
        
        glBindVertexArray(qVao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST); // Re-enable
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    return 0;
}