// main.cpp
// Real-Time GPU Raytracer (OpenGL 4.1 Core Profile)
// Features: Analytical Plane, Temporal Accumulation, Safe BVH
//
// Controls:
//   SPACE : Pause/Resume camera (allows image to converge/clear noise)
//   ESC   : Quit
//
// Compilation (Linux):
// g++ main.cpp -o raytracer -lglfw -lGLEW -lGL -O3 -std=c++11
//
// Compilation (macOS):
// clang++ main.cpp -o raytracer -lglfw -lGLEW -framework OpenGL -O3 -std=c++11

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <unistd.h>

using namespace std;

// ---------------------------------------------------------
// Math utilities
// ---------------------------------------------------------

struct vec3 {
    float x, y, z;
    vec3() : x(0), y(0), z(0) {}
    vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    vec3(float v) : x(v), y(v), z(v) {}
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

struct vec4 {
    float x, y, z, w;
    vec4() : x(0), y(0), z(0), w(0) {}
    vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
};

vec3 operator+(vec3 a, vec3 b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
vec3 operator-(vec3 a, vec3 b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
vec3 operator*(vec3 a, float s) { return vec3(a.x * s, a.y * s, a.z * s); }
vec3 operator*(float s, vec3 a) { return vec3(a.x * s, a.y * s, a.z * s); }
vec3 operator*(vec3 a, vec3 b) { return vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
vec3 operator/(vec3 a, float s) { return vec3(a.x / s, a.y / s, a.z / s); }
vec3 operator-(vec3 a) { return vec3(-a.x, -a.y, -a.z); }

float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
vec3 cross(vec3 a, vec3 b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float length(vec3 a) { return sqrtf(dot(a, a)); }
vec3 normalize(vec3 a) {
    float l = length(a);
    if (l > 0.00001f) return a / l;
    return vec3(0);
}
vec3 min_v3(vec3 a, vec3 b) { return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
vec3 max_v3(vec3 a, vec3 b) { return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
vec3 reflect(vec3 v, vec3 n) { return v - n * (2.0f * dot(v, n)); }
vec3 refract(vec3 uv, vec3 n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    vec3 r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    vec3 r_out_parallel = n * (-sqrtf(fabsf(1.0f - dot(r_out_perp, r_out_perp))));
    return r_out_perp + r_out_parallel;
}

// ---------------------------------------------------------
// Scene structures
// ---------------------------------------------------------

struct Sphere { vec3 pos; float radius; };

struct Material {
    float m1[4]; // xyz albedo/attenuation, w type
    float m2[4]; // w fuzz/density/ref_idx
};

struct AnimSphere {
    vec3 base_pos;
    float base_radius;
    int mat_idx;
    int anim_type; // 0 static, 1 x-axis, 2 y-axis, 3 orbit
    float speed;
    float amp;
};

Material make_diffuse(vec3 albedo) {
    Material m; m.m1[0]=albedo.x; m.m1[1]=albedo.y; m.m1[2]=albedo.z; m.m1[3]=0.0f;
    m.m2[0]=0; m.m2[1]=0; m.m2[2]=0; m.m2[3]=0; return m;
}
Material make_metal(vec3 albedo, float fuzz) {
    Material m; m.m1[0]=albedo.x; m.m1[1]=albedo.y; m.m1[2]=albedo.z; m.m1[3]=1.0f;
    m.m2[0]=0; m.m2[1]=0; m.m2[2]=0; m.m2[3]=fuzz; return m;
}
Material make_glass(vec3 attenuation, float ref_idx) {
    Material m; m.m1[0]=attenuation.x; m.m1[1]=attenuation.y; m.m1[2]=attenuation.z; m.m1[3]=2.0f;
    m.m2[0]=0; m.m2[1]=0; m.m2[2]=0; m.m2[3]=ref_idx; return m;
}

// ---------------------------------------------------------
// BVH builder CPU
// ---------------------------------------------------------

struct AABB {
    vec3 min, max;
    AABB() : min(1e38f, 1e38f, 1e38f), max(-1e38f, -1e38f, -1e38f) {}
    void expand(vec3 p) { min = min_v3(min, p); max = max_v3(max, p); }
    int max_extent() const {
        vec3 d = max - min;
        if (d.x > d.y && d.x > d.z) return 0;
        if (d.y > d.z) return 1;
        return 2;
    }
};

struct BVHNode {
    float min_x, min_y, min_z; int left;
    float max_x, max_y, max_z; int right;
};

int build_bvh(int start, int end, const vector<Sphere>& spheres, vector<int>& primitive_indices, vector<BVHNode>& bvh_nodes) {
    AABB bounds;
    for (int i = start; i < end; ++i) {
        int p_idx = primitive_indices[i];
        vec3 r(spheres[p_idx].radius);
        bounds.expand(spheres[p_idx].pos - r);
        bounds.expand(spheres[p_idx].pos + r);
    }
    int count = end - start;
    int node_idx = (int)bvh_nodes.size();
    bvh_nodes.push_back(BVHNode());

    if (count == 1) {
        bvh_nodes[node_idx].min_x = bounds.min.x; bvh_nodes[node_idx].min_y = bounds.min.y; bvh_nodes[node_idx].min_z = bounds.min.z;
        bvh_nodes[node_idx].left = primitive_indices[start];
        bvh_nodes[node_idx].max_x = bounds.max.x; bvh_nodes[node_idx].max_y = bounds.max.y; bvh_nodes[node_idx].max_z = bounds.max.z;
        bvh_nodes[node_idx].right = -1; // leaf marker
        return node_idx;
    }

    int axis = bounds.max_extent();
    sort(primitive_indices.begin() + start, primitive_indices.begin() + end, [&](int a, int b) {
        float va = (axis == 0) ? spheres[a].pos.x : (axis == 1) ? spheres[a].pos.y : spheres[a].pos.z;
        float vb = (axis == 0) ? spheres[b].pos.x : (axis == 1) ? spheres[b].pos.y : spheres[b].pos.z;
        return va < vb;
    });

    int mid = (start + end) / 2;
    int left_child = build_bvh(start, mid, spheres, primitive_indices, bvh_nodes);
    int right_child = build_bvh(mid, end, spheres, primitive_indices, bvh_nodes);

    bvh_nodes[node_idx].min_x = bounds.min.x; bvh_nodes[node_idx].min_y = bounds.min.y; bvh_nodes[node_idx].min_z = bounds.min.z;
    bvh_nodes[node_idx].left = left_child;
    bvh_nodes[node_idx].max_x = bounds.max.x; bvh_nodes[node_idx].max_y = bounds.max.y; bvh_nodes[node_idx].max_z = bounds.max.z;
    bvh_nodes[node_idx].right = right_child;
    return node_idx;
}

// ---------------------------------------------------------
// Shaders
// ---------------------------------------------------------

const char* vertex_shader_source = R"(#version 410 core
layout(location = 0) in vec2 pos;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    TexCoord = pos * 0.5 + 0.5;
}
)";

const char* fragment_shader_source = R"(#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform vec3 u_camPos;
uniform vec3 u_camDir;
uniform vec3 u_camRight;
uniform vec3 u_camUp;
uniform float u_time;
uniform int u_sphere_count;
uniform int u_material_count;
uniform int u_bvh_node_count;
uniform vec3 u_sunDir;

// Floor plane uniforms
uniform vec3 u_floorNormal;
uniform float u_floorD;
uniform int u_floorMatIdx;

// Accumulation uniforms
uniform sampler2D u_accumTex;
uniform int u_frameIndex;
uniform int u_resetAccum;
uniform int u_mode; // 0 = trace, 1 = display

uniform samplerBuffer u_spheres_pos_rad;
uniform samplerBuffer u_spheres_mat_idx;
uniform samplerBuffer u_mat_data;
uniform samplerBuffer u_bvhNodes;

#define MAX_BOUNCES 6
#define MAX_STACK 64

uint hash(uint x) {
    x += (x << 10u); x ^= (x >> 6u); x += (x << 3u); x ^= (x >> 11u); x += (x << 15u);
    return x;
}
uint hash3(uint x, uint y, uint z) { return hash(x ^ hash(y ^ hash(z))); }
float random(inout uint seed) {
    seed = hash(seed);
    return float(seed) / float(0xFFFFFFFFu);
}
vec3 random_dir(inout uint seed) {
    float z = random(seed) * 2.0 - 1.0;
    float a = random(seed) * 2.0 * 3.14159265;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return vec3(r * cos(a), r * sin(a), z);
}

float safe_inv_component(float v) {
    if (abs(v) > 1e-8) return 1.0 / v;
    return v >= 0.0 ? 1e8 : -1e8;
}
vec3 safe_inv_dir(vec3 d) {
    return vec3(safe_inv_component(d.x), safe_inv_component(d.y), safe_inv_component(d.z));
}

bool intersect_sphere(vec3 ro, vec3 rd, vec3 center, float radius, out float t0, out float t1) {
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    t0 = -b - h; t1 = -b + h;
    return true;
}

struct Hit { float t; int sphere_idx; };

bool intersect_bvh(vec3 ro, vec3 rd, out Hit hit) {
    vec3 inv_rd = safe_inv_dir(rd);
    hit.t = 1e38; hit.sphere_idx = -1;
    if (u_bvh_node_count <= 0) return false;

    int stack[MAX_STACK];
    int ptr = 0;
    stack[ptr++] = 0;
    int guard = 0;

    while (ptr > 0 && guard < 10000) {
        ++guard;
        int node_idx = stack[--ptr];
        if (node_idx < 0 || node_idx >= u_bvh_node_count) continue;

        vec4 node0 = texelFetch(u_bvhNodes, node_idx * 2);
        vec4 node1 = texelFetch(u_bvhNodes, node_idx * 2 + 1);
        vec3 bmin = node0.xyz; vec3 bmax = node1.xyz;
        int left = int(node0.w); float right_value = node1.w;

        vec3 t0s = (bmin - ro) * inv_rd; vec3 t1s = (bmax - ro) * inv_rd;
        vec3 tsmaller = min(t0s, t1s); vec3 tbigger = max(t0s, t1s);
        float tmin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
        float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);

        if (tmax < max(0.0, tmin) || tmin > hit.t) continue;

        if (right_value < 0.0) {
            int s_idx = left;
            if (s_idx < 0 || s_idx >= u_sphere_count) continue;
            vec4 pr = texelFetch(u_spheres_pos_rad, s_idx);
            float t0_sph, t1_sph;
            if (intersect_sphere(ro, rd, pr.xyz, pr.w, t0_sph, t1_sph)) {
                float t_hit = t0_sph > 0.001 ? t0_sph : t1_sph;
                if (t_hit > 0.001 && t_hit < hit.t) {
                    hit.t = t_hit; hit.sphere_idx = s_idx;
                }
            }
        } else {
            int right = int(right_value);
            if (left >= 0 && left < u_bvh_node_count && ptr < MAX_STACK) stack[ptr++] = left;
            if (right >= 0 && right < u_bvh_node_count && ptr < MAX_STACK) stack[ptr++] = right;
        }
    }
    return hit.sphere_idx != -1;
}

vec3 trace_ray(vec3 ro, vec3 rd, inout uint seed) {
    vec3 throughput = vec3(1.0);
    vec3 radiance = vec3(0.0);

    for (int i = 0; i < MAX_BOUNCES; i++) {
        Hit hit;
        hit.t = 1e38; hit.sphere_idx = -1;

        // 1. Check BVH
        if (u_bvh_node_count > 0 && u_sphere_count > 0) {
            intersect_bvh(ro, rd, hit);
        }

        // 2. Check Floor Plane
        float denom = dot(u_floorNormal, rd);
        if (abs(denom) > 1e-6) {
            float t_plane = (u_floorD - dot(u_floorNormal, ro)) / denom;
            if (t_plane > 0.001 && t_plane < hit.t) {
                hit.t = t_plane;
                hit.sphere_idx = -2; // Special ID for floor
            }
        }

        if (hit.sphere_idx == -1) {
            if (dot(rd, u_sunDir) > 0.999) radiance += throughput * vec3(10.0);
            else radiance += throughput * vec3(0.5, 0.7, 1.0);
            break;
        }

        vec3 hit_pos = ro + rd * hit.t;
        vec3 normal;
        int mat_idx;

        if (hit.sphere_idx == -2) {
            normal = u_floorNormal;
            mat_idx = u_floorMatIdx;
        } else {
            int s_idx = hit.sphere_idx;
            vec4 pr = texelFetch(u_spheres_pos_rad, s_idx);
            normal = normalize(hit_pos - pr.xyz);
            mat_idx = int(texelFetch(u_spheres_mat_idx, s_idx).x);
        }

        if (mat_idx < 0 || mat_idx >= u_material_count) break;

        vec4 m1 = texelFetch(u_mat_data, mat_idx * 2);
        vec4 m2 = texelFetch(u_mat_data, mat_idx * 2 + 1);
        int type = int(m1.w);

        if (type == 0) { // Diffuse
            vec3 new_dir = normal + random_dir(seed);
            if (dot(new_dir, new_dir) < 1e-8) new_dir = normal;
            rd = normalize(new_dir);
            ro = hit_pos + 0.001 * normal;
            throughput *= m1.xyz;
        } else if (type == 1) { // Metal
            float fuzz = m2.w;
            vec3 refl = reflect(rd, normal);
            vec3 new_dir = refl + fuzz * random_dir(seed);
            if (dot(new_dir, new_dir) < 1e-8) new_dir = refl;
            rd = normalize(new_dir);
            ro = hit_pos + 0.001 * normal;
            throughput *= m1.xyz;
        } else if (type == 2) { // Glass
            float ref_idx = m2.w;
            if (ref_idx <= 0.0) ref_idx = 1.0;
            vec3 outward_normal = normal;
            float ni_over_nt = 1.0 / ref_idx;
            float cosine = dot(-rd, outward_normal);
            if (cosine < 0.0) { outward_normal = -normal; cosine = -cosine; ni_over_nt = ref_idx; }
            cosine = clamp(cosine, 0.0, 1.0);
            float r0 = (1.0 - ref_idx) / (1.0 + ref_idx); r0 = r0 * r0;
            float reflect_prob = r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
            if (random(seed) < reflect_prob) {
                rd = reflect(rd, outward_normal);
                ro = hit_pos + 0.001 * outward_normal;
            } else {
                vec3 refr_dir = refract(rd, outward_normal, ni_over_nt);
                if (length(refr_dir) == 0.0) {
                    rd = reflect(rd, outward_normal); ro = hit_pos + 0.001 * outward_normal;
                } else {
                    rd = normalize(refr_dir); ro = hit_pos - 0.001 * outward_normal;
                    if (ni_over_nt > 1.0) throughput *= exp(-m1.xyz * hit.t);
                }
            }
        } else {
            break;
        }

        if (abs(rd.x) < 1e-6) rd.x = (rd.x >= 0.0) ? 1e-6 : -1e-6;
        if (abs(rd.y) < 1e-6) rd.y = (rd.y >= 0.0) ? 1e-6 : -1e-6;
        if (abs(rd.z) < 1e-6) rd.z = (rd.z >= 0.0) ? 1e-6 : -1e-6;
        rd = normalize(rd);
    }
    return radiance;
}

void main() {
    if (u_mode == 1) {
        FragColor = texture(u_accumTex, TexCoord);
        return;
    }

    uint seed = hash3(uint(gl_FragCoord.x), uint(gl_FragCoord.y), uint(u_frameIndex * 1000 + u_time));
    vec2 uv = (TexCoord * 2.0 - 1.0);
    uv.x *= 16.0 / 9.0;
    vec3 rd = normalize(uv.x * u_camRight + uv.y * u_camUp + u_camDir);
    vec3 ro = u_camPos;

    // Trace 1 sample per frame for accumulation speed
    vec3 col = trace_ray(ro, rd, seed);
    
    vec3 prev = texture(u_accumTex, TexCoord).rgb;
    if (u_resetAccum == 1 || u_frameIndex <= 1) {
        FragColor = vec4(col, 1.0);
    } else {
        float weight = 1.0 / float(u_frameIndex);
        FragColor = vec4(mix(prev, col, weight), 1.0);
    }
}
)";

// ---------------------------------------------------------
// OpenGL helpers
// ---------------------------------------------------------

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[2048];
        glGetShaderInfoLog(shader, sizeof(info_log), NULL, info_log);
        cerr << "[shader] Compilation failed:\n" << info_log << endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

struct RuntimeConfig {
    int width = 200;
    int height = 200;
    int target_fps = 15; // Cap to prevent CPU starvation/lag
    bool vsync = false;
};

static RuntimeConfig parse_config(int argc, char** argv) {
    RuntimeConfig cfg;
    // Simple parsing omitted for brevity, defaults are fine
    return cfg;
}

class FrameLimiter {
public:
    explicit FrameLimiter(int fps)
        : frame_time(fps > 0 ? chrono::duration<double>(1.0 / double(fps)) : chrono::duration<double>(0.0)),
          next_frame(chrono::steady_clock::now()) {}
    void wait() {
        if (frame_time.count() <= 0.0) return;
        next_frame += chrono::duration_cast<chrono::steady_clock::duration>(frame_time);
        auto now = chrono::steady_clock::now();
        if (now < next_frame) this_thread::sleep_for(next_frame - now);
        else next_frame = now;
    }
private:
    chrono::duration<double> frame_time;
    chrono::steady_clock::time_point next_frame;
};

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

int main(int argc, char** argv) {
    const RuntimeConfig cfg = parse_config(argc, argv);
    FrameLimiter limiter(cfg.target_fps);

    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(cfg.width, cfg.height, "Real-time Path Tracer (Space to Pause)", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(cfg.vsync ? 1 : 0);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { cerr << "GLEW Failed" << endl; return -1; }

    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    if (!vs || !fs) return -1;

    GLuint program = glCreateProgram();
    glAttachShader(program, vs); glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs); glDeleteShader(fs);

    GLint linked; glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) { cerr << "Link Failed" << endl; return -1; }

    glUseProgram(program);

    auto set_sampler = [&](const char* name, GLint unit) {
        GLint loc = glGetUniformLocation(program, name);
        if (loc != -1) glUniform1i(loc, unit);
    };
    set_sampler("u_spheres_pos_rad", 0);
    set_sampler("u_spheres_mat_idx", 1);
    set_sampler("u_mat_data", 2);
    set_sampler("u_bvhNodes", 3);
    set_sampler("u_accumTex", 4);

    const GLint loc_cam_pos = glGetUniformLocation(program, "u_camPos");
    const GLint loc_cam_dir = glGetUniformLocation(program, "u_camDir");
    const GLint loc_cam_right = glGetUniformLocation(program, "u_camRight");
    const GLint loc_cam_up = glGetUniformLocation(program, "u_camUp");
    const GLint loc_time = glGetUniformLocation(program, "u_time");
    const GLint loc_sphere_count = glGetUniformLocation(program, "u_sphere_count");
    const GLint loc_material_count = glGetUniformLocation(program, "u_material_count");
    const GLint loc_bvh_node_count = glGetUniformLocation(program, "u_bvh_node_count");
    const GLint loc_sun_dir = glGetUniformLocation(program, "u_sunDir");
    
    const GLint loc_floorNormal = glGetUniformLocation(program, "u_floorNormal");
    const GLint loc_floorD = glGetUniformLocation(program, "u_floorD");
    const GLint loc_floorMatIdx = glGetUniformLocation(program, "u_floorMatIdx");

    const GLint loc_mode = glGetUniformLocation(program, "u_mode");
    const GLint loc_frameIndex = glGetUniformLocation(program, "u_frameIndex");
    const GLint loc_resetAccum = glGetUniformLocation(program, "u_resetAccum");

    // Fullscreen triangle
    float tri_vertices[] = { -1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri_vertices), tri_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Ping-pong FBOs for accumulation
    GLuint accumFBO, accumTex[2];
    glGenFramebuffers(1, &accumFBO);
    glGenTextures(2, accumTex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, accumTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cfg.width, cfg.height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    // TBOs setup (identical to before)
    GLuint tbo_pos_rad, tbo_mat_idx, tbo_mat, tbo_bvh;
    GLuint tex_pos_rad, tex_mat_idx, tex_mat, tex_bvh;
    glGenBuffers(1, &tbo_pos_rad); glGenBuffers(1, &tbo_mat_idx); glGenBuffers(1, &tbo_mat); glGenBuffers(1, &tbo_bvh);
    glGenTextures(1, &tex_pos_rad); glGenTextures(1, &tex_mat_idx); glGenTextures(1, &tex_mat); glGenTextures(1, &tex_bvh);
    
    int max_spheres = 1024;
    glBindBuffer(GL_TEXTURE_BUFFER, tbo_pos_rad); glBufferData(GL_TEXTURE_BUFFER, max_spheres * sizeof(vec4), NULL, GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, tex_pos_rad); glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, tbo_pos_rad);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo_mat_idx); glBufferData(GL_TEXTURE_BUFFER, max_spheres * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, tex_mat_idx); glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo_mat_idx);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo_mat); glBufferData(GL_TEXTURE_BUFFER, 256 * 2 * sizeof(vec4), NULL, GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, tex_mat); glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, tbo_mat);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo_bvh); glBufferData(GL_TEXTURE_BUFFER, max_spheres * 4 * sizeof(vec4), NULL, GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, tex_bvh); glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, tbo_bvh);

    vector<Material> materials;
    materials.push_back(make_diffuse(vec3(0.5f, 0.5f, 0.5f))); // 0: Floor
    materials.push_back(make_diffuse(vec3(0.8f, 0.3f, 0.3f))); // 1: Red
    materials.push_back(make_metal(vec3(0.9f, 0.9f, 0.9f), 0.05f)); // 2: Metal
    materials.push_back(make_glass(vec3(0.1f, 0.1f, 0.1f), 1.5f)); // 3: Glass

    vector<vec4> mat_array(materials.size() * 2);
    for (size_t i = 0; i < materials.size(); ++i) {
        mat_array[i * 2 + 0] = vec4(materials[i].m1[0], materials[i].m1[1], materials[i].m1[2], materials[i].m1[3]);
        mat_array[i * 2 + 1] = vec4(materials[i].m2[0], materials[i].m2[1], materials[i].m2[2], materials[i].m2[3]);
    }
    glBindBuffer(GL_TEXTURE_BUFFER, tbo_mat);
    glBufferSubData(GL_TEXTURE_BUFFER, 0, mat_array.size() * sizeof(vec4), mat_array.data());

    vector<AnimSphere> anim_spheres;
    // No floor sphere! We use analytical plane intersection.
    anim_spheres.push_back({vec3(-1.2f, 0.2f, 0.0f), 0.5f, 1, 0, 0.0f, 0.0f}); // Red Diffuse
    anim_spheres.push_back({vec3(0.0f, 0.2f, 0.0f), 0.5f, 2, 2, 1.0f, 0.5f}); // Metal Bouncing
    anim_spheres.push_back({vec3(1.2f, 0.2f, 0.0f), 0.5f, 3, 3, 0.5f, 1.0f}); // Glass Orbit

    vector<BVHNode> bvh_nodes; vector<int> primitive_indices;
    vector<Sphere> current_spheres; vector<int> current_mat_indices;
    vector<vec4> pos_rad_array; vector<float> mat_idx_floats; vector<vec4> bvh_array;

    bool animationPaused = false;
    bool spacePressed = false;
    bool resetAccum = true;
    int frameIndex = 1;
    int readAccum = 0;
    float staticTime = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents(); // MOVED TO TOP: Fixes window drag lag!

        float currentTime = (float)glfwGetTime();
        
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !spacePressed) {
            animationPaused = !animationPaused;
            spacePressed = true;
            resetAccum = true; frameIndex = 1;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) spacePressed = false;

        float time;
        bool camera_moved = true;
        if (animationPaused) {
            time = staticTime;
            camera_moved = false; // Allow accumulation to converge
        } else {
            time = currentTime;
            staticTime = time;
        }

        if (camera_moved) { resetAccum = true; frameIndex = 1; }

        current_spheres.clear(); current_mat_indices.clear();
        for (const auto& as : anim_spheres) {
            Sphere s; s.pos = as.base_pos; s.radius = as.base_radius;
            if (as.anim_type == 1) s.pos.x += sinf(time * as.speed) * as.amp;
            else if (as.anim_type == 2) s.pos.y += fabsf(sinf(time * as.speed)) * as.amp;
            else if (as.anim_type == 3) { s.pos.x += cosf(time * as.speed) * as.amp; s.pos.z += sinf(time * as.speed) * as.amp; }
            current_spheres.push_back(s); current_mat_indices.push_back(as.mat_idx);
        }

        bvh_nodes.clear(); primitive_indices.clear();
        for (int i = 0; i < (int)current_spheres.size(); ++i) primitive_indices.push_back(i);
        if (!current_spheres.empty()) build_bvh(0, (int)current_spheres.size(), current_spheres, primitive_indices, bvh_nodes);

        pos_rad_array.resize(current_spheres.size());
        for (size_t i = 0; i < current_spheres.size(); ++i) pos_rad_array[i] = vec4(current_spheres[i].pos.x, current_spheres[i].pos.y, current_spheres[i].pos.z, current_spheres[i].radius);
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_pos_rad); glBufferSubData(GL_TEXTURE_BUFFER, 0, pos_rad_array.size() * sizeof(vec4), pos_rad_array.data());

        mat_idx_floats.resize(current_spheres.size());
        for (size_t i = 0; i < current_mat_indices.size(); ++i) mat_idx_floats[i] = (float)current_mat_indices[i];
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_mat_idx); glBufferSubData(GL_TEXTURE_BUFFER, 0, mat_idx_floats.size() * sizeof(float), mat_idx_floats.data());

        bvh_array.resize(bvh_nodes.size() * 2);
        for (size_t i = 0; i < bvh_nodes.size(); ++i) {
            bvh_array[i * 2 + 0] = vec4(bvh_nodes[i].min_x, bvh_nodes[i].min_y, bvh_nodes[i].min_z, (float)bvh_nodes[i].left);
            bvh_array[i * 2 + 1] = vec4(bvh_nodes[i].max_x, bvh_nodes[i].max_y, bvh_nodes[i].max_z, (float)bvh_nodes[i].right);
        }
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_bvh); glBufferSubData(GL_TEXTURE_BUFFER, 0, bvh_array.size() * sizeof(vec4), bvh_array.data());

        vec3 cam_pos = vec3(sinf(time * 0.3f) * 3.0f, 1.5f, cosf(time * 0.3f) * 3.0f);
        vec3 cam_target = vec3(0.0f, 0.2f, 0.0f); // FIXED: Look directly at spheres
        vec3 cam_dir = normalize(cam_target - cam_pos);
        vec3 cam_right = normalize(cross(cam_dir, vec3(0.0f, 1.0f, 0.0f)));
        vec3 cam_up = cross(cam_right, cam_dir);

        int writeAccum = 1 - readAccum;

        // Pass 1: Accumulate
        glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex[writeAccum], 0);
        glViewport(0, 0, cfg.width, cfg.height);

        glUseProgram(program);
        glUniform1i(loc_mode, 0);
        glUniform1i(loc_frameIndex, frameIndex);
        glUniform1i(loc_resetAccum, resetAccum ? 1 : 0);

        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, accumTex[readAccum]);
        
        glUniform3fv(loc_cam_pos, 1, &cam_pos.x); glUniform3fv(loc_cam_dir, 1, &cam_dir.x);
        glUniform3fv(loc_cam_right, 1, &cam_right.x); glUniform3fv(loc_cam_up, 1, &cam_up.x);
        glUniform1f(loc_time, time);
        glUniform1i(loc_sphere_count, (GLint)current_spheres.size());
        glUniform1i(loc_material_count, (GLint)materials.size());
        glUniform1i(loc_bvh_node_count, (GLint)bvh_nodes.size());

        vec3 sun_dir = normalize(vec3(0.5f, 1.0f, 0.2f));
        glUniform3fv(loc_sun_dir, 1, &sun_dir.x);

        // Floor Plane
        vec3 floorNormal = vec3(0.0f, 1.0f, 0.0f);
        glUniform3fv(loc_floorNormal, 1, &floorNormal.x);
        glUniform1f(loc_floorD, 0.0f);
        glUniform1i(loc_floorMatIdx, 0);

        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_BUFFER, tex_pos_rad);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_BUFFER, tex_mat_idx);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_BUFFER, tex_mat);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_BUFFER, tex_bvh);

        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Pass 2: Display
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, cfg.width, cfg.height);
        glClear(GL_COLOR_BUFFER_BIT);

        glUniform1i(loc_mode, 1);
        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, accumTex[writeAccum]);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);

        readAccum = writeAccum;
        resetAccum = false;
        if (!camera_moved) frameIndex++;
usleep(100);
        limiter.wait();
    }

    glfwTerminate();
    return 0;
}