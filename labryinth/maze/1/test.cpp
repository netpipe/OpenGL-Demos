#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

static const float PI = 3.14159265358979323846f;

enum Direction
{
    DIR_XP = 0,
    DIR_XM = 1,
    DIR_YP = 2,
    DIR_YM = 3,
    DIR_ZP = 4,
    DIR_ZM = 5
};

static const int DX[6] = { 1, -1,  0,  0,  0,  0 };
static const int DY[6] = { 0,  0,  1, -1,  0,  0 }; // grid Y maps to world Z
static const int DZ[6] = { 0,  0,  0,  0,  1, -1 }; // grid Z maps to world Y / layer
static const int OPP[6] = { 1, 0, 3, 2, 5, 4 };

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
};

static float dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static float length(const Vec3& v)
{
    return std::sqrt(dot(v, v));
}

static Vec3 normalize(const Vec3& v)
{
    float l = length(v);
    if (l < 1e-6f) return Vec3(0.0f, 0.0f, 1.0f);
    return v * (1.0f / l);
}

struct Mat4
{
    float m[16];

    Mat4()
    {
        for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    }
};

static Mat4 identity()
{
    Mat4 r;
    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 multiply(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
            {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

static Mat4 translate(float x, float y, float z)
{
    Mat4 r = identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

static Mat4 scale(float x, float y, float z)
{
    Mat4 r = identity();
    r.m[0] = x;
    r.m[5] = y;
    r.m[10] = z;
    return r;
}

static Mat4 perspective(float fovyRadians, float aspect, float zNear, float zFar)
{
    Mat4 r;
    float f = 1.0f / std::tan(fovyRadians * 0.5f);

    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zFar + zNear) / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);

    return r;
}

static Mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar)
{
    Mat4 r;

    r.m[0] = 2.0f / (right - left);
    r.m[5] = 2.0f / (top - bottom);
    r.m[10] = -2.0f / (zFar - zNear);
    r.m[12] = -(right + left) / (right - left);
    r.m[13] = -(top + bottom) / (top - bottom);
    r.m[14] = -(zFar + zNear) / (zFar - zNear);
    r.m[15] = 1.0f;

    return r;
}

static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up)
{
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r;

    r.m[0] = s.x;
    r.m[4] = s.y;
    r.m[8] = s.z;
    r.m[12] = -dot(s, eye);

    r.m[1] = u.x;
    r.m[5] = u.y;
    r.m[9] = u.z;
    r.m[13] = -dot(u, eye);

    r.m[2] = -f.x;
    r.m[6] = -f.y;
    r.m[10] = -f.z;
    r.m[14] = dot(f, eye);

    r.m[15] = 1.0f;

    return r;
}

struct Vertex
{
    float pos[3];
    float nrm[3];
    float col[3];
};

struct MeshBuilder
{
    std::vector<Vertex> verts;

    void pushVert(const Vec3& p, const Vec3& n, const Vec3& c)
    {
        Vertex v{};
        v.pos[0] = p.x; v.pos[1] = p.y; v.pos[2] = p.z;
        v.nrm[0] = n.x; v.nrm[1] = n.y; v.nrm[2] = n.z;
        v.col[0] = c.x; v.col[1] = c.y; v.col[2] = c.z;
        verts.push_back(v);
    }

    void quad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d,
              const Vec3& n, const Vec3& col)
    {
        pushVert(a, n, col);
        pushVert(b, n, col);
        pushVert(c, n, col);

        pushVert(a, n, col);
        pushVert(c, n, col);
        pushVert(d, n, col);
    }

    void box(const Vec3& mn, const Vec3& mx, const Vec3& col)
    {
        // +X
        quad(Vec3(mx.x, mn.y, mn.z), Vec3(mx.x, mx.y, mn.z),
             Vec3(mx.x, mx.y, mx.z), Vec3(mx.x, mn.y, mx.z),
             Vec3(1.0f, 0.0f, 0.0f), col);

        // -X
        quad(Vec3(mn.x, mn.y, mx.z), Vec3(mn.x, mx.y, mx.z),
             Vec3(mn.x, mx.y, mn.z), Vec3(mn.x, mn.y, mn.z),
             Vec3(-1.0f, 0.0f, 0.0f), col);

        // +Y
        quad(Vec3(mn.x, mx.y, mn.z), Vec3(mn.x, mx.y, mx.z),
             Vec3(mx.x, mx.y, mx.z), Vec3(mx.x, mx.y, mn.z),
             Vec3(0.0f, 1.0f, 0.0f), col);

        // -Y
        quad(Vec3(mn.x, mn.y, mn.z), Vec3(mx.x, mn.y, mn.z),
             Vec3(mx.x, mn.y, mx.z), Vec3(mn.x, mn.y, mx.z),
             Vec3(0.0f, -1.0f, 0.0f), col);

        // +Z
        quad(Vec3(mn.x, mn.y, mx.z), Vec3(mx.x, mn.y, mx.z),
             Vec3(mx.x, mx.y, mx.z), Vec3(mn.x, mx.y, mx.z),
             Vec3(0.0f, 0.0f, 1.0f), col);

        // -Z
        quad(Vec3(mx.x, mn.y, mn.z), Vec3(mn.x, mn.y, mn.z),
             Vec3(mn.x, mx.y, mn.z), Vec3(mx.x, mx.y, mn.z),
             Vec3(0.0f, 0.0f, -1.0f), col);
    }
};

struct Cell
{
    bool open[6];
    unsigned char vtype[6]; // for vertical dirs: 0 none, 1 ladder, 2 stairs

    Cell()
    {
        for (int i = 0; i < 6; ++i)
        {
            open[i] = false;
            vtype[i] = 0;
        }
    }
};

struct Config
{
    int width = 12;
    int length = 12;
    int layers = 3;
    float height = 3.0f;

    bool ghosts = true;
    bool multiEntranceExit = false;
    int entranceCount = 3;
    int exitCount = 3;

    bool randomStart = true;
    bool randomGoal = true;
    bool revealGoal = true;
    bool showSolution = false;

    std::uint32_t seed = 12345;
};

struct Maze
{
    int W = 0;
    int L = 0;
    int H = 0;
    float fh = 3.0f;

    std::vector<Cell> cells;

    std::vector<int> entrances;
    std::vector<int> exits;

    int startEntranceIndex = 0;
    int goalIndex = 0;

    int startCell = -1;
    int goalCell = -1;

    std::vector<int> solution;

    std::mt19937 rng;

    int idx(int x, int y, int z) const
    {
        return (z * L + y) * W + x;
    }

    bool inBounds(int x, int y, int z) const
    {
        return x >= 0 && x < W && y >= 0 && y < L && z >= 0 && z < H;
    }

    int degree(int i) const
    {
        int x = i % W;
        int y = (i / W) % L;
        int z = i / (W * L);

        int d = 0;
        for (int dir = 0; dir < 6; ++dir)
        {
            int nx = x + DX[dir];
            int ny = y + DY[dir];
            int nz = z + DZ[dir];
            if (inBounds(nx, ny, nz) && cells[i].open[dir]) ++d;
        }
        return d;
    }

    std::vector<int> bfsDist(int start) const
    {
        int N = W * L * H;
        std::vector<int> dist(N, -1);
        std::queue<int> q;

        dist[start] = 0;
        q.push(start);

        while (!q.empty())
        {
            int cur = q.front();
            q.pop();

            int x = cur % W;
            int y = (cur / W) % L;
            int z = cur / (W * L);

            for (int dir = 0; dir < 6; ++dir)
            {
                if (!cells[cur].open[dir]) continue;

                int nx = x + DX[dir];
                int ny = y + DY[dir];
                int nz = z + DZ[dir];

                if (!inBounds(nx, ny, nz)) continue;

                int ni = idx(nx, ny, nz);
                if (dist[ni] == -1)
                {
                    dist[ni] = dist[cur] + 1;
                    q.push(ni);
                }
            }
        }

        return dist;
    }

    std::vector<int> bfsPath(int start, int goal) const
    {
        int N = W * L * H;
        std::vector<int> parent(N, -1);
        std::queue<int> q;

        parent[start] = start;
        q.push(start);

        while (!q.empty())
        {
            int cur = q.front();
            q.pop();

            if (cur == goal) break;

            int x = cur % W;
            int y = (cur / W) % L;
            int z = cur / (W * L);

            for (int dir = 0; dir < 6; ++dir)
            {
                if (!cells[cur].open[dir]) continue;

                int nx = x + DX[dir];
                int ny = y + DY[dir];
                int nz = z + DZ[dir];

                if (!inBounds(nx, ny, nz)) continue;

                int ni = idx(nx, ny, nz);
                if (parent[ni] == -1)
                {
                    parent[ni] = cur;
                    q.push(ni);
                }
            }
        }

        std::vector<int> path;
        if (parent[goal] == -1) return path;

        int cur = goal;
        while (cur != start)
        {
            path.push_back(cur);
            cur = parent[cur];
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        return path;
    }

    void buildSolution()
    {
        if (startCell >= 0 && goalCell >= 0)
            solution = bfsPath(startCell, goalCell);
        else
            solution.clear();
    }

    void selectPoints(const Config& cfg)
    {
        int N = W * L * H;

        std::vector<int> boundaryLeaves;
        std::vector<int> leaves;
        std::vector<int> all;

        all.reserve(N);

        for (int i = 0; i < N; ++i)
        {
            all.push_back(i);

            if (degree(i) != 1) continue;

            int x = i % W;
            int y = (i / W) % L;
            int z = i / (W * L);

            bool boundary = (x == 0 || x == W - 1 ||
                             y == 0 || y == L - 1 ||
                             z == 0 || z == H - 1);

            if (boundary) boundaryLeaves.push_back(i);
            else leaves.push_back(i);
        }

        std::vector<int> candidates;

        if (boundaryLeaves.size() >= 2)
            candidates = boundaryLeaves;
        else if (leaves.size() >= 2)
            candidates = leaves;
        else
            candidates = all;

        std::shuffle(candidates.begin(), candidates.end(), rng);

        int avail = (int)candidates.size();

        if (avail < 2)
        {
            startCell = 0;
            goalCell = (N > 1 ? N - 1 : 0);
            entrances = { startCell };
            exits = { goalCell };
            startEntranceIndex = 0;
            goalIndex = 0;
            buildSolution();
            return;
        }

        if (!cfg.multiEntranceExit)
        {
            startCell = candidates[0];

            std::vector<int> dist = bfsDist(startCell);
            int far = startCell;
            for (int i = 0; i < N; ++i)
            {
                if (dist[i] > dist[far]) far = i;
            }

            goalCell = far;

            entrances = { startCell };
            exits = { goalCell };
            startEntranceIndex = 0;
            goalIndex = 0;
        }
        else
        {
            int eCount = std::min(std::max(1, cfg.entranceCount), std::max(1, avail / 2));
            int xCount = std::min(std::max(1, cfg.exitCount), std::max(1, avail - eCount));

            if (eCount + xCount > avail) xCount = avail - eCount;
            if (xCount < 1)
            {
                eCount = avail - 1;
                xCount = 1;
            }

            entrances.clear();
            exits.clear();

            for (int i = 0; i < eCount && i < avail; ++i)
                entrances.push_back(candidates[i]);

            for (int i = 0; i < xCount && eCount + i < avail; ++i)
                exits.push_back(candidates[eCount + i]);

            if (entrances.empty()) entrances.push_back(candidates[0]);
            if (exits.empty()) exits.push_back(candidates.back());

            startEntranceIndex = cfg.randomStart ? (int)(rng() % entrances.size()) : 0;
            goalIndex = cfg.randomGoal ? (int)(rng() % exits.size()) : 0;

            startCell = entrances[startEntranceIndex];
            goalCell = exits[goalIndex];

            if (startCell == goalCell && exits.size() > 1)
            {
                goalIndex = (goalIndex + 1) % (int)exits.size();
                goalCell = exits[goalIndex];
            }
        }

        buildSolution();
    }

    void generate(const Config& cfg)
    {
        W = std::max(2, cfg.width);
        L = std::max(2, cfg.length);
        H = std::max(1, cfg.layers);
        fh = std::max(1.5f, cfg.height);

        int N = W * L * H;
        cells.assign(N, Cell());

        if (cfg.seed != 0)
            rng.seed(cfg.seed);
        else
            rng.seed((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

        // Randomized DFS / recursive backtracker in 3D.
        std::vector<char> visited(N, 0);
        std::vector<int> st;
        st.reserve(N);

        std::uniform_int_distribution<int> startDist(0, N - 1);
        int start = startDist(rng);

        visited[start] = 1;
        st.push_back(start);

        struct Nb
        {
            int idx;
            int dir;
        };

        while (!st.empty())
        {
            int cur = st.back();

            int cx = cur % W;
            int cy = (cur / W) % L;
            int cz = cur / (W * L);

            std::vector<Nb> nb;
            nb.reserve(6);

            for (int dir = 0; dir < 6; ++dir)
            {
                int nx = cx + DX[dir];
                int ny = cy + DY[dir];
                int nz = cz + DZ[dir];

                if (!inBounds(nx, ny, nz)) continue;

                int ni = idx(nx, ny, nz);
                if (!visited[ni]) nb.push_back({ ni, dir });
            }

            if (nb.empty())
            {
                st.pop_back();
                continue;
            }

            Nb chosen = nb[rng() % nb.size()];

            cells[cur].open[chosen.dir] = true;
            cells[chosen.idx].open[OPP[chosen.dir]] = true;

            if (chosen.dir >= 4)
            {
                unsigned char vt = (rng() & 1) ? 1 : 2; // 1 ladder, 2 stairs
                cells[cur].vtype[chosen.dir] = vt;
                cells[chosen.idx].vtype[OPP[chosen.dir]] = vt;
            }

            visited[chosen.idx] = 1;
            st.push_back(chosen.idx);
        }

        selectPoints(cfg);
    }
};

struct Player
{
    float x = 0.5f;
    float z = 0.5f;
    float feetY = 0.0f;

    float yaw = 0.0f;
    float pitch = 0.0f;

    bool climbing = false;
    float climbTarget = 0.0f;
};

struct Ghost
{
    int cell = 0;
    int layer = 0;

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    int targetCell = 0;
    int prevCell = -1;
    float t = 1.0f;
};

enum class AppState
{
    Preview,
    Play,
    Win,
    Dead
};

static AppState appState = AppState::Preview;
static Config cfg;
static Maze maze;
static Player player;
static std::vector<Ghost> ghosts;

static GLFWwindow* g_window = nullptr;

static double lastX = 0.0;
static double lastY = 0.0;
static bool leftDrag = false;
static bool mouseCaptured = false;

static float orbitYaw = 0.7f;
static float orbitPitch = 0.45f;
static float orbitDist = 35.0f;

static float g_time = 0.0f;

static GLuint programID = 0;
static GLuint vaoStatic = 0, vboStatic = 0;
static GLuint vaoCube = 0, vboCube = 0;
static GLuint vaoQuad = 0, vboQuad = 0;
static GLsizei staticCount = 0;

struct ShaderUniforms
{
    GLint model = -1;
    GLint view = -1;
    GLint proj = -1;
    GLint lightDir = -1;
    GLint tint = -1;
    GLint alpha = -1;
};

static ShaderUniforms U;

static const char* vertexSrc = R"(#version 410 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec3 FragPos;
out vec3 Normal;
out vec3 VertColor;

void main()
{
    vec4 wp = model * vec4(aPos, 1.0);
    FragPos = wp.xyz;
    gl_Position = proj * view * wp;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    VertColor = aColor;
}
)";

static const char* fragmentSrc = R"(#version 410 core

in vec3 FragPos;
in vec3 Normal;
in vec3 VertColor;

uniform vec3 lightDir;
uniform vec3 tint;
uniform float alpha;

out vec4 fragColor;

void main()
{
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightDir);

    float diff = max(dot(N, L), 0.0);
    vec3 c = VertColor * tint * (0.38 + 0.72 * diff);
    c = min(c, vec3(1.0));

    fragColor = vec4(c, alpha);
}
)";

static std::uint32_t seedNow()
{
    return (std::uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);

    if (!ok)
    {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> info((size_t)std::max(len, 1));
        glGetShaderInfoLog(s, len, nullptr, info.data());
        std::fprintf(stderr, "Shader compile error:\n%s\n", info.data());
    }

    return s;
}

static GLuint linkProgram(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);

    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);

    if (!ok)
    {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> info((size_t)std::max(len, 1));
        glGetProgramInfoLog(p, len, nullptr, info.data());
        std::fprintf(stderr, "Program link error:\n%s\n", info.data());
        glDeleteProgram(p);
        return 0;
    }

    return p;
}

static void createBufferVAO(GLuint& vao, GLuint& vbo, const std::vector<Vertex>& verts)
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(sizeof(Vertex) * verts.size()),
                 verts.empty() ? nullptr : verts.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*)offsetof(Vertex, pos));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*)offsetof(Vertex, nrm));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*)offsetof(Vertex, col));

    glBindVertexArray(0);
}

static void initGL()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    programID = linkProgram(vs, fs);

    glDeleteShader(vs);
    glDeleteShader(fs);

    if (programID == 0)
    {
        std::fprintf(stderr, "Failed to create shader program.\n");
        std::exit(1);
    }

    U.model = glGetUniformLocation(programID, "model");
    U.view = glGetUniformLocation(programID, "view");
    U.proj = glGetUniformLocation(programID, "proj");
    U.lightDir = glGetUniformLocation(programID, "lightDir");
    U.tint = glGetUniformLocation(programID, "tint");
    U.alpha = glGetUniformLocation(programID, "alpha");

    createBufferVAO(vaoStatic, vboStatic, {});

    MeshBuilder cubeMB;
    cubeMB.box(Vec3(-0.5f, -0.5f, -0.5f), Vec3(0.5f, 0.5f, 0.5f), Vec3(1.0f, 1.0f, 1.0f));
    createBufferVAO(vaoCube, vboCube, cubeMB.verts);

    auto makeQuadVert = [](float x, float y) -> Vertex
    {
        Vertex v{};
        v.pos[0] = x;
        v.pos[1] = y;
        v.pos[2] = 0.0f;

        v.nrm[0] = 0.0f;
        v.nrm[1] = 0.0f;
        v.nrm[2] = 1.0f;

        v.col[0] = 1.0f;
        v.col[1] = 1.0f;
        v.col[2] = 1.0f;

        return v;
    };

    std::vector<Vertex> quadVerts = {
        makeQuadVert(0.0f, 0.0f),
        makeQuadVert(1.0f, 0.0f),
        makeQuadVert(1.0f, 1.0f),

        makeQuadVert(0.0f, 0.0f),
        makeQuadVert(1.0f, 1.0f),
        makeQuadVert(0.0f, 1.0f)
    };

    createBufferVAO(vaoQuad, vboQuad, quadVerts);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

static Vec3 cellMarkerPos(int cellIndex, float yOffset)
{
    int x = cellIndex % maze.W;
    int y = (cellIndex / maze.W) % maze.L;
    int z = cellIndex / (maze.W * maze.L);

    return Vec3((float)x + 0.5f, (float)z * maze.fh + yOffset, (float)y + 0.5f);
}

static Vec3 cellWorldCenter(int cellIndex, int layer)
{
    int x = cellIndex % maze.W;
    int y = (cellIndex / maze.W) % maze.L;

    return Vec3((float)x + 0.5f,
                (float)layer * maze.fh + maze.fh * 0.5f,
                (float)y + 0.5f);
}

static void buildMazeMesh(MeshBuilder& mb)
{
    if (maze.cells.empty()) return;

    const Vec3 colFloor(0.45f, 0.42f, 0.38f);
    const Vec3 colCeil(0.24f, 0.24f, 0.28f);
    const Vec3 colWall(0.66f, 0.63f, 0.58f);
    const Vec3 colExt(0.40f, 0.41f, 0.46f);
    const Vec3 colLadder(0.55f, 0.34f, 0.15f);
    const Vec3 colStair(0.52f, 0.52f, 0.55f);

    auto wallX = [&](float x, float y0, float y1, float z0, float z1,
                     const Vec3& n, const Vec3& c)
    {
        mb.quad(Vec3(x, y0, z0), Vec3(x, y1, z0),
                Vec3(x, y1, z1), Vec3(x, y0, z1), n, c);
    };

    auto wallZ = [&](float z, float y0, float y1, float x0, float x1,
                     const Vec3& n, const Vec3& c)
    {
        mb.quad(Vec3(x0, y0, z), Vec3(x0, y1, z),
                Vec3(x1, y1, z), Vec3(x1, y0, z), n, c);
    };

    auto floorY = [&](float y, float x0, float x1, float z0, float z1,
                      const Vec3& n, const Vec3& c)
    {
        mb.quad(Vec3(x0, y, z0), Vec3(x0, y, z1),
                Vec3(x1, y, z1), Vec3(x1, y, z0), n, c);
    };

    auto addLadder = [&](float x0, float x1, float z0, float z1, float y0, float y1)
    {
        float cx = (x0 + x1) * 0.5f;
        float cz = (z0 + z1) * 0.5f;

        float railOffset = 0.22f;
        float railHalf = 0.04f;
        float rungHalf = 0.03f;

        mb.box(Vec3(cx - railOffset - railHalf, y0, cz - railHalf),
               Vec3(cx - railOffset + railHalf, y1, cz + railHalf),
               colLadder);

        mb.box(Vec3(cx + railOffset - railHalf, y0, cz - railHalf),
               Vec3(cx + railOffset + railHalf, y1, cz + railHalf),
               colLadder);

        for (float yy = y0 + 0.25f; yy < y1 - 0.08f; yy += 0.35f)
        {
            mb.box(Vec3(cx - railOffset, yy, cz - rungHalf),
                   Vec3(cx + railOffset, yy + 0.05f, cz + rungHalf),
                   colLadder);
        }
    };

    auto addStairs = [&](float x0, float x1, float z0, float z1, float y0, float y1)
    {
        float fh = y1 - y0;
        int steps = std::max(3, (int)(fh / 0.28f));
        float sh = fh / (float)steps;
        float sd = (z1 - z0) / (float)steps;

        for (int i = 0; i < steps; ++i)
        {
            float ys = y0 + (float)(i + 1) * sh;
            float zs = z0 + (float)i * sd;
            float ze = z0 + (float)(i + 1) * sd;

            mb.box(Vec3(x0 + 0.08f, ys - 0.07f, zs),
                   Vec3(x1 - 0.08f, ys, ze),
                   colStair);

            if (i > 0)
            {
                float ry = y0 + (float)i * sh;
                mb.box(Vec3(x0 + 0.08f, ry - 0.07f, zs - 0.02f),
                       Vec3(x1 - 0.08f, ry + 0.01f, zs + 0.02f),
                       colStair);
            }
        }
    };

    for (int z = 0; z < maze.H; ++z)
    {
        for (int y = 0; y < maze.L; ++y)
        {
            for (int x = 0; x < maze.W; ++x)
            {
                const Cell& c = maze.cells[maze.idx(x, y, z)];

                float x0 = (float)x;
                float x1 = x0 + 1.0f;
                float z0 = (float)y;
                float z1 = z0 + 1.0f;
                float y0 = (float)z * maze.fh;
                float y1 = (float)(z + 1) * maze.fh;

                // Floor / intermediate slab.
                if (z == 0)
                {
                    floorY(y0, x0, x1, z0, z1, Vec3(0.0f, 1.0f, 0.0f), colFloor);
                }
                else if (!c.open[DIR_ZM])
                {
                    floorY(y0, x0, x1, z0, z1, Vec3(0.0f, 1.0f, 0.0f), colFloor);
                }

                // Top ceiling.
                if (z == maze.H - 1)
                {
                    floorY(y1, x0, x1, z0, z1, Vec3(0.0f, -1.0f, 0.0f), colCeil);
                }

                // Walls.
                if (x == 0)
                {
                    wallX(x0, y0, y1, z0, z1, Vec3(1.0f, 0.0f, 0.0f), colExt);
                }

                if (x == maze.W - 1 || !c.open[DIR_XP])
                {
                    Vec3 col = (x == maze.W - 1) ? colExt : colWall;
                    wallX(x1, y0, y1, z0, z1, Vec3(1.0f, 0.0f, 0.0f), col);
                }

                if (y == 0)
                {
                    wallZ(z0, y0, y1, x0, x1, Vec3(0.0f, 0.0f, 1.0f), colExt);
                }

                if (y == maze.L - 1 || !c.open[DIR_YP])
                {
                    Vec3 col = (y == maze.L - 1) ? colExt : colWall;
                    wallZ(z1, y0, y1, x0, x1, Vec3(0.0f, 0.0f, 1.0f), col);
                }

                // Vertical connector: draw only from lower cell upward.
                if (c.open[DIR_ZP])
                {
                    unsigned char vt = c.vtype[DIR_ZP];
                    if (vt == 0) vt = c.vtype[DIR_ZM];
                    if (vt == 0) vt = 2;

                    if (vt == 1)
                        addLadder(x0, x1, z0, z1, y0, y1);
                    else
                        addStairs(x0, x1, z0, z1, y0, y1);
                }
            }
        }
    }
}

static void rebuildStaticMesh()
{
    MeshBuilder mb;
    buildMazeMesh(mb);

    staticCount = (GLsizei)mb.verts.size();

    glBindBuffer(GL_ARRAY_BUFFER, vboStatic);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(sizeof(Vertex) * mb.verts.size()),
                 mb.verts.empty() ? nullptr : mb.verts.data(),
                 GL_STATIC_DRAW);
}

static int getPlayerLayer()
{
    if (maze.H <= 0 || maze.fh <= 0.0f) return 0;

    int l = (int)std::floor(player.feetY / maze.fh + 0.001f);
    if (l < 0) l = 0;
    if (l >= maze.H) l = maze.H - 1;
    return l;
}

static bool canStand(float x, float z, int layer)
{
    if (layer < 0 || layer >= maze.H) return false;

    int gx = (int)std::floor(x);
    int gy = (int)std::floor(z);

    if (gx < 0 || gx >= maze.W) return false;
    if (gy < 0 || gy >= maze.L) return false;

    const Cell& c = maze.cells[maze.idx(gx, gy, layer)];

    const float r = 0.24f;

    if (x + r > gx + 1.0f && !c.open[DIR_XP]) return false;
    if (x - r < gx + 0.0f && !c.open[DIR_XM]) return false;

    if (z + r > gy + 1.0f && !c.open[DIR_YP]) return false;
    if (z - r < gy + 0.0f && !c.open[DIR_YM]) return false;

    return true;
}

static bool canClimbUp()
{
    int layer = getPlayerLayer();
    if (layer >= maze.H - 1) return false;

    int gx = (int)std::floor(player.x);
    int gy = (int)std::floor(player.z);

    if (gx < 0 || gx >= maze.W) return false;
    if (gy < 0 || gy >= maze.L) return false;

    const Cell& c = maze.cells[maze.idx(gx, gy, layer)];

    return c.open[DIR_ZP] &&
           std::fabs(player.feetY - (float)layer * maze.fh) < 0.25f;
}

static bool canClimbDown()
{
    int layer = getPlayerLayer();
    if (layer <= 0) return false;

    int gx = (int)std::floor(player.x);
    int gy = (int)std::floor(player.z);

    if (gx < 0 || gx >= maze.W) return false;
    if (gy < 0 || gy >= maze.L) return false;

    const Cell& c = maze.cells[maze.idx(gx, gy, layer)];

    return c.open[DIR_ZM] &&
           std::fabs(player.feetY - (float)layer * maze.fh) < 0.25f;
}

static void resetPlayer()
{
    if (maze.startCell < 0) return;

    int sc = maze.startCell;

    int x = sc % maze.W;
    int y = (sc / maze.W) % maze.L;
    int z = sc / (maze.W * maze.L);

    player.x = (float)x + 0.5f;
    player.z = (float)y + 0.5f;
    player.feetY = (float)z * maze.fh;

    player.yaw = 0.0f;
    player.pitch = 0.0f;
    player.climbing = false;
    player.climbTarget = player.feetY;

    const Cell& c = maze.cells[sc];

    for (int dir = 0; dir < 4; ++dir)
    {
        if (!c.open[dir]) continue;

        if (dir == DIR_XP) player.yaw = 0.0f;
        else if (dir == DIR_XM) player.yaw = PI;
        else if (dir == DIR_YP) player.yaw = PI * 0.5f;
        else if (dir == DIR_YM) player.yaw = -PI * 0.5f;

        break;
    }
}

static void initGhosts()
{
    ghosts.clear();

    if (!cfg.ghosts || maze.H <= 0) return;

    int cellsPerLayer = maze.W * maze.L;
    if (cellsPerLayer <= 0) return;

    for (int z = 0; z < maze.H; ++z)
    {
        int offset = z * cellsPerLayer;
        int chosen = offset + (int)(maze.rng() % cellsPerLayer);

        for (int attempt = 0; attempt < 10; ++attempt)
        {
            if (chosen != maze.startCell && chosen != maze.goalCell) break;
            chosen = offset + (int)(maze.rng() % cellsPerLayer);
        }

        Ghost g;
        g.cell = chosen;
        g.layer = z;
        g.targetCell = chosen;
        g.prevCell = -1;
        g.t = 1.0f;

        Vec3 p = cellWorldCenter(chosen, z);
        g.x = p.x;
        g.y = p.y;
        g.z = p.z;

        ghosts.push_back(g);
    }
}

static void updateGhosts(float dt)
{
    if (ghosts.empty()) return;

    const float ghostSpeed = 1.35f;
    const float catchDistSq = 0.42f;

    for (Ghost& g : ghosts)
    {
        if (g.t >= 1.0f)
        {
            int cx = g.cell % maze.W;
            int cy = (g.cell / maze.W) % maze.L;
            int cz = g.layer;

            const Cell& c = maze.cells[g.cell];

            std::vector<int> all;
            all.reserve(4);

            for (int dir = 0; dir < 4; ++dir)
            {
                if (!c.open[dir]) continue;

                int nx = cx + DX[dir];
                int ny = cy + DY[dir];
                int nz = cz + DZ[dir];

                if (!maze.inBounds(nx, ny, nz)) continue;

                int ni = maze.idx(nx, ny, nz);
                all.push_back(ni);
            }

            std::vector<int> opts;
            opts.reserve(all.size());

            for (int ni : all)
            {
                if (ni != g.prevCell) opts.push_back(ni);
            }

            if (opts.empty()) opts = all;

            if (opts.empty())
            {
                g.targetCell = g.cell;
            }
            else
            {
                g.targetCell = opts[maze.rng() % opts.size()];
            }

            g.t = 0.0f;
        }

        Vec3 from = cellWorldCenter(g.cell, g.layer);
        Vec3 to = cellWorldCenter(g.targetCell, g.layer);

        g.t += dt * ghostSpeed;
        if (g.t > 1.0f) g.t = 1.0f;

        g.x = from.x + (to.x - from.x) * g.t;
        g.z = from.z + (to.z - from.z) * g.t;
        g.y = (float)g.layer * maze.fh + maze.fh * 0.5f +
              std::sin(g_time * 3.0f + (float)g.cell * 0.7f) * 0.12f;

        if (g.t >= 1.0f)
        {
            g.prevCell = g.cell;
            g.cell = g.targetCell;
            g.t = 1.0f;
        }

        if (appState == AppState::Play && !player.climbing)
        {
            int pl = getPlayerLayer();
            if (pl == g.layer)
            {
                float dx = g.x - player.x;
                float dz = g.z - player.z;
                float dy = g.y - (player.feetY + maze.fh * 0.5f);

                if (dx * dx + dz * dz < catchDistSq && std::fabs(dy) < maze.fh * 0.65f)
                {
                    appState = AppState::Dead;
                    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                    mouseCaptured = false;
                    std::printf("Caught by ghost!\n");
                }
            }
        }
    }
}

static void updatePlayer(float dt)
{
    if (appState != AppState::Play) return;

    const float walkSpeed = 3.3f;
    const float sprintSpeed = 6.0f;
    const float climbSpeed = 2.2f;

    bool climbingInput = false;

    if (!player.climbing)
    {
        float f = 0.0f;
        float s = 0.0f;

        if (glfwGetKey(g_window, GLFW_KEY_W) == GLFW_PRESS) f += 1.0f;
        if (glfwGetKey(g_window, GLFW_KEY_S) == GLFW_PRESS) f -= 1.0f;
        if (glfwGetKey(g_window, GLFW_KEY_D) == GLFW_PRESS) s += 1.0f;
        if (glfwGetKey(g_window, GLFW_KEY_A) == GLFW_PRESS) s -= 1.0f;

        float speed = (glfwGetKey(g_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                          ? sprintSpeed
                          : walkSpeed;

        float fx = std::cos(player.yaw);
        float fz = std::sin(player.yaw);

        float rx = -std::sin(player.yaw);
        float rz = std::cos(player.yaw);

        float mx = (fx * f + rx * s) * speed * dt;
        float mz = (fz * f + rz * s) * speed * dt;

        int layer = getPlayerLayer();

        float nx = player.x + mx;
        if (canStand(nx, player.z, layer)) player.x = nx;

        float nz = player.z + mz;
        if (canStand(player.x, nz, layer)) player.z = nz;

        player.feetY = (float)layer * maze.fh;
    }

    if (glfwGetKey(g_window, GLFW_KEY_SPACE) == GLFW_PRESS && canClimbUp())
    {
        player.climbing = true;
        climbingInput = true;
        player.climbTarget = (float)(getPlayerLayer() + 1) * maze.fh;
    }

    if (!climbingInput &&
        (glfwGetKey(g_window, GLFW_KEY_C) == GLFW_PRESS ||
         glfwGetKey(g_window, GLFW_KEY_X) == GLFW_PRESS) &&
        canClimbDown())
    {
        player.climbing = true;
        player.climbTarget = (float)(getPlayerLayer() - 1) * maze.fh;
    }

    if (player.climbing)
    {
        float dir = (player.climbTarget > player.feetY) ? 1.0f : -1.0f;
        player.feetY += dir * climbSpeed * dt;

        bool done = false;
        if (dir > 0.0f && player.feetY >= player.climbTarget) done = true;
        if (dir < 0.0f && player.feetY <= player.climbTarget) done = true;

        if (done)
        {
            player.feetY = player.climbTarget;
            player.climbing = false;
        }
    }

    if (!player.climbing)
    {
        int layer = getPlayerLayer();

        if (std::fabs(player.feetY - (float)layer * maze.fh) < 0.1f)
        {
            int gx = (int)std::floor(player.x);
            int gy = (int)std::floor(player.z);

            if (gx >= 0 && gx < maze.W && gy >= 0 && gy < maze.L)
            {
                int cur = maze.idx(gx, gy, layer);
                if (cur == maze.goalCell)
                {
                    appState = AppState::Win;
                    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                    mouseCaptured = false;
                    std::printf("You escaped the maze!\n");
                }
            }
        }
    }
}

static void enterPreviewMode()
{
    appState = AppState::Preview;
    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    mouseCaptured = false;
    leftDrag = false;
}

static void startPlay()
{
    resetPlayer();
    initGhosts();

    appState = AppState::Play;

    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(g_window, &cx, &cy);
    lastX = cx;
    lastY = cy;

    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    mouseCaptured = true;

    std::printf("Starting play. Reach the marked exit. Avoid ghosts.\n");
}

static void printConfig()
{
    std::printf("Config: %d x %d x %d layers, height %.2f, ghosts %s, multi %s, "
                "entrances %zu, exits %zu, randomStart %s, randomGoal %s, reveal %s, solution %s, seed %u\n",
                maze.W,
                maze.L,
                maze.H,
                maze.fh,
                cfg.ghosts ? "on" : "off",
                cfg.multiEntranceExit ? "on" : "off",
                maze.entrances.size(),
                maze.exits.size(),
                cfg.randomStart ? "on" : "off",
                cfg.randomGoal ? "on" : "off",
                cfg.revealGoal ? "on" : "off",
                cfg.showSolution ? "on" : "off",
                (unsigned)cfg.seed);
}

static void regenerate()
{
    maze.generate(cfg);
    rebuildStaticMesh();

    resetPlayer();
    initGhosts();

    float maxDim = std::max((float)maze.W, std::max((float)maze.L, (float)maze.H * maze.fh));
    orbitDist = maxDim * 1.8f + 10.0f;

    printConfig();
}

static void printControls()
{
    std::printf("\n=== 3D Maze Controls ===\n");
    std::printf("Preview:\n");
    std::printf("  Mouse drag: orbit, Wheel: zoom\n");
    std::printf("  1/2 width, 3/4 length, 5/6 layers, -/= height\n");
    std::printf("  g ghosts, m multi entrances/exits\n");
    std::printf("  o random start, p random goal, v reveal goal, s solution\n");
    std::printf("  n new seed, r regenerate, Enter play, Esc quit\n\n");

    std::printf("Play:\n");
    std::printf("  WASD move, mouse look, Shift sprint\n");
    std::printf("  Space climb up, C/X climb down\n");
    std::printf("  E preview, R regenerate, N new maze, Esc preview\n\n");
}

static void cursorPosCallback(GLFWwindow* w, double x, double y)
{
    if (appState == AppState::Play && mouseCaptured)
    {
        float dx = (float)(x - lastX);
        float dy = (float)(y - lastY);

        player.yaw += dx * 0.0022f;
        player.pitch -= dy * 0.0022f;

        const float limit = 1.5f;
        if (player.pitch > limit) player.pitch = limit;
        if (player.pitch < -limit) player.pitch = -limit;
    }
    else if (appState == AppState::Preview && leftDrag)
    {
        float dx = (float)(x - lastX);
        float dy = (float)(y - lastY);

        orbitYaw -= dx * 0.005f;
        orbitPitch -= dy * 0.005f;

        const float limit = 1.45f;
        if (orbitPitch > limit) orbitPitch = limit;
        if (orbitPitch < -limit) orbitPitch = -limit;
    }

    lastX = x;
    lastY = y;
}

static void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
    (void)mods;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        leftDrag = (action == GLFW_PRESS);

        if (leftDrag && appState == AppState::Preview)
        {
            double x = 0.0, y = 0.0;
            glfwGetCursorPos(w, &x, &y);
            lastX = x;
            lastY = y;
        }
    }
}

static void scrollCallback(GLFWwindow* w, double xoffset, double yoffset)
{
    (void)w;
    (void)xoffset;

    if (appState == AppState::Preview)
    {
        orbitDist *= std::exp((float)(-yoffset * 0.1));
        if (orbitDist < 5.0f) orbitDist = 5.0f;
        if (orbitDist > 300.0f) orbitDist = 300.0f;
    }
}

static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;

    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    if (key == GLFW_KEY_ESCAPE)
    {
        if (appState == AppState::Preview)
        {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
        else
        {
            enterPreviewMode();
        }
        return;
    }

    if (appState == AppState::Preview)
    {
        switch (key)
        {
        case GLFW_KEY_1:
            cfg.width = std::max(4, cfg.width - 1);
            regenerate();
            break;

        case GLFW_KEY_2:
            cfg.width = std::min(40, cfg.width + 1);
            regenerate();
            break;

        case GLFW_KEY_3:
            cfg.length = std::max(4, cfg.length - 1);
            regenerate();
            break;

        case GLFW_KEY_4:
            cfg.length = std::min(40, cfg.length + 1);
            regenerate();
            break;

        case GLFW_KEY_5:
            cfg.layers = std::max(1, cfg.layers - 1);
            regenerate();
            break;

        case GLFW_KEY_6:
            cfg.layers = std::min(12, cfg.layers + 1);
            regenerate();
            break;

        case GLFW_KEY_MINUS:
            cfg.height = std::max(1.5f, cfg.height - 0.25f);
            regenerate();
            break;

        case GLFW_KEY_EQUAL:
            cfg.height = std::min(8.0f, cfg.height + 0.25f);
            regenerate();
            break;

        case GLFW_KEY_G:
            cfg.ghosts = !cfg.ghosts;
            initGhosts();
            printConfig();
            break;

        case GLFW_KEY_M:
            cfg.multiEntranceExit = !cfg.multiEntranceExit;
            regenerate();
            break;

        case GLFW_KEY_O:
            cfg.randomStart = !cfg.randomStart;
            regenerate();
            break;

        case GLFW_KEY_P:
            cfg.randomGoal = !cfg.randomGoal;
            regenerate();
            break;

        case GLFW_KEY_V:
            cfg.revealGoal = !cfg.revealGoal;
            printConfig();
            break;

        case GLFW_KEY_S:
            cfg.showSolution = !cfg.showSolution;
            printConfig();
            break;

        case GLFW_KEY_N:
            cfg.seed = seedNow();
            regenerate();
            break;

        case GLFW_KEY_R:
            regenerate();
            break;

        case GLFW_KEY_ENTER:
        case GLFW_KEY_KP_ENTER:
            startPlay();
            break;

        default:
            break;
        }

        return;
    }

    // Play / Win / Dead
    if (key == GLFW_KEY_E)
    {
        enterPreviewMode();
        return;
    }

    if (key == GLFW_KEY_R)
    {
        regenerate();
        startPlay();
        return;
    }

    if (key == GLFW_KEY_N)
    {
        cfg.seed = seedNow();
        regenerate();
        startPlay();
        return;
    }

    if (appState != AppState::Play &&
        (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER))
    {
        startPlay();
    }
}

static void framebufferSizeCallback(GLFWwindow* w, int width, int height)
{
    (void)w;
    glViewport(0, 0, width, height);
}

static void setMat4(GLint loc, const Mat4& m)
{
    glUniformMatrix4fv(loc, 1, GL_FALSE, m.m);
}

static Mat4 modelTRS(const Vec3& t, const Vec3& s)
{
    Mat4 m = translate(t.x, t.y, t.z);
    Mat4 sc = scale(s.x, s.y, s.z);
    return multiply(m, sc);
}

static void drawCubeInstance(const Vec3& pos, const Vec3& size, const Vec3& tint, float alpha)
{
    Mat4 model = modelTRS(pos, size);

    setMat4(U.model, model);
    glUniform3f(U.tint, tint.x, tint.y, tint.z);
    glUniform1f(U.alpha, alpha);

    if (alpha < 0.99f) glDepthMask(GL_FALSE);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    if (alpha < 0.99f) glDepthMask(GL_TRUE);
}

static void drawHUDQuad(float x, float y, float w, float h, const Vec3& tint, float alpha)
{
    Mat4 model = translate(x, y, 0.0f);
    Mat4 sc = scale(w, h, 1.0f);
    model = multiply(model, sc);

    setMat4(U.model, model);
    glUniform3f(U.tint, tint.x, tint.y, tint.z);
    glUniform1f(U.alpha, alpha);

    glDrawArrays(GL_TRIANGLES, 0, 6);
}

static void updateTitle(float dt)
{
    static float acc = 0.0f;
    acc += dt;
    if (acc < 0.25f) return;
    acc = 0.0f;

    const char* modeStr = "PREVIEW";
    if (appState == AppState::Play) modeStr = "PLAY";
    else if (appState == AppState::Win) modeStr = "WIN";
    else if (appState == AppState::Dead) modeStr = "DEAD";

    int curLayer = getPlayerLayer() + 1;

    char buf[512];
    std::snprintf(buf,
                  sizeof(buf),
                  "%s | %dx%dx%d | h=%.2f | ghosts=%s | multi=%s | level %d/%d | seed=%u",
                  modeStr,
                  maze.W,
                  maze.L,
                  maze.H,
                  maze.fh,
                  cfg.ghosts ? "on" : "off",
                  cfg.multiEntranceExit ? "on" : "off",
                  curLayer,
                  maze.H,
                  (unsigned)cfg.seed);

    glfwSetWindowTitle(g_window, buf);
}

static void render()
{
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(g_window, &fbw, &fbh);

    if (fbw < 1) fbw = 1;
    if (fbh < 1) fbh = 1;

    glViewport(0, 0, fbw, fbh);

    glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_TRUE);

    glUseProgram(programID);

    float aspect = (float)fbw / (float)fbh;

    Mat4 proj;
    Mat4 view;
    Vec3 eye(0.0f, 0.0f, 0.0f);

    if (appState == AppState::Preview)
    {
        proj = perspective(50.0f * PI / 180.0f, aspect, 0.1f, 1000.0f);

        Vec3 center((float)maze.W * 0.5f,
                    (float)maze.H * maze.fh * 0.5f,
                    (float)maze.L * 0.5f);

        float d = orbitDist;
        if (d < 1.0f) d = 1.0f;

        eye = center + Vec3(
            d * std::cos(orbitPitch) * std::sin(orbitYaw),
            d * std::sin(orbitPitch),
            d * std::cos(orbitPitch) * std::cos(orbitYaw)
        );

        view = lookAt(eye, center, Vec3(0.0f, 1.0f, 0.0f));
    }
    else
    {
        proj = perspective(72.0f * PI / 180.0f, aspect, 0.05f, 1000.0f);

        eye = Vec3(player.x, player.feetY + 1.65f, player.z);

        Vec3 front(
            std::cos(player.yaw) * std::cos(player.pitch),
            std::sin(player.pitch),
            std::sin(player.yaw) * std::cos(player.pitch)
        );

        view = lookAt(eye, eye + front, Vec3(0.0f, 1.0f, 0.0f));
    }

    setMat4(U.proj, proj);
    setMat4(U.view, view);
    glUniform3f(U.lightDir, 0.35f, 0.85f, 0.40f);

    // Static maze mesh.
    glBindVertexArray(vaoStatic);
    Mat4 id = identity();
    setMat4(U.model, id);
    glUniform3f(U.tint, 1.0f, 1.0f, 1.0f);
    glUniform1f(U.alpha, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, staticCount);

    // Dynamic objects.
    glBindVertexArray(vaoCube);

    // Entrances.
    for (size_t i = 0; i < maze.entrances.size(); ++i)
    {
        Vec3 pos = cellMarkerPos(maze.entrances[i], 0.25f);
        Vec3 size(0.22f, 0.50f, 0.22f);

        Vec3 tint(0.15f, 0.45f, 0.95f);
        if ((int)i == maze.startEntranceIndex)
            tint = Vec3(0.25f, 0.85f, 1.00f);

        drawCubeInstance(pos, size, tint, 1.0f);
    }

    // Exits.
    for (size_t i = 0; i < maze.exits.size(); ++i)
    {
        Vec3 pos = cellMarkerPos(maze.exits[i], 0.28f);
        Vec3 size(0.28f, 0.56f, 0.28f);

        Vec3 tint;

        if (cfg.revealGoal)
        {
            if ((int)i == maze.goalIndex)
                tint = Vec3(0.15f, 0.95f, 0.25f);
            else
                tint = Vec3(0.75f, 0.15f, 0.15f);
        }
        else
        {
            // All exits look similar when the true exit is hidden.
            tint = Vec3(0.95f, 0.45f, 0.10f);
        }

        drawCubeInstance(pos, size, tint, 1.0f);
    }

    // Solution path in preview.
    if (appState == AppState::Preview && cfg.showSolution)
    {
        for (int cell : maze.solution)
        {
            Vec3 pos = cellMarkerPos(cell, 0.12f);
            Vec3 size(0.12f, 0.12f, 0.12f);
            drawCubeInstance(pos, size, Vec3(1.0f, 0.95f, 0.25f), 0.75f);
        }
    }

    // Ghosts.
    if (cfg.ghosts)
    {
        for (const Ghost& g : ghosts)
        {
            Vec3 pos(g.x, g.y, g.z);
            Vec3 size(0.34f, 0.56f, 0.34f);
            drawCubeInstance(pos, size, Vec3(0.78f, 0.25f, 0.95f), 0.55f);
        }
    }

    // HUD.
    glDisable(GL_DEPTH_TEST);

    Mat4 hudProj = ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
    Mat4 hudView = identity();

    setMat4(U.proj, hudProj);
    setMat4(U.view, hudView);
    glUniform3f(U.lightDir, 0.0f, 0.0f, 1.0f);

    glBindVertexArray(vaoQuad);

    int layers = std::max(1, maze.H);
    float barH = std::min(0.07f, 0.65f / (float)layers);
    float startY = 0.5f - (float)layers * barH * 0.5f;

    int curLayer = getPlayerLayer();

    int startLayer = 0;
    int goalLayer = 0;

    if (maze.startCell >= 0) startLayer = maze.startCell / (maze.W * maze.L);
    if (maze.goalCell >= 0) goalLayer = maze.goalCell / (maze.W * maze.L);

    for (int i = 0; i < layers; ++i)
    {
        Vec3 col(0.32f, 0.32f, 0.36f);

        if (i == startLayer)
            col = Vec3(0.18f, 0.45f, 0.95f);

        if (cfg.revealGoal && i == goalLayer)
            col = Vec3(0.18f, 0.90f, 0.25f);

        if (i == curLayer)
            col = Vec3(1.0f, 1.0f, 1.0f);

        drawHUDQuad(0.02f, startY + (float)i * barH, 0.045f, barH * 0.70f, col, 1.0f);
    }

    // Current-layer highlight frame.
    if (curLayer >= 0 && curLayer < layers)
    {
        float y = startY + (float)curLayer * barH;
        drawHUDQuad(0.015f, y - 0.004f, 0.055f, barH * 0.70f + 0.008f,
                    Vec3(1.0f, 0.9f, 0.2f), 0.75f);
    }

    if (appState == AppState::Dead)
    {
        drawHUDQuad(0.0f, 0.0f, 1.0f, 1.0f, Vec3(0.8f, 0.05f, 0.05f), 0.35f);
    }
    else if (appState == AppState::Win)
    {
        drawHUDQuad(0.0f, 0.0f, 1.0f, 1.0f, Vec3(0.05f, 0.7f, 0.15f), 0.25f);
    }

    glEnable(GL_DEPTH_TEST);

    glfwSwapBuffers(g_window);
}

int main()
{
    if (!glfwInit())
    {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    g_window = glfwCreateWindow(1280, 720, "3D Maze Generator", nullptr, nullptr);
    if (!g_window)
    {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(1);

    glfwSetCursorPosCallback(g_window, cursorPosCallback);
    glfwSetMouseButtonCallback(g_window, mouseButtonCallback);
    glfwSetScrollCallback(g_window, scrollCallback);
    glfwSetKeyCallback(g_window, keyCallback);
    glfwSetFramebufferSizeCallback(g_window, framebufferSizeCallback);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK)
    {
        std::fprintf(stderr, "Failed to initialize GLEW: %s\n",
                     glewGetErrorString(glewErr));
        glfwTerminate();
        return 1;
    }

    // Some GLEW versions generate a harmless invalid enum after init in core profile.
    while (glGetError() == GL_INVALID_ENUM) {}

    if (!GLEW_VERSION_4_1)
    {
        std::fprintf(stderr, "Warning: OpenGL 4.1 not reported by GLEW. Trying anyway.\n");
    }

    initGL();

    cfg.seed = seedNow();
    regenerate();

    printControls();

    double lastFrame = glfwGetTime();

    while (!glfwWindowShouldClose(g_window))
    {
        double now = glfwGetTime();
        float dt = (float)(now - lastFrame);
        lastFrame = now;

        if (dt > 0.1f) dt = 0.1f;
        g_time += dt;

        if (appState == AppState::Play)
        {
            updatePlayer(dt);
            updateGhosts(dt);
        }

        updateTitle(dt);
        render();

        glfwPollEvents();
    }

    glDeleteBuffers(1, &vboStatic);
    glDeleteVertexArrays(1, &vaoStatic);

    glDeleteBuffers(1, &vboCube);
    glDeleteVertexArrays(1, &vaoCube);

    glDeleteBuffers(1, &vboQuad);
    glDeleteVertexArrays(1, &vaoQuad);

    glDeleteProgram(programID);

    glfwDestroyWindow(g_window);
    glfwTerminate();

    return 0;
}