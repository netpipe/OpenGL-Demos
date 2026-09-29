// bvh_player.cpp
//
// Single-file OpenGL 4.1 BVH skeleton player.
// Uses GLFW + GLEW. No GLM.
//
// Controls:
//   Space       play/pause
//   Left/Right  step frame
//   Up/Down     speed
//   + / -       scale
//   C           cycle axis correction (useful for Blender/Z-up BVH)
//   F           fit camera to skeleton
//   T           toggle follow root
//   G           toggle grid
//   R           reset
//   H           help
//
// Example Linux build:
//   g++ -std=c++17 bvh_player.cpp -o bvh_player -lGLEW -lglfw -lGL
//
// Example macOS Homebrew build:
//   g++ -std=c++17 bvh_player.cpp -o bvh_player -I/opt/homebrew/include -L/opt/homebrew/lib -lGLEW -lglfw -framework OpenGL
//
// Example Windows MSYS2/MinGW:
//   g++ -std=c++17 bvh_player.cpp -o bvh_player.exe -lglew32 -lglfw3 -lopengl32 -lgdi32

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <cfloat>
#include <iterator>
#include <iomanip>

using namespace std;

static const float PI = 3.14159265358979323846f;

static float deg2rad(float d)
{
    return d * PI / 180.0f;
}

static string toUpper(string s)
{
    for (char& c : s)
        c = (char)toupper((unsigned char)c);
    return s;
}

// -----------------------------------------------------------------------------
// Minimal math (no GLM)
// -----------------------------------------------------------------------------

struct Vec3
{
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

    Vec3& operator+=(const Vec3& o)
    {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }
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
    return sqrtf(dot(v, v));
}

static Vec3 normalize(Vec3 v)
{
    float l = length(v);
    if (l > 1e-8f)
        return v * (1.0f / l);
    return Vec3(0.0f, 0.0f, 0.0f);
}

struct Mat4
{
    float m[16];

    Mat4()
    {
        for (int i = 0; i < 16; ++i)
            m[i] = 0.0f;
    }

    static Mat4 identity();
};

Mat4 Mat4::identity()
{
    Mat4 r;
    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;
    return r;
}

// Column-major multiplication: result = a * b
static Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int c = 0; c < 4; ++c)
    {
        for (int row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = sum;
        }
    }
    return r;
}

static Vec3 transformPoint(const Mat4& mat, const Vec3& v)
{
    return Vec3(
        mat.m[0] * v.x + mat.m[4] * v.y + mat.m[8]  * v.z + mat.m[12],
        mat.m[1] * v.x + mat.m[5] * v.y + mat.m[9]  * v.z + mat.m[13],
        mat.m[2] * v.x + mat.m[6] * v.y + mat.m[10] * v.z + mat.m[14]
    );
}

static Vec3 translationOf(const Mat4& mat)
{
    return Vec3(mat.m[12], mat.m[13], mat.m[14]);
}

static Mat4 translateMat(const Vec3& v)
{
    Mat4 r = Mat4::identity();
    r.m[12] = v.x;
    r.m[13] = v.y;
    r.m[14] = v.z;
    return r;
}

static Mat4 scaleMat(float s)
{
    Mat4 r = Mat4::identity();
    r.m[0] = s;
    r.m[5] = s;
    r.m[10] = s;
    return r;
}

static Mat4 rotateXDeg(float deg)
{
    float rad = deg2rad(deg);
    float c = cosf(rad);
    float s = sinf(rad);

    Mat4 r = Mat4::identity();
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

static Mat4 rotateYDeg(float deg)
{
    float rad = deg2rad(deg);
    float c = cosf(rad);
    float s = sinf(rad);

    Mat4 r = Mat4::identity();
    r.m[0] = c;
    r.m[2] = -s;
    r.m[8] = s;
    r.m[10] = c;
    return r;
}

static Mat4 rotateZDeg(float deg)
{
    float rad = deg2rad(deg);
    float c = cosf(rad);
    float s = sinf(rad);

    Mat4 r = Mat4::identity();
    r.m[0] = c;
    r.m[1] = s;
    r.m[4] = -s;
    r.m[5] = c;
    return r;
}

static Mat4 perspectiveMat(float fovDeg, float aspect, float zNear, float zFar)
{
    Mat4 r;
    float f = 1.0f / tanf(deg2rad(fovDeg) * 0.5f);

    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zFar + zNear) / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    return r;
}

static Mat4 lookAtMat(const Vec3& eye, const Vec3& center, const Vec3& up)
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

// -----------------------------------------------------------------------------
// BVH parsing
// -----------------------------------------------------------------------------

enum class Channel
{
    XPosition,
    YPosition,
    ZPosition,
    XRotation,
    YRotation,
    ZRotation
};

static Channel parseChannel(const string& s)
{
    string u = toUpper(s);

    if (u == "XPOSITION") return Channel::XPosition;
    if (u == "YPOSITION") return Channel::YPosition;
    if (u == "ZPOSITION") return Channel::ZPosition;
    if (u == "XROTATION") return Channel::XRotation;
    if (u == "YROTATION") return Channel::YRotation;
    if (u == "ZROTATION") return Channel::ZRotation;

    throw runtime_error("Unknown BVH channel: " + s);
}

static bool isRotationChannel(Channel c)
{
    return c == Channel::XRotation ||
           c == Channel::YRotation ||
           c == Channel::ZRotation;
}

struct Joint
{
    string name;
    int parent = -1;

    Vec3 offset;
    vector<Channel> channels;
    int channelStart = 0;

    Vec3 endOffset;
    bool hasEnd = false;

    vector<int> children;
};

struct BVH
{
    vector<Joint> joints;
    int root = -1;

    int frameCount = 0;
    float frameTime = 1.0f / 30.0f;

    int channelsPerFrame = 0;
    vector<float> motion;
};

struct TokenReader
{
    vector<string> tokens;
    size_t pos = 0;

    TokenReader(istream& in)
    {
        string content((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());

        // Make braces separate tokens.
        string processed;
        processed.reserve(content.size() + 64);

        for (char c : content)
        {
            if (c == '{' || c == '}')
            {
                processed.push_back(' ');
                processed.push_back(c);
                processed.push_back(' ');
            }
            else if (c == '\r' || c == '\n' || c == '\t')
            {
                processed.push_back(' ');
            }
            else
            {
                processed.push_back(c);
            }
        }

        stringstream ss(processed);
        string tok;
        while (ss >> tok)
            tokens.push_back(tok);
    }

    bool next(string& out)
    {
        if (pos >= tokens.size())
            return false;
        out = tokens[pos++];
        return true;
    }

    string nextRequired()
    {
        string s;
        if (!next(s))
            throw runtime_error("Unexpected end of BVH file");
        return s;
    }

    bool expect(const string& expected)
    {
        string t;
        if (!next(t))
            return false;
        return t == expected;
    }

    bool nextNumberOpt(float& out)
    {
        string tok;
        while (next(tok))
        {
            char* endPtr = nullptr;
            float v = strtof(tok.c_str(), &endPtr);

            // Accept anything that starts like a number.
            if (endPtr != tok.c_str())
            {
                out = v;
                return true;
            }
        }
        return false;
    }

    float nextNumber()
    {
        float v = 0.0f;
        if (nextNumberOpt(v))
            return v;
        throw runtime_error("Expected numeric value in BVH file");
    }
};

static int addJoint(BVH& bvh, int parent, const string& name)
{
    int idx = (int)bvh.joints.size();
    bvh.joints.push_back(Joint());
    bvh.joints[idx].name = name;
    bvh.joints[idx].parent = parent;
    return idx;
}

static void parseEndSite(TokenReader& tr, BVH& bvh, int jointIndex)
{
    string site = tr.nextRequired();
    if (toUpper(site) != "SITE")
        throw runtime_error("Expected SITE after END");

    if (!tr.expect("{"))
        throw runtime_error("Expected { in End Site");

    string tok;
    while (tr.next(tok))
    {
        if (tok == "}")
            return;

        if (toUpper(tok) == "OFFSET")
        {
            float x = tr.nextNumber();
            float y = tr.nextNumber();
            float z = tr.nextNumber();

            bvh.joints[jointIndex].endOffset = Vec3(x, y, z);
            bvh.joints[jointIndex].hasEnd = true;
        }
    }

    throw runtime_error("Unterminated End Site");
}

static void parseJointContents(TokenReader& tr, BVH& bvh, int idx, int& nextChannel)
{
    string tok;

    while (tr.next(tok))
    {
        if (tok == "}")
            return;

        string u = toUpper(tok);

        if (u == "OFFSET")
        {
            float x = tr.nextNumber();
            float y = tr.nextNumber();
            float z = tr.nextNumber();
            bvh.joints[idx].offset = Vec3(x, y, z);
        }
        else if (u == "CHANNELS")
        {
            int n = (int)tr.nextNumber();
            if (n < 0)
                n = 0;

            bvh.joints[idx].channelStart = nextChannel;

            for (int i = 0; i < n; ++i)
            {
                string channelName = tr.nextRequired();
                bvh.joints[idx].channels.push_back(parseChannel(channelName));
                ++nextChannel;
            }
        }
        else if (u == "JOINT")
        {
            string name = tr.nextRequired();

            if (!tr.expect("{"))
                throw runtime_error("Expected { after JOINT");

            int child = addJoint(bvh, idx, name);
            bvh.joints[idx].children.push_back(child);

            parseJointContents(tr, bvh, child, nextChannel);
        }
        else if (u == "END")
        {
            parseEndSite(tr, bvh, idx);
        }
        else if (u == "SITE")
        {
            // Tolerate a bare SITE block, although normal BVH uses "End Site".
            if (!tr.expect("{"))
                throw runtime_error("Expected { in SITE");

            string t;
            bool closed = false;
            while (tr.next(t))
            {
                if (t == "}")
                {
                    closed = true;
                    break;
                }

                if (toUpper(t) == "OFFSET")
                {
                    float x = tr.nextNumber();
                    float y = tr.nextNumber();
                    float z = tr.nextNumber();

                    bvh.joints[idx].endOffset = Vec3(x, y, z);
                    bvh.joints[idx].hasEnd = true;
                }
            }

            if (!closed)
                throw runtime_error("Unterminated SITE");
        }
    }

    throw runtime_error("Unterminated joint block");
}

static BVH parseBVH(istream& in)
{
    TokenReader tr(in);

    string tok = tr.nextRequired();
    if (toUpper(tok).find("HIERARCHY") == string::npos)
        throw runtime_error("Expected HIERARCHY");

    tok = tr.nextRequired();
    if (toUpper(tok).find("ROOT") == string::npos)
        throw runtime_error("Expected ROOT");

    string rootName = tr.nextRequired();

    if (!tr.expect("{"))
        throw runtime_error("Expected { after ROOT");

    BVH bvh;
    int root = addJoint(bvh, -1, rootName);
    bvh.root = root;

    int nextChannel = 0;
    parseJointContents(tr, bvh, root, nextChannel);
    bvh.channelsPerFrame = nextChannel;

    bool foundMotion = false;
    while (tr.next(tok))
    {
        if (toUpper(tok).find("MOTION") != string::npos)
        {
            foundMotion = true;
            break;
        }
    }

    if (!foundMotion)
        throw runtime_error("MOTION section not found");

    // nextNumber() will skip text like "Frames:" and "Frame Time:".
    int frames = (int)tr.nextNumber();
    float frameTime = tr.nextNumber();

    if (frames < 0)
        frames = 0;
    if (!(frameTime > 0.0f))
        frameTime = 1.0f / 30.0f;

    bvh.frameCount = frames;
    bvh.frameTime = frameTime;

    size_t needed = (size_t)frames * (size_t)nextChannel;
    bvh.motion.reserve(needed);

    for (size_t i = 0; i < needed; ++i)
    {
        float v = 0.0f;
        tr.nextNumberOpt(v);
        bvh.motion.push_back(v);
    }

    return bvh;
}

static bool loadBVHFile(const string& path, BVH& out)
{
    ifstream f(path, ios::binary);
    if (!f.is_open())
        return false;

    try
    {
        out = parseBVH(f);
        return true;
    }
    catch (const exception& e)
    {
        cerr << "BVH parse error: " << e.what() << "\n";
        return false;
    }
}

static bool loadBVHString(const string& text, BVH& out)
{
    stringstream ss(text);
    try
    {
        out = parseBVH(ss);
        return true;
    }
    catch (const exception& e)
    {
        cerr << "BVH string parse error: " << e.what() << "\n";
        return false;
    }
}

// -----------------------------------------------------------------------------
// BVH evaluation
// -----------------------------------------------------------------------------

static float sampleChannel(const BVH& bvh, int frame, int channelIndex)
{
    if (bvh.motion.empty() || bvh.frameCount <= 0 || bvh.channelsPerFrame <= 0)
        return 0.0f;

    if (channelIndex < 0 || channelIndex >= bvh.channelsPerFrame)
        return 0.0f;

    if (frame < 0)
        frame = 0;
    if (frame >= bvh.frameCount)
        frame = bvh.frameCount - 1;

    size_t idx = (size_t)frame * (size_t)bvh.channelsPerFrame + (size_t)channelIndex;
    if (idx >= bvh.motion.size())
        return 0.0f;

    return bvh.motion[idx];
}

static float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

static float lerpAngleDeg(float a, float b, float t)
{
    float d = fmodf(b - a, 360.0f);
    if (d > 180.0f)
        d -= 360.0f;
    if (d < -180.0f)
        d += 360.0f;
    return a + d * t;
}

static void evaluateBVH(const BVH& bvh, float frame, const Mat4& base, vector<Mat4>& world)
{
    world.assign(bvh.joints.size(), Mat4::identity());

    if (bvh.joints.empty())
        return;

    int f0 = 0;
    int f1 = 0;
    float alpha = 0.0f;

    if (bvh.frameCount > 0)
    {
        f0 = (int)floorf(frame);
        f1 = f0 + 1;

        if (f0 < 0)
            f0 = 0;
        if (f0 >= bvh.frameCount)
            f0 = bvh.frameCount - 1;

        if (f1 >= bvh.frameCount)
            f1 = bvh.frameCount - 1;

        alpha = frame - floorf(frame);
        if (alpha < 0.0f)
            alpha = 0.0f;
        if (alpha > 1.0f)
            alpha = 1.0f;
    }

    for (int i = 0; i < (int)bvh.joints.size(); ++i)
    {
        const Joint& j = bvh.joints[i];

        Vec3 trans = j.offset;
        Mat4 rot = Mat4::identity();

        for (int k = 0; k < (int)j.channels.size(); ++k)
        {
            int channelIndex = j.channelStart + k;

            float a = sampleChannel(bvh, f0, channelIndex);
            float b = sampleChannel(bvh, f1, channelIndex);

            float val;
            if (isRotationChannel(j.channels[k]))
                val = lerpAngleDeg(a, b, alpha);
            else
                val = lerp(a, b, alpha);

            switch (j.channels[k])
            {
                case Channel::XPosition: trans.x += val; break;
                case Channel::YPosition: trans.y += val; break;
                case Channel::ZPosition: trans.z += val; break;

                case Channel::XRotation: rot = rot * rotateXDeg(val); break;
                case Channel::YRotation: rot = rot * rotateYDeg(val); break;
                case Channel::ZRotation: rot = rot * rotateZDeg(val); break;
            }
        }

        Mat4 local = translateMat(trans) * rot;

        if (j.parent < 0)
        {
            world[i] = base * local;
        }
        else if (j.parent < (int)world.size())
        {
            world[i] = world[j.parent] * local;
        }
        else
        {
            world[i] = base * local;
        }
    }
}

// -----------------------------------------------------------------------------
// Drawing helpers
// -----------------------------------------------------------------------------

static void addVertex(vector<float>& verts, const Vec3& p, const Vec3& c)
{
    verts.push_back(p.x);
    verts.push_back(p.y);
    verts.push_back(p.z);
    verts.push_back(c.x);
    verts.push_back(c.y);
    verts.push_back(c.z);
}

static void buildSkeletonVertices(
    const BVH& bvh,
    const vector<Mat4>& world,
    vector<float>& combined,
    int& lineVertCount,
    int& pointVertCount
)
{
    combined.clear();

    const Vec3 boneColor(0.85f, 0.85f, 0.92f);
    const Vec3 endColor(0.35f, 0.80f, 1.00f);
    const Vec3 jointColor(1.00f, 0.65f, 0.10f);

    int n = (int)bvh.joints.size();

    // Bone lines.
    for (int i = 0; i < n; ++i)
    {
        const Joint& j = bvh.joints[i];

        if (j.parent >= 0 && j.parent < n)
        {
            Vec3 a = translationOf(world[j.parent]);
            Vec3 b = translationOf(world[i]);

            addVertex(combined, a, boneColor);
            addVertex(combined, b, boneColor);
        }
    }

    // End-site lines.
    for (int i = 0; i < n; ++i)
    {
        const Joint& j = bvh.joints[i];

        if (!j.hasEnd)
            continue;

        Vec3 a = translationOf(world[i]);
        Vec3 b = transformPoint(world[i], j.endOffset);

        addVertex(combined, a, endColor);
        addVertex(combined, b, endColor);
    }

    lineVertCount = (int)(combined.size() / 6);

    // Joint points.
    for (int i = 0; i < n; ++i)
    {
        Vec3 p = translationOf(world[i]);
        addVertex(combined, p, jointColor);
    }

    pointVertCount = n;
}

static void buildGrid(vector<float>& verts, float extent = 100.0f, int divisions = 20)
{
    verts.clear();

    if (divisions < 1)
        divisions = 1;

    const Vec3 gridColor(0.22f, 0.22f, 0.24f);
    const Vec3 xAxisColor(0.75f, 0.20f, 0.20f);
    const Vec3 zAxisColor(0.20f, 0.30f, 0.80f);
    const Vec3 yAxisColor(0.20f, 0.75f, 0.20f);

    float half = extent * 0.5f;
    float step = extent / (float)divisions;

    for (int i = 0; i <= divisions; ++i)
    {
        float p = -half + step * (float)i;

        addVertex(verts, Vec3(p, 0.0f, -half), gridColor);
        addVertex(verts, Vec3(p, 0.0f,  half), gridColor);

        addVertex(verts, Vec3(-half, 0.0f, p), gridColor);
        addVertex(verts, Vec3( half, 0.0f, p), gridColor);
    }

    // Axis helpers.
    addVertex(verts, Vec3(-half, 0.0f, 0.0f), xAxisColor);
    addVertex(verts, Vec3( half, 0.0f, 0.0f), xAxisColor);

    addVertex(verts, Vec3(0.0f, 0.0f, -half), zAxisColor);
    addVertex(verts, Vec3(0.0f, 0.0f,  half), zAxisColor);

    addVertex(verts, Vec3(0.0f, 0.0f, 0.0f), yAxisColor);
    addVertex(verts, Vec3(0.0f, half, 0.0f), yAxisColor);
}

struct DynamicBuffer
{
    GLuint vao = 0;
    GLuint vbo = 0;

    void init()
    {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void upload(const vector<float>& verts)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        GLsizeiptr sz = (GLsizeiptr)(verts.size() * sizeof(float));
        glBufferData(GL_ARRAY_BUFFER, sz, verts.empty() ? nullptr : verts.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void destroy()
    {
        if (vbo)
        {
            glDeleteBuffers(1, &vbo);
            vbo = 0;
        }
        if (vao)
        {
            glDeleteVertexArrays(1, &vao);
            vao = 0;
        }
    }
};

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);

    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        glDeleteShader(shader);
        throw runtime_error(string("Shader compile error: ") + log);
    }

    return shader;
}

static GLuint createProgram(const char* vsSrc, const char* fsSrc)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);

    if (!ok)
    {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        glDeleteProgram(program);
        throw runtime_error(string("Program link error: ") + log);
    }

    return program;
}

static const char* vertexShaderSource =
"#version 410 core\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aColor;\n"
"uniform mat4 uMVP;\n"
"out vec3 vColor;\n"
"void main()\n"
"{\n"
"    gl_Position = uMVP * vec4(aPos, 1.0);\n"
"    gl_PointSize = 7.0;\n"
"    vColor = aColor;\n"
"}\n";

static const char* fragmentShaderSource =
"#version 410 core\n"
"in vec3 vColor;\n"
"out vec4 fragColor;\n"
"void main()\n"
"{\n"
"    fragColor = vec4(vColor, 1.0);\n"
"}\n";

// -----------------------------------------------------------------------------
// App state / camera / input
// -----------------------------------------------------------------------------

struct App
{
    BVH bvh;
    bool loaded = false;

    float currentFrame = 0.0f;
    bool playing = true;
    float speed = 1.0f;
    float scale = 1.0f;

    bool showGrid = true;
    bool follow = false;

    // Axis correction:
    // 0 = none
    // 1 = rotate X -90, often useful for Z-up BVH
    // 2 = rotate X +90
    // 3 = rotate Y 180
    int axisMode = 0;

    float yaw = 45.0f;
    float pitch = 22.0f;
    float dist = 30.0f;
    Vec3 target = Vec3(0.0f, 10.0f, 0.0f);

    bool dragging = false;
    double lastX = 0.0;
    double lastY = 0.0;

    Vec3 homeTarget = Vec3(0.0f, 10.0f, 0.0f);
    float homeYaw = 45.0f;
    float homePitch = 22.0f;
    float homeDist = 30.0f;
    float homeScale = 1.0f;
};

static Mat4 axisCorrectionMat(int mode)
{
    switch (mode)
    {
        case 1: return rotateXDeg(-90.0f);
        case 2: return rotateXDeg( 90.0f);
        case 3: return rotateYDeg(180.0f);
        default: return Mat4::identity();
    }
}

static Mat4 globalBaseMat(const App& app)
{
    return axisCorrectionMat(app.axisMode) * scaleMat(app.scale);
}

static void fitCamera(App& app, bool setHome)
{
    if (!app.loaded || app.bvh.joints.empty())
        return;

    vector<Mat4> world;
    evaluateBVH(app.bvh, 0.0f, globalBaseMat(app), world);

    Vec3 mn(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    bool any = false;

    auto includePoint = [&](const Vec3& p)
    {
        any = true;
        mn.x = min(mn.x, p.x);
        mn.y = min(mn.y, p.y);
        mn.z = min(mn.z, p.z);

        mx.x = max(mx.x, p.x);
        mx.y = max(mx.y, p.y);
        mx.z = max(mx.z, p.z);
    };

    for (int i = 0; i < (int)world.size(); ++i)
    {
        includePoint(translationOf(world[i]));

        const Joint& j = app.bvh.joints[i];
        if (j.hasEnd)
            includePoint(transformPoint(world[i], j.endOffset));
    }

    if (!any)
        return;

    Vec3 center = (mn + mx) * 0.5f;
    float radius = length(mx - mn) * 0.5f;
    if (radius < 1e-3f)
        radius = 1.0f;

    app.target = center;
    app.dist = radius * 2.4f + 1.0f;
    app.yaw = 45.0f;
    app.pitch = 22.0f;

    if (setHome)
    {
        app.homeTarget = app.target;
        app.homeYaw = app.yaw;
        app.homePitch = app.pitch;
        app.homeDist = app.dist;
        app.homeScale = app.scale;
    }
}

static Vec3 cameraEye(const App& app)
{
    float cp = cosf(deg2rad(app.pitch));
    float sp = sinf(deg2rad(app.pitch));
    float cy = cosf(deg2rad(app.yaw));
    float sy = sinf(deg2rad(app.yaw));

    return app.target + Vec3(cy * cp * app.dist, sp * app.dist, sy * cp * app.dist);
}

static void printHelp()
{
    cout <<
        "BVH player controls:\n"
        "  Space        play/pause\n"
        "  Left/Right   step frame\n"
        "  Up/Down      speed\n"
        "  + / -        scale\n"
        "  C            cycle axis correction\n"
        "  F            fit camera to skeleton\n"
        "  T            toggle follow root\n"
        "  G            toggle grid\n"
        "  R            reset\n"
        "  H            help\n"
        "  Esc          quit\n"
        "\n"
        "If a Blender-exported BVH appears lying down, press C until it looks right.\n"
        "BVH is skeleton animation only. For skinned meshes, export glTF from Blender.\n";
}

static void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!app || action == GLFW_RELEASE)
        return;

    bool press = (action == GLFW_PRESS);

    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            if (press)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;

        case GLFW_KEY_SPACE:
            if (press)
                app->playing = !app->playing;
            break;

        case GLFW_KEY_RIGHT:
            app->playing = false;
            app->currentFrame += 1.0f;
            break;

        case GLFW_KEY_LEFT:
            app->playing = false;
            app->currentFrame -= 1.0f;
            break;

        case GLFW_KEY_UP:
            app->speed *= 1.1f;
            break;

        case GLFW_KEY_DOWN:
            app->speed /= 1.1f;
            if (app->speed < 0.01f)
                app->speed = 0.01f;
            break;

        case GLFW_KEY_EQUAL:
        case GLFW_KEY_KP_ADD:
            app->scale *= 1.1f;
            break;

        case GLFW_KEY_MINUS:
        case GLFW_KEY_KP_SUBTRACT:
            app->scale /= 1.1f;
            if (app->scale < 0.001f)
                app->scale = 0.001f;
            break;

        case GLFW_KEY_R:
            if (press)
            {
                app->currentFrame = 0.0f;
                app->speed = 1.0f;
                app->scale = app->homeScale;
                app->axisMode = 0;
                app->follow = false;
                app->target = app->homeTarget;
                app->yaw = app->homeYaw;
                app->pitch = app->homePitch;
                app->dist = app->homeDist;
                app->playing = true;
            }
            break;

        case GLFW_KEY_G:
            if (press)
                app->showGrid = !app->showGrid;
            break;

        case GLFW_KEY_C:
            if (press)
            {
                app->axisMode = (app->axisMode + 1) % 4;
                fitCamera(*app, false);
            }
            break;

        case GLFW_KEY_F:
            if (press)
                fitCamera(*app, false);
            break;

        case GLFW_KEY_T:
            if (press)
                app->follow = !app->follow;
            break;

        case GLFW_KEY_H:
            if (press)
                printHelp();
            break;

        default:
            break;
    }
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!app)
        return;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            app->dragging = true;
            glfwGetCursorPos(window, &app->lastX, &app->lastY);
        }
        else if (action == GLFW_RELEASE)
        {
            app->dragging = false;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double x, double y)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!app || !app->dragging)
        return;

    double dx = x - app->lastX;
    double dy = y - app->lastY;

    app->yaw += (float)dx * 0.30f;
    app->pitch += (float)dy * 0.30f;

    if (app->pitch > 89.0f)
        app->pitch = 89.0f;
    if (app->pitch < -89.0f)
        app->pitch = -89.0f;

    app->lastX = x;
    app->lastY = y;
}

static void scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!app)
        return;

    float factor = 1.0f - (float)yoffset * 0.10f;
    if (factor < 0.10f)
        factor = 0.10f;

    app->dist *= factor;

    if (app->dist < 0.1f)
        app->dist = 0.1f;
}

// -----------------------------------------------------------------------------
// Sample BVH, used if no file is supplied
// -----------------------------------------------------------------------------

static string makeSampleBVHString()
{
    int frames = 120;
    float frameTime = 1.0f / 30.0f;

    ostringstream ss;
    ss << fixed << setprecision(5);

    ss << "HIERARCHY\n";
    ss << "ROOT Hips\n{\n";
    ss << " OFFSET 0.0 0.0 0.0\n";
    ss << " CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n";

    ss << " JOINT Spine\n {\n";
    ss << "  OFFSET 0.0 5.0 0.0\n";
    ss << "  CHANNELS 3 Xrotation Yrotation Zrotation\n";

    ss << "  JOINT Head\n  {\n";
    ss << "   OFFSET 0.0 5.0 0.0\n";
    ss << "   CHANNELS 3 Xrotation Yrotation Zrotation\n";
    ss << "   End Site\n   {\n";
    ss << "    OFFSET 0.0 3.0 0.0\n";
    ss << "   }\n";
    ss << "  }\n";
    ss << " }\n";

    ss << " JOINT LeftArm\n {\n";
    ss << "  OFFSET 4.0 4.0 0.0\n";
    ss << "  CHANNELS 3 Xrotation Yrotation Zrotation\n";
    ss << "  End Site\n  {\n";
    ss << "   OFFSET 4.0 0.0 0.0\n";
    ss << "  }\n";
    ss << " }\n";

    ss << " JOINT RightArm\n {\n";
    ss << "  OFFSET -4.0 4.0 0.0\n";
    ss << "  CHANNELS 3 Xrotation Yrotation Zrotation\n";
    ss << "  End Site\n  {\n";
    ss << "   OFFSET -4.0 0.0 0.0\n";
    ss << "  }\n";
    ss << " }\n";

    ss << "}\n";
    ss << "MOTION\n";
    ss << "Frames: " << frames << "\n";
    ss << "Frame Time: " << frameTime << "\n";

    for (int i = 0; i < frames; ++i)
    {
        float t = (float)i / (float)(frames - 1) * 2.0f * PI;

        float rootX = sinf(t) * 4.0f;
        float rootY = 20.0f + sinf(t * 2.0f);
        float rootZ = 0.0f;

        float rootXR = 0.0f;
        float rootYR = 0.0f;
        float rootZR = sinf(t) * 5.0f;

        float spineZ = sinf(t) * 15.0f;
        float headZ = -sinf(t) * 25.0f;

        float leftZ = -35.0f + sinf(t) * 25.0f;
        float rightZ =  35.0f - sinf(t) * 25.0f;

        ss << rootX << ' ' << rootY << ' ' << rootZ << ' '
           << rootXR << ' ' << rootYR << ' ' << rootZR << ' '
           << 0.0f << ' ' << 0.0f << ' ' << spineZ << ' '
           << 0.0f << ' ' << 0.0f << ' ' << headZ << ' '
           << 0.0f << ' ' << 0.0f << ' ' << leftZ << ' '
           << 0.0f << ' ' << 0.0f << ' ' << rightZ << '\n';
    }

    return ss.str();
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (!glfwInit())
    {
        cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "BVH Player", nullptr, nullptr);
    if (!window)
    {
        cerr << "Failed to create OpenGL 4.1 core context\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK)
    {
        cerr << "GLEW init failed: " << glewGetErrorString(glewErr) << "\n";
        glfwTerminate();
        return -1;
    }

    // Some drivers emit a harmless error after glewInit on core profiles.
    while (glGetError() != GL_NO_ERROR) {}

    App app;

    bool ok = false;
    if (argc > 1)
    {
        ok = loadBVHFile(argv[1], app.bvh);
        if (!ok)
        {
            cerr << "Could not load '" << argv[1] << "', falling back to sample BVH.\n";
            ok = loadBVHString(makeSampleBVHString(), app.bvh);
        }
    }
    else
    {
        cout << "No BVH file supplied. Running built-in sample.\n";
        cout << "Usage: " << argv[0] << " file.bvh\n\n";
        ok = loadBVHString(makeSampleBVHString(), app.bvh);
    }

    app.loaded = ok;

    if (!ok)
        cerr << "No BVH loaded.\n";

    app.currentFrame = 0.0f;
    app.playing = true;
    app.scale = 1.0f;

    fitCamera(app, true);
    printHelp();

    GLuint program = 0;
    try
    {
        program = createProgram(vertexShaderSource, fragmentShaderSource);
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        glfwTerminate();
        return -1;
    }

    GLint mvpLoc = glGetUniformLocation(program, "uMVP");

    DynamicBuffer gridBuf;
    DynamicBuffer skelBuf;
    gridBuf.init();
    skelBuf.init();

    vector<float> gridVerts;
    float gridExtent = max(100.0f, app.dist * 2.0f);
    buildGrid(gridVerts, gridExtent, 20);
    gridBuf.upload(gridVerts);
    int gridVertCount = (int)(gridVerts.size() / 6);

    glfwSetWindowUserPointer(window, &app);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
    glLineWidth(2.0f);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        double now = glfwGetTime();
        double dt = now - lastTime;
        lastTime = now;

        if (app.playing && app.loaded && app.bvh.frameCount > 0 && app.bvh.frameTime > 1e-6f)
        {
            app.currentFrame += (float)(dt / (double)app.bvh.frameTime) * app.speed;
        }

        if (app.loaded && app.bvh.frameCount > 0)
        {
            float count = (float)app.bvh.frameCount;
            app.currentFrame = fmodf(app.currentFrame, count);
            if (app.currentFrame < 0.0f)
                app.currentFrame += count;
        }

        // Evaluate skeleton before camera so follow-root can update target.
        vector<Mat4> world;
        bool haveSkeleton = false;

        if (app.loaded && !app.bvh.joints.empty())
        {
            evaluateBVH(app.bvh, app.currentFrame, globalBaseMat(app), world);
            haveSkeleton = true;

            if (app.follow && !world.empty())
            {
                int rootIdx = app.bvh.root;
                if (rootIdx < 0 || rootIdx >= (int)world.size())
                    rootIdx = 0;

                app.target = translationOf(world[rootIdx]);
            }
        }

        int width = 1;
        int height = 1;
        glfwGetFramebufferSize(window, &width, &height);
        if (width < 1) width = 1;
        if (height < 1) height = 1;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (float)width / (float)height;

        float zNear = max(0.01f, app.dist * 0.001f);
        float zFar = max(zNear + 10.0f, app.dist * 100.0f + 1000.0f);

        Mat4 proj = perspectiveMat(45.0f, aspect, zNear, zFar);
        Vec3 eye = cameraEye(app);
        Mat4 view = lookAtMat(eye, app.target, Vec3(0.0f, 1.0f, 0.0f));
        Mat4 vp = proj * view;

        glUseProgram(program);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, vp.m);

        if (app.showGrid)
        {
            glBindVertexArray(gridBuf.vao);
            glDrawArrays(GL_LINES, 0, gridVertCount);
            glBindVertexArray(0);
        }

        if (haveSkeleton)
        {
            vector<float> skelVerts;
            int lineVerts = 0;
            int pointVerts = 0;

            buildSkeletonVertices(app.bvh, world, skelVerts, lineVerts, pointVerts);
            skelBuf.upload(skelVerts);

            glBindVertexArray(skelBuf.vao);

            if (lineVerts > 0)
                glDrawArrays(GL_LINES, 0, lineVerts);

            if (pointVerts > 0)
                glDrawArrays(GL_POINTS, lineVerts, pointVerts);

            glBindVertexArray(0);
        }

        ostringstream title;
        title << "BVH Player - frame "
              << (int)app.currentFrame << "/" << app.bvh.frameCount
              << " | speed " << app.speed
              << " | scale " << app.scale
              << " | axis " << app.axisMode
              << " | H help";
        glfwSetWindowTitle(window, title.str().c_str());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    gridBuf.destroy();
    skelBuf.destroy();
    glDeleteProgram(program);

    glfwTerminate();
    return 0;
}