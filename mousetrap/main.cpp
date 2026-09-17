// ============================================================================
//  MOUSE TRAP (1981) - 2D/3D recreation
//  Build:  see bottom of file. Needs GLFW + GLEW. No GLM, no GLU.
//
//  CONTROLS
//    Arrows / WASD ... move the mouse
//    STOP ............ camera rises -> 2D top-down map view
//    MOVE ............ camera dives -> 3D mouse-eye view
//    TAB (hold) ...... look behind you in 3D
//    1 / 2 / 3 ....... slam open/close the cyan / magenta / yellow doors
//                      (a door closing on a cat traps it!)
//    SPACE ........... use a stored bone -> become a DOG (cats fear you)
//    P pause, ENTER start, ESC back to title
// ============================================================================
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <vector>
#include <string>

#define PI 3.14159265359f
static int lastDX=0, lastDZ=0;   // last direction key pressed (single axis)
// ---------------------------------------------------------------- math -----
struct V3 { float x,y,z; };
static V3 v3(float x,float y,float z){ V3 r={x,y,z}; return r; }
static V3 operator+(V3 a,V3 b){ return v3(a.x+b.x,a.y+b.y,a.z+b.z); }
static V3 operator-(V3 a,V3 b){ return v3(a.x-b.x,a.y-b.y,a.z-b.z); }
static V3 operator*(V3 a,float k){ return v3(a.x*k,a.y*k,a.z*k); }
static float dot(V3 a,V3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static V3 cross(V3 a,V3 b){ return v3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
static float vlen(V3 a){ return sqrtf(dot(a,a)); }
static V3 vnorm(V3 a){ float l=vlen(a); if(l<1e-8f) return v3(0,0,0); return a*(1.f/l); }
static V3 vlerp(V3 a,V3 b,float t){ return a+((b-a)*t); }

struct Mat4 { float m[16]; };
static Mat4 matIdentity(){ Mat4 r; memset(r.m,0,sizeof(r.m)); r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
static Mat4 matMul(const Mat4&a,const Mat4&b){
    Mat4 r; for(int c=0;c<4;c++) for(int rw=0;rw<4;rw++){
        float s=0; for(int k=0;k<4;k++) s+=a.m[k*4+rw]*b.m[c*4+k];
        r.m[c*4+rw]=s;
    } return r;
}
static Mat4 matLerp(const Mat4&a,const Mat4&b,float t){
    Mat4 r; for(int i=0;i<16;i++) r.m[i]=a.m[i]+(b.m[i]-a.m[i])*t; return r;
}
static Mat4 matPerspective(float fovDeg,float aspect,float zn,float zf){
    Mat4 r; memset(r.m,0,sizeof(r.m));
    float f=1.f/tanf(fovDeg*PI/360.f);
    r.m[0]=f/aspect; r.m[5]=f; r.m[10]=(zf+zn)/(zn-zf); r.m[11]=-1; r.m[14]=2*zf*zn/(zn-zf);
    return r;
}
static Mat4 matOrtho(float l,float r_,float b,float t,float zn,float zf){
    Mat4 r; memset(r.m,0,sizeof(r.m));
    r.m[0]=2/(r_-l); r.m[5]=2/(t-b); r.m[10]=-2/(zf-zn);
    r.m[12]=-(r_+l)/(r_-l); r.m[13]=-(t+b)/(t-b); r.m[14]=-(zf+zn)/(zf-zn); r.m[15]=1;
    return r;
}
static Mat4 matLookAt(V3 e,V3 c,V3 up){
    V3 f=vnorm(c-e), s=vnorm(cross(f,up)), u=cross(s,f);
    Mat4 r=matIdentity();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8]=s.z;   r.m[12]=-dot(s,e);
    r.m[1]=u.x; r.m[5]=u.y; r.m[9]=u.z;   r.m[13]=-dot(u,e);
    r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z; r.m[14]= dot(f,e);
    return r;
}
static float smoothstep01(float t){ t=t<0?0:(t>1?1:t); return t*t*(3-2*t); }

struct RGBA { float r,g,b,a; };
static RGBA rgba(float r,float g,float b,float a=1){ RGBA c={r,g,b,a}; return c; }

// ---------------------------------------------------------------- mesh -----
struct Vert { float x,y,z,r,g,b,a; };
struct Mesh {
    std::vector<Vert> V;
    GLuint vao=0, vbo=0;
    void init(){
        glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vert),(void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(Vert),(void*)(3*sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
    }
    void reset(){ V.clear(); }
    void vtx(V3 p,RGBA c){ V.push_back({p.x,p.y,p.z,c.r,c.g,c.b,c.a}); }
    void tri(V3 a,V3 b,V3 c,RGBA col){ vtx(a,col); vtx(b,col); vtx(c,col); }
    void quad(V3 a,V3 b,V3 c,V3 d,RGBA col){ tri(a,b,c,col); tri(a,c,d,col); }
    void disc(V3 c,V3 R,V3 U,float rad,RGBA col,int seg=12){
        for(int i=0;i<seg;i++){
            float a0=2*PI*i/seg, a1=2*PI*(i+1)/seg;
            tri(c, c+R*cosf(a0)*rad+U*sinf(a0)*rad,
                   c+R*cosf(a1)*rad+U*sinf(a1)*rad, col);
        }
    }
    void draw(GLuint p,const Mat4&mvp,V3 cam,V3 fogC,float fs,float fe,float fa){
        if(V.empty()) return;
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(V.size()*sizeof(Vert)),
                     V.data(),GL_STREAM_DRAW);
        glUseProgram(p);
        glUniformMatrix4fv(glGetUniformLocation(p,"uMVP"),1,GL_FALSE,mvp.m);
        glUniform3f(glGetUniformLocation(p,"uCamPos"),cam.x,cam.y,cam.z);
        glUniform3f(glGetUniformLocation(p,"uFogColor"),fogC.x,fogC.y,fogC.z);
        glUniform1f(glGetUniformLocation(p,"uFogStart"),fs);
        glUniform1f(glGetUniformLocation(p,"uFogEnd"),fe);
        glUniform1f(glGetUniformLocation(p,"uFogAmt"),fa);
        glDrawArrays(GL_TRIANGLES,0,(GLsizei)V.size());
        glBindVertexArray(0);
    }
};
// ---------------------------------------------------------------- font -----
// tiny embedded 5x7 bitmap font (A-Z, 0-9 and a few symbols)
struct Glyph { char c; unsigned char d[7]; };
static const Glyph GTAB[] = {
 {'A',{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}},{'B',{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
 {'C',{0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}},{'D',{0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
 {'E',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}},{'F',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
 {'G',{0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}},{'H',{0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
 {'I',{0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}},{'J',{0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
 {'K',{0x11,0x12,0x14,0x18,0x14,0x12,0x11}},{'L',{0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
 {'M',{0x11,0x1B,0x15,0x15,0x11,0x11,0x11}},{'N',{0x11,0x19,0x15,0x13,0x11,0x11,0x11}},
 {'O',{0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}},{'P',{0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
 {'Q',{0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}},{'R',{0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
 {'S',{0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}},{'T',{0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
 {'U',{0x11,0x11,0x11,0x11,0x11,0x11,0x0E}},{'V',{0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
 {'W',{0x11,0x11,0x11,0x15,0x15,0x15,0x0A}},{'X',{0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
 {'Y',{0x11,0x11,0x0A,0x04,0x04,0x04,0x04}},{'Z',{0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
 {'0',{0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}},{'1',{0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
 {'2',{0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}},{'3',{0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}},
 {'4',{0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}},{'5',{0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
 {'6',{0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}},{'7',{0x1F,0x01,0x02,0x04,0x04,0x08,0x08}},
 {'8',{0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}},{'9',{0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
 {' ',{0,0,0,0,0,0,0}},
 {'!',{0x04,0x04,0x04,0x04,0x04,0x00,0x04}},{'.',{0,0,0,0,0,0x06,0x06}},
 {'-',{0,0,0,0x1F,0,0,0}},{':',{0,0x06,0x06,0,0x06,0x06,0}},
 {'/',{0x01,0x01,0x02,0x04,0x08,0x10,0x10}},{'?',{0x0E,0x11,0x01,0x06,0x04,0x00,0x04}},
 {'+',{0,0x04,0x04,0x1F,0x04,0x04,0}},{ '=',{0,0,0x1F,0,0x1F,0,0}},
 {'(',{0x02,0x04,0x08,0x08,0x08,0x04,0x02}},{')',{0x08,0x04,0x02,0x02,0x02,0x04,0x08}},
};
static const unsigned char* getGlyph(char c){
    if(c>='a'&&c<='z') c=c-'a'+'A';
    for(auto&g:GTAB) if(g.c==c) return g.d;
    return nullptr;
}
static float textW(const char*s,float sc){ return 6.f*sc*(float)strlen(s); }

static void drawText2D(Mesh&m,float x,float y,float sc,RGBA col,const char*s){
    for(;*s;s++){
        const unsigned char*g=getGlyph(*s);
        if(g) for(int r=0;r<7;r++) for(int c=0;c<5;c++)
            if(g[r]&(1<<(4-c)))
                m.quad(v3(x+c*sc,y+(6-r)*sc,0), v3(x+(c+1)*sc,y+(6-r)*sc,0),
                       v3(x+(c+1)*sc,y+(7-r)*sc,0), v3(x+c*sc,y+(7-r)*sc,0), col);
        x+=6*sc;
    }
}
static void drawText3D(Mesh&m,V3 cen,V3 R,V3 U,float sc,RGBA col,const char*s){
    float w=6.f*sc*(float)strlen(s);
    V3 p=cen-R*(w*0.5f)-U*(3.5f*sc);
    for(const char*ch=s;*ch;ch++){
        const unsigned char*g=getGlyph(*ch);
        if(g) for(int r=0;r<7;r++) for(int c=0;c<5;c++)
            if(g[r]&(1<<(4-c)))
                m.quad(p+R*(c*sc)+U*((6-r)*sc), p+R*((c+1)*sc)+U*((6-r)*sc),
                       p+R*((c+1)*sc)+U*((7-r)*sc), p+R*(c*sc)+U*((7-r)*sc), col);
        p=p+R*(6*sc);
    }
}

// ------------------------------------------------------- sprite drawing ----
// All sprites are vector-drawn billboards: basis vectors R/U let the same
// code draw world billboards (R/U = camera axes) and flat HUD icons.
static void bquad(Mesh&m,V3 c,V3 R,V3 U,float x0,float y0,float x1,float y1,RGBA col){
    m.quad(c+R*x0+U*y0, c+R*x1+U*y0, c+R*x1+U*y1, c+R*x0+U*y1, col);
}
static void btri(Mesh&m,V3 c,V3 R,V3 U,float x0,float y0,float x1,float y1,float x2,float y2,RGBA col){
    m.tri(c+R*x0+U*y0, c+R*x1+U*y1, c+R*x2+U*y2, col);
}
static void sprCheese(Mesh&m,V3 c,V3 R,V3 U,float s){
    bquad(m,c,R,U,-0.34f,-0.34f,0.34f,-0.24f, rgba(0.95f,0.62f,0.10f));
    btri (m,c,R,U,-0.34f,-0.24f,0.34f,-0.24f,0.08f,0.36f, rgba(1,0.84f,0.25f));
    bquad(m,c,R,U,-0.06f,-0.02f,0.05f,0.08f, rgba(0.80f,0.62f,0.15f));
    bquad(m,c,R,U, 0.10f,-0.16f,0.18f,-0.09f, rgba(0.80f,0.62f,0.15f));
}
static void sprMouse(Mesh&m,V3 c,V3 R,V3 U,float s,RGBA fur=rgba(0.66f,0.68f,0.75f)){
    bquad(m,c,R,U,-0.30f,-0.38f,0.30f,0.14f, fur);
    btri (m,c,R,U,-0.02f,-0.38f,0.30f,-0.55f,0.05f,-0.30f, rgba(0.85f,0.6f,0.7f)); // tail
    m.disc(c+R*(-0.27f*s)+U*(0.28f*s), R*(s),U*(s), 0.17f, fur);   // ears
    m.disc(c+R*( 0.27f*s)+U*(0.28f*s), R*(s),U*(s), 0.17f, fur);
    m.disc(c+R*(-0.27f*s)+U*(0.28f*s), R*(s),U*(s), 0.09f, rgba(1,0.62f,0.72f));
    m.disc(c+R*( 0.27f*s)+U*(0.28f*s), R*(s),U*(s), 0.09f, rgba(1,0.62f,0.72f));
    m.disc(c+R*(-0.12f*s)+U*(-0.02f*s), R*(s),U*(s), 0.05f, rgba(0.1f,0.1f,0.12f)); // eyes
    m.disc(c+R*( 0.12f*s)+U*(-0.02f*s), R*(s),U*(s), 0.05f, rgba(0.1f,0.1f,0.12f));
    m.disc(c+U*(-0.16f*s), R*(s),U*(s), 0.055f, rgba(1,0.55f,0.68f));               // nose
}
static void sprCat(Mesh&m,V3 c,V3 R,V3 U,float s,RGBA fur,float squash=1){
    float sy=s*squash;
    bquad(m,c,R,U,-0.34f*s,-0.42f*sy,0.34f*s,0.16f*sy, fur);
    btri (m,c,R,U,-0.34f*s,0.16f*sy,-0.14f*s,0.16f*sy,-0.30f*s,0.46f*sy, fur);
    btri (m,c,R,U, 0.34f*s,0.16f*sy, 0.14f*s,0.16f*sy, 0.30f*s,0.46f*sy, fur);
    btri (m,c,R,U,-0.30f*s,0.20f*sy,-0.19f*s,0.20f*sy,-0.28f*s,0.38f*sy, rgba(1,0.6f,0.7f));
    btri (m,c,R,U, 0.30f*s,0.20f*sy, 0.19f*s,0.20f*sy, 0.28f*s,0.38f*sy, rgba(1,0.6f,0.7f));
    btri (m,c,R,U, 0.30f*s,-0.34f*sy, 0.58f*s,-0.46f*sy, 0.42f*s,-0.04f*sy, fur); // tail
    m.disc(c+R*(-0.15f*s)+U*(0.02f*sy), R*(s),U*(sy), 0.075f, rgba(1,1,0.95f));
    m.disc(c+R*( 0.15f*s)+U*(0.02f*sy), R*(s),U*(sy), 0.075f, rgba(1,1,0.95f));
    m.disc(c+R*(-0.15f*s)+U*(0.02f*sy), R*(s),U*(sy), 0.038f, rgba(0.1f,0.1f,0.1f));
    m.disc(c+R*( 0.15f*s)+U*(0.02f*sy), R*(s),U*(sy), 0.038f, rgba(0.1f,0.1f,0.1f));
    btri (m,c,R,U,-0.04f*s,-0.10f*sy,0.04f*s,-0.10f*sy,0,-0.17f*sy, rgba(1,0.55f,0.65f));
}
static void sprDog(Mesh&m,V3 c,V3 R,V3 U,float s){
    RGBA fur=rgba(0.74f,0.52f,0.28f), dark=rgba(0.5f,0.33f,0.16f);
    bquad(m,c,R,U,-0.34f*s,-0.42f*s,0.34f*s,0.16f*s, fur);
    bquad(m,c,R,U,-0.40f*s,-0.14f*s,-0.14f*s,0.30f*s, dark);  // floppy ears
    bquad(m,c,R,U, 0.40f*s,-0.14f*s, 0.14f*s,0.30f*s, dark);
    m.disc(c+R*(-0.14f*s)+U*(0.02f*s), R*(s),U*(s), 0.07f, rgba(1,1,0.95f));
    m.disc(c+R*( 0.14f*s)+U*(0.02f*s), R*(s),U*(s), 0.07f, rgba(1,1,0.95f));
    m.disc(c+R*(-0.14f*s)+U*(0.02f*s), R*(s),U*(s), 0.035f, rgba(0.1f,0.1f,0.1f));
    m.disc(c+R*( 0.14f*s)+U*(0.02f*s), R*(s),U*(s), 0.035f, rgba(0.1f,0.1f,0.1f));
    m.disc(c+U*(-0.13f*s), R*(s),U*(s), 0.075f, rgba(0.1f,0.1f,0.1f));           // nose
    bquad(m,c,R,U,-0.26f*s,-0.30f*s,0.26f*s,-0.22f*s, rgba(0.85f,0.15f,0.15f)); // collar
}
static void sprHawk(Mesh&m,V3 c,V3 R,V3 U,float s,float flap){
    RGBA br=rgba(0.38f,0.24f,0.12f);
    btri (m,c,R,U,-0.06f*s,0.02f*s,-0.80f*s,(0.16f+flap)*s,-0.20f*s,-0.18f*s, br);
    btri (m,c,R,U, 0.06f*s,0.02f*s, 0.80f*s,(0.16f+flap)*s, 0.20f*s,-0.18f*s, br);
    btri (m,c,R,U,0,-0.08f*s,0.14f*s,-0.34f*s,-0.14f*s,-0.34f*s, br);
    m.disc(c+U*(0.06f*s), R*(s),U*(s), 0.14f, br);
    btri (m,c,R,U,-0.03f*s,0.18f*s,0.03f*s,0.18f*s,0,0.28f*s, rgba(0.95f,0.8f,0.2f));
}
static void sprBone(Mesh&m,V3 c,V3 R,V3 U,float s){
    RGBA w=rgba(0.95f,0.95f,0.9f);
    bquad(m,c,R,U,-0.24f*s,-0.06f*s,0.24f*s,0.06f*s, w);
    m.disc(c+R*(-0.26f*s)+U*(0.08f*s), R*(s),U*(s), 0.10f, w);
    m.disc(c+R*(-0.26f*s)-U*(0.08f*s), R*(s),U*(s), 0.10f, w);
    m.disc(c+R*( 0.26f*s)+U*(0.08f*s), R*(s),U*(s), 0.10f, w);
    m.disc(c+R*( 0.26f*s)-U*(0.08f*s), R*(s),U*(s), 0.10f, w);
}
static void sprBonus(Mesh&m,V3 c,V3 R,V3 U,float s){
    bquad(m,c,R,U,-0.36f*s,-0.30f*s,-0.02f*s,0.04f*s, rgba(0.95f,0.92f,0.85f));
    m.disc(c+R*(-0.36f*s)+U*(-0.28f*s), R*(s),U*(s), 0.08f, rgba(0.95f,0.92f,0.85f));
    m.disc(c+R*(0.10f*s)+U*(0.10f*s), R*(s),U*(s), 0.24f, rgba(0.78f,0.46f,0.16f));
    m.disc(c+R*(0.17f*s)+U*(0.17f*s), R*(s),U*(s), 0.08f, rgba(0.92f,0.66f,0.36f));
}

// ---------------------------------------------------------------- maze -----
static const int MW=21, MH=15;
static const float CELL=2.0f, WALL_H=1.35f, EYE_H=0.55f;
static const char* MAP[MH]={
 "#####################",
 "#...#...........#...#",
 "#.#.#.###.#.###.#.#.#",
 "#.#.......2.......#.#",
 "#.#.#####.#.#####.#.#",
 "#.#.....#.#.#.....#.#",
 "#.#####.#1#3#.#####.#",
 "#.......#C#X#.......#",
 "#.#####.#.#.#.#####.#",
 "#.#.....#.#.#.....#.#",
 "#.#.###.#.#.#.###.#.#",
 "#.#..1..#...#..3..#.#",
 "#.#.###.#####.###.#.#",
 "#.....2.....P.......#",
 "#####################",
};
// grid: 0 open, 1 wall, 2 door grp0, 3 door grp1, 4 door grp2
static int grid[MH][MW];
static bool cheese[MH][MW];
static int startX,startY,denX,denY,bonX,bonY;
static bool doorOpen[3]; static float doorCool[3], doorFlash[3];
static int cheeseLeft;

static RGBA DOOR_COL[3]={ rgba(0.15f,0.95f,0.95f), rgba(0.98f,0.3f,0.85f), rgba(0.98f,0.87f,0.25f) };

static void initMaze(){
    for(int y=0;y<MH;y++) for(int x=0;x<MW;x++){
        char c=MAP[y][x]; cheese[y][x]=false;
        switch(c){
            case '#': grid[y][x]=1; break;
            case '1': grid[y][x]=2; break;
            case '2': grid[y][x]=3; break;
            case '3': grid[y][x]=4; break;
            case 'P': grid[y][x]=0; startX=x; startY=y; break;
            case 'C': grid[y][x]=0; denX=x; denY=y; break;
            case 'X': grid[y][x]=0; bonX=x; bonY=y; break;
            default:  grid[y][x]=0; break;
        }
    }
}
static V3 cellCenter(int cx,int cy){ return v3((cx+0.5f)*CELL,0,(cy+0.5f)*CELL); }
static int cellOfX(float x){ return (int)(x/CELL); }
static bool inB(int x,int y){ return x>=0&&y>=0&&x<MW&&y<MH; }
static bool passP(int x,int y){ return inB(x,y)&&grid[y][x]!=1; }
static bool passC(int x,int y){
    if(!inB(x,y)||grid[y][x]==1) return false;
    if(grid[y][x]>=2 && !doorOpen[grid[y][x]-2]) return false;
    return true;
}

// ------------------------------------------------------------- entities ----
struct Player { float x,z,dx,dz,ndx,ndz; bool moving; float bob; } P;
struct Cat { float x,z,dx,dz,speed; int state; float timer; int col; } cats[6];
static int nCats;
struct { bool on; float x,z,t; } hawk;
struct { bool on; int cx,cy; } bone;
struct { bool on; float t,timer; } bonus;
static float boneTimer;

static int score,hiscore,lives,level,bones,nextLife;
static float dogTimer;
static int state; static float stateT;   // 0 title 1 ready 2 play 3 dying 4 clear 5 gameover 6 pause
static char msg[96]; static float msgT;
static float gTime=0;

struct FloatTxt { char t[32]; float x,z; float age; };
static FloatTxt floats[10]; static int nFloats=0;
static void addFloat(const char*s,V3 p){
    if(nFloats>=10) return;
    FloatTxt&f=floats[nFloats++];
    strncpy(f.t,s,31); f.t[31]=0; f.x=p.x; f.z=p.z; f.age=0;
}
static char fmtBuf[8][256]; static int fmtIdx=0;
static const char* fmt(const char*f,...){
    char*b=fmtBuf[fmtIdx=(fmtIdx+1)&7];
    va_list ap; va_start(ap,f); vsnprintf(b,256,f,ap); va_end(ap); return b;
}
static float frand(){ return rand()/(float)RAND_MAX; }
static V3 playerPos(){ return v3(P.x,0,P.z); }

static void setMsg(const char*s){ strncpy(msg,s,95); msg[95]=0; msgT=2.6f; }
static void addScore(int n){
    score+=n;
    if(score>=nextLife){ lives++; nextLife+=10000; setMsg("EXTRA LIFE!"); }
}
static void captureCat(Cat&c,int pts,const char*what){
    c.state=2; c.timer=5.f; c.dx=c.dz=0;
    addScore(pts);
    addFloat(fmt("%s +%d",what,pts), v3(c.x,0.6f,c.z));
}

static void resetPositions(){
    P.x=(startX+0.5f)*CELL; P.z=(startY+0.5f)*CELL;
    P.dx=P.dz=P.ndx=P.ndz=0; P.moving=false; P.bob=0;
    for(int i=0;i<nCats;i++){
        Cat&c=cats[i];
        c.state=0; c.timer=1.2f+i*2.2f; c.dx=c.dz=0;
        V3 d=cellCenter(denX,denY); c.x=d.x+((i%3)-1)*0.4f; c.z=d.z;
    }
    hawk.on=false; dogTimer=0;
    for(int g=0;g<3;g++) doorOpen[g]=false;
}
static void initLevel(){
    cheeseLeft=0;
    for(int y=0;y<MH;y++) for(int x=0;x<MW;x++){
        bool c = grid[y][x]==0 && !(x==startX&&y==startY) && !(x==denX&&y==denY) && !(x==bonX&&y==bonY);
        cheese[y][x]=c; if(c) cheeseLeft++;
    }
    nCats = 3+level; if(nCats>6) nCats=6;
    for(int i=0;i<6;i++){
        cats[i].col=i;
        cats[i].speed = 2.45f + 0.14f*level + 0.05f*i;
        if(cats[i].speed>3.5f) cats[i].speed=3.5f;
    }
    bone.on=false; boneTimer=7.f;
    bonus.on=false; bonus.timer=6.f; bonus.t=0;
    resetPositions();
}
static void startGame(){
    score=0; lives=3; level=1; bones=0; dogTimer=0; nextLife=10000;
    initLevel(); state=1; stateT=1.6f; setMsg("LEVEL 1 - GET THE CHEESE!");
}
static void loadHi(){ hiscore=0; FILE*f=fopen("mousetrap_hiscore.txt","r"); if(f){ fscanf(f,"%d",&hiscore); fclose(f);} }
static void saveHi(){ FILE*f=fopen("mousetrap_hiscore.txt","w"); if(f){ fprintf(f,"%d",hiscore); fclose(f);} }

// ---------------------------------------------------------------- logic ----
static void toggleDoors(int g){
    if(doorCool[g]>0) return;
    doorCool[g]=0.35f; doorOpen[g]=!doorOpen[g]; doorFlash[g]=1.f;
    if(!doorOpen[g]){ // slamming shut: trap cats standing in these doors
        for(int i=0;i<nCats;i++){ Cat&c=cats[i];
            if(c.state==1){
                int cx=cellOfX(c.x), cy=cellOfX(c.z);
                if(inB(cx,cy)&&grid[cy][cx]==2+g) captureCat(c,200,"TRAPPED");
            }
        }
        setMsg("DOORS SLAM!");
    }
}
static void useBone(){
    if(bones>0 && dogTimer<=0){ bones--; dogTimer=7.f; setMsg("DOG POWER! CATS FEAR YOU!"); }
}
static void killPlayer(const char*why){
    if(state!=2) return;
    state=3; stateT=1.8f; setMsg(why);
}
static void spawnHawk(){
    hawk.on=true; hawk.t=0;
    const int cx[4]={1,MW-2,1,MW-2}, cy[4]={1,1,MH-2,MH-2};
    int best=0; float bd=-1;
    for(int i=0;i<4;i++){
        V3 c=cellCenter(cx[i],cy[i]);
        float d=(c.x-P.x)*(c.x-P.x)+(c.z-P.z)*(c.z-P.z);
        if(d>bd){ bd=d; best=i; }
    }
    V3 c=cellCenter(cx[best],cy[best]); hawk.x=c.x; hawk.z=c.z;
    setMsg("BONUS! ...BUT THE HAWK IS COMING!");
}

static void decideCat(Cat&c,int cx,int cy){
    int pcx=cellOfX(P.x), pcy=cellOfX(P.z);
    const int DX[4]={1,-1,0,0}, DZ[4]={0,0,1,-1};
    float best=1e30f; int bi=-1;
    int opt[4],nopt=0;
    for(int d=0;d<4;d++){
        if(!passC(cx+DX[d],cy+DZ[d])) continue;
        opt[nopt++]=d;
        float dist=(float)(abs(cx+DX[d]-pcx)+abs(cy+DZ[d]-pcy));
        float s = (dogTimer>0) ? -dist : dist;      // flee when we're a dog
        s += frand()*0.9f;
        if(DX[d]==-(int)c.dx && DZ[d]==-(int)c.dz) s+=3.f; // avoid reversing
        if(s<best){ best=s; bi=d; }
    }
    if(nopt==0){ c.dx=c.dz=0; return; }
    if(frand()<0.12f) bi=opt[rand()%nopt];          // a little chaos
    c.dx=DX[bi]; c.dz=DZ[bi];
}

static void readHeld(int&hx,int&hz){
    hx=hz=0;
    GLFWwindow*w=glfwGetCurrentContext();
    if(glfwGetKey(w,GLFW_KEY_LEFT)==GLFW_PRESS ||glfwGetKey(w,GLFW_KEY_A)==GLFW_PRESS) hx--;
    if(glfwGetKey(w,GLFW_KEY_RIGHT)==GLFW_PRESS||glfwGetKey(w,GLFW_KEY_D)==GLFW_PRESS) hx++;
    if(glfwGetKey(w,GLFW_KEY_UP)==GLFW_PRESS   ||glfwGetKey(w,GLFW_KEY_W)==GLFW_PRESS) hz--;  // -z = screen UP
    if(glfwGetKey(w,GLFW_KEY_DOWN)==GLFW_PRESS ||glfwGetKey(w,GLFW_KEY_S)==GLFW_PRESS) hz++;  // +z = screen DOWN
}

static void updatePlayer(float dt){
    int hx,hz; readHeld(hx,hz);
    bool anyHeld=(hx||hz);
    if(P.ndx==0&&P.ndz==0&&anyHeld){ P.ndx=lastDX; P.ndz=lastDZ; }
    if(P.dx==0&&P.dz==0){                                    // stopped: try to start
        if(anyHeld){
            int tx=P.ndx, tz=P.ndz;
                        if(tx==0&&tz==0){ tx=lastDX; tz=lastDZ; }
            int cx=cellOfX(P.x), cy=cellOfX(P.z);
            if(passP(cx+tx,cy+tz)){ P.dx=tx; P.dz=tz; P.ndx=tx; P.ndz=tz; }
        }
    }
    if(P.dx||P.dz){
        float nx=P.x+P.dx*3.4f*dt, nz=P.z+P.dz*3.4f*dt;
        int cx=cellOfX(P.x), cy=cellOfX(P.z);
        float ccx=(cx+0.5f)*CELL, ccz=(cy+0.5f)*CELL;
        bool crossed=false;                                  // STRICT compare here:
        if(P.dx>0&&P.x< ccx&&nx>=ccx){nx=ccx;crossed=true;}
        if(P.dx<0&&P.x> ccx&&nx<=ccx){nx=ccx;crossed=true;}
        if(P.dz>0&&P.z< ccz&&nz>=ccz){nz=ccz;crossed=true;}
        if(P.dz<0&&P.z> ccz&&nz<=ccz){nz=ccz;crossed=true;}
        P.x=nx; P.z=nz;
        if(crossed){
            if(!anyHeld){                                    // released keys -> stop at center
                P.dx=P.dz=0;
            } else {
                if((P.ndx||P.ndz)&&(P.ndx!=P.dx||P.ndz!=P.dz)&&passP(cx+P.ndx,cy+P.ndz)){
                    P.dx=P.ndx; P.dz=P.ndz;                  // buffered turn
                }
                if(!passP(cx+P.dx,cy+P.dz)){ P.dx=P.dz=0; }  // hit a wall
            }
        }
    }
    P.moving=(P.dx||P.dz);
    if(P.moving) P.bob+=dt*11;
}

static void updateCats(float dt){
    for(int i=0;i<nCats;i++){ Cat&c=cats[i];
        if(c.state==0){ c.timer-=dt;
            if(c.timer<=0){ c.state=1; V3 d=cellCenter(denX,denY); c.x=d.x; c.z=d.z; c.dx=0;c.dz=1; }
            continue;
        }
        if(c.state==2){ c.timer-=dt;
            if(c.timer<=0){ c.state=0; c.timer=1.5f; V3 d=cellCenter(denX,denY); c.x=d.x; c.z=d.z; }
            continue;
        }
        float nx=c.x+c.dx*c.speed*dt, nz=c.z+c.dz*c.speed*dt;
        int cx=cellOfX(c.x), cy=cellOfX(c.z);
        float ccx=(cx+0.5f)*CELL, ccz=(cy+0.5f)*CELL;
        bool crossed=false;
        if(c.dx>0&&c.x< ccx&&nx>=ccx){nx=ccx;crossed=true;}
        if(c.dx<0&&c.x> ccx&&nx<=ccx){nx=ccx;crossed=true;}
        if(c.dz>0&&c.z< ccz&&nz>=ccz){nz=ccz;crossed=true;}
        if(c.dz<0&&c.z> ccz&&nz<=ccz){nz=ccz;crossed=true;}
        c.x=nx; c.z=nz;
        if(crossed) decideCat(c,cx,cy);
    }
}
static void updateHawk(float dt){
    if(!hawk.on) return;
    hawk.t+=dt;
    if(hawk.t>9.f){ hawk.on=false; return; }
    V3 d=vnorm(playerPos()-v3(hawk.x,0,hawk.z));
    float sp=2.55f+0.08f*level;
    hawk.x+=d.x*sp*dt; hawk.z+=d.z*sp*dt;
    float dx=hawk.x-P.x, dz=hawk.z-P.z;
    if(dx*dx+dz*dz<0.55f*0.55f) killPlayer("THE HAWK GOT YOU!");
}
static void updateItems(float dt){
    if(!bone.on && bones<3){
        boneTimer-=dt;
        if(boneTimer<=0){
            boneTimer=11.f+frand()*5.f;
            for(int t=0;t<80;t++){
                int x=1+rand()%(MW-2), y=1+rand()%(MH-2);
                if(grid[y][x]!=0) continue;
                V3 c=cellCenter(x,y);
                float d=(c.x-P.x)*(c.x-P.x)+(c.z-P.z)*(c.z-P.z);
                if(d<100) continue;
                bone.on=true; bone.cx=x; bone.cy=y; break;
            }
        }
    }
    if(bonus.on){ bonus.t+=dt; if(bonus.t>9.f){ bonus.on=false; bonus.timer=12.f+frand()*5.f; } }
    else { bonus.timer-=dt; if(bonus.timer<=0){ bonus.on=true; bonus.t=0; } }
}
static void checkPickups(){
    int cx=cellOfX(P.x), cy=cellOfX(P.z);
    if(inB(cx,cy)&&cheese[cy][cx]){
        cheese[cy][cx]=false; cheeseLeft--; addScore(25);
        if(cheeseLeft==0){ state=4; stateT=2.5f; addScore(500); setMsg("LEVEL CLEAR! +500"); }
    }
    if(bone.on&&cx==bone.cx&&cy==bone.cy){
        bone.on=false; bones++; addScore(100); addFloat("GOT A BONE!",playerPos()+v3(0,0.5f,0));
    }
    if(bonus.on){
        V3 b=cellCenter(bonX,bonY);
        float dx=P.x-b.x, dz=P.z-b.z;
        if(dx*dx+dz*dz<0.9f*0.9f){ bonus.on=false; addScore(500); spawnHawk(); }
    }
}
static void checkCats(){
    for(int i=0;i<nCats;i++){ Cat&c=cats[i];
        if(c.state!=1) continue;
        float dx=P.x-c.x, dz=P.z-c.z;
        if(dx*dx+dz*dz<0.62f*0.62f){
            if(dogTimer>0) captureCat(c,300,"SCARED");
            else { killPlayer("CAUGHT BY A CAT!"); return; }
        }
    }
}
static void update(float dt){
    gTime+=dt;
    for(int g=0;g<3;g++){ doorCool[g]-=dt; doorFlash[g]-=dt*2.5f; if(doorFlash[g]<0)doorFlash[g]=0; }
    if(msgT>0) msgT-=dt;
    for(int i=0;i<nFloats;i++){ floats[i].age+=dt; if(floats[i].age>1.1f){ floats[i]=floats[--nFloats]; i--; } }

    switch(state){
    case 2: // play
        if(dogTimer>0) dogTimer-=dt;
        updatePlayer(dt); updateCats(dt); updateHawk(dt); updateItems(dt);
        checkPickups(); if(state==2) checkCats();
        break;
    case 1: // ready
        stateT-=dt;
        if(stateT<=0) state=2;
        break;
    case 3: // dying
        stateT-=dt;
        if(stateT<=0){
            lives--;
            if(lives<=0){ state=5; if(score>hiscore){hiscore=score;} saveHi(); }
            else { resetPositions(); state=1; stateT=1.4f; }
        }
        break;
    case 4: // level clear
        stateT-=dt;
        if(stateT<=0){ level++; initLevel(); state=1; stateT=1.6f; setMsg(fmt("LEVEL %d - FASTER CATS!",level)); }
        break;
    }
}

// -------------------------------------------------------------- rendering --
static const char* VSRC=
 "#version 330 core\n"
 "layout(location=0) in vec3 aPos;\n"
 "layout(location=1) in vec4 aCol;\n"
 "uniform mat4 uMVP;\n"
 "out vec4 vCol; out vec3 vWorld;\n"
 "void main(){ vCol=aCol; vWorld=aPos; gl_Position=uMVP*vec4(aPos,1.0); }\n";
static const char* FSRC=
 "#version 330 core\n"
 "in vec4 vCol; in vec3 vWorld;\n"
 "uniform vec3 uCamPos; uniform vec3 uFogColor;\n"
 "uniform float uFogStart,uFogEnd,uFogAmt;\n"
 "out vec4 frag;\n"
 "void main(){\n"
 "  frag=vCol;\n"
 "  float d=distance(vWorld,uCamPos);\n"
 "  float f=clamp((d-uFogStart)/(uFogEnd-uFogStart),0.0,1.0)*uFogAmt;\n"
 "  frag.rgb=mix(frag.rgb,uFogColor,f);\n"
 "}\n";

static GLuint prog; static GLint uMVP,uCamPos,uFogColor,uFogStart,uFogEnd,uFogAmt;
static Mesh world,hud;
static int fbW=1280, fbH=720;
static float tBlend=1.f, lookBack=0.f, yaw=0;
static V3 camR=v3(1,0,0), camU=v3(0,1,0), camPosW=v3(0,0,0);

static GLuint compile(GLenum type,const char*src){
    GLuint s=glCreateShader(type); glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char log[1024]; glGetShaderInfoLog(s,1024,nullptr,log); fprintf(stderr,"SHADER: %s\n",log); }
    return s;
}

static void drawWallCell(Mesh&m,int x,int z,RGBA side,RGBA top){
    float x0=x*CELL,x1=(x+1)*CELL,z0=z*CELL,z1=(z+1)*CELL;
    V3 a=v3(x0,0,z0),b=v3(x1,0,z0),c=v3(x1,0,z1),d=v3(x0,0,z1);
    V3 A=a+v3(0,WALL_H,0),B=b+v3(0,WALL_H,0),C=c+v3(0,WALL_H,0),D=d+v3(0,WALL_H,0);
    m.quad(a,b,B,A,rgba(side.r*0.85f,side.g*0.85f,side.b*0.85f));
    m.quad(b,c,C,B,side);
    m.quad(c,d,D,C,rgba(side.r*0.85f,side.g*0.85f,side.b*0.85f));
    m.quad(d,a,A,D,side);
    m.quad(A,B,C,D,top);
}
static void buildWorld(float s){
    world.reset();
    // floor
    for(int y=0;y<MH;y++) for(int x=0;x<MW;x++){
        if(grid[y][x]==1) continue;
        float x0=x*CELL,x1=(x+1)*CELL,z0=y*CELL,z1=(y+1)*CELL;
        RGBA f = ((x+y)&1)?rgba(0.10f,0.10f,0.17f):rgba(0.085f,0.085f,0.15f);
        world.quad(v3(x0,0,z0),v3(x1,0,z0),v3(x1,0,z1),v3(x0,0,z1),f);
    }
    // bonus pad
    {
        float x0=bonX*CELL+0.2f,x1=(bonX+1)*CELL-0.2f,z0=bonY*CELL+0.2f,z1=(bonY+1)*CELL-0.2f;
        RGBA g=rgba(0.6f,0.5f,0.15f);
        world.quad(v3(x0,0.01f,z0),v3(x1,0.01f,z0),v3(x1,0.01f,z0+0.12f),v3(x0,0.01f,z0+0.12f),g);
        world.quad(v3(x0,0.01f,z1),v3(x1,0.01f,z1),v3(x1,0.01f,z1-0.12f),v3(x0,0.01f,z1-0.12f),g);
        world.quad(v3(x0,0.01f,z0),v3(x0+0.12f,0.01f,z0),v3(x0+0.12f,0.01f,z1),v3(x0,0.01f,z1),g);
        world.quad(v3(x1,0.01f,z0),v3(x1-0.12f,0.01f,z0),v3(x1-0.12f,0.01f,z1),v3(x1,0.01f,z1),g);
    }
    // open door floor tiles (translucent)
    for(int y=0;y<MH;y++) for(int x=0;x<MW;x++){
        int g=grid[y][x]; if(g<2) continue; g-=2;
        if(!doorOpen[g]) continue;
        RGBA c=DOOR_COL[g]; c.a=0.30f+0.4f*doorFlash[g];
        world.quad(v3(x*CELL,0.02f,y*CELL),v3((x+1)*CELL,0.02f,y*CELL),
                   v3((x+1)*CELL,0.02f,(y+1)*CELL),v3(x*CELL,0.02f,(y+1)*CELL),c);
    }
    // hawk shadow
    if(hawk.on)
        world.disc(v3(hawk.x,0.03f,hawk.z),v3(0.55f,0,0),v3(0,0,0.4f),1.f,rgba(0,0,0,0.35f),14);
    // walls and closed doors
    for(int y=0;y<MH;y++) for(int x=0;x<MW;x++){
        int g=grid[y][x];
        if(g==1){
            float v=((x*7+y*13)%5)*0.012f;
            drawWallCell(world,x,y, rgba(0.10f+v,0.16f+v,0.42f+v), rgba(0.28f+v,0.5f+v,0.95f+v));
        } else if(g>=2){
            int gi=g-2;
            if(!doorOpen[gi]){
                float pulse=0.12f*sinf(gTime*6+gi*2)+0.35f*doorFlash[gi];
                RGBA c=DOOR_COL[gi];
                drawWallCell(world,x,y, rgba(c.r*0.5f,c.g*0.5f,c.b*0.5f),
                             rgba(c.r*0.75f+pulse,c.g*0.75f+pulse,c.b*0.75f+pulse));
            } else { // posts at wall-adjacent edges
                RGBA c=DOOR_COL[gi];
                bool L=inB(x-1,y)&&grid[y][x-1]==1, Rr=inB(x+1,y)&&grid[y][x+1]==1;
                bool Uu=inB(x,y-1)&&grid[y-1][x]==1, Dd=inB(x,y+1)&&grid[y+1][x]==1;
                auto post=[&](float px,float pz){
                    world.tri(v3(px,0,pz),v3(px,WALL_H,pz),v3(px,WALL_H+0.001f,pz),c); // marker
                    drawWallCell(world,x,y,c,rgba(1,1,1,0)); // placeholder avoided below
                };
                // simple thin posts via small boxes:
                auto mini=[&](float cx,float cz){
                    float r=0.12f;
                    V3 a=v3(cx-r,0,cz-r),b=v3(cx+r,0,cz-r),cc=v3(cx+r,0,cz+r),d=v3(cx-r,0,cz+r);
                    V3 A=a+v3(0,WALL_H,0),B=b+v3(0,WALL_H,0),C=cc+v3(0,WALL_H,0),D=d+v3(0,WALL_H,0);
                    world.quad(a,b,B,A,c); world.quad(b,cc,C,B,c);
                    world.quad(cc,d,D,C,c); world.quad(d,a,A,D,c); world.quad(A,B,C,D,c);
                };
                (void)post;
                if(L) mini(x*CELL+0.1f,(y+0.5f)*CELL);
                if(Rr) mini((x+1)*CELL-0.1f,(y+0.5f)*CELL);
                if(Uu) mini((x+0.5f)*CELL,y*CELL+0.1f);
                if(Dd) mini((x+0.5f)*CELL,(y+1)*CELL-0.1f);
            }
        }
    }
    // cheese
    for(int y=0;y<MH;y++) for(int x=0;x<MW;x++)
        if(cheese[y][x]) sprCheese(world, cellCenter(x,y)+v3(0,0.32f,0), camR,camU, 0.55f);
    // bone
    if(bone.on){
        float hop=0.35f+0.06f*sinf(gTime*4);
        sprBone(world, cellCenter(bone.cx,bone.cy)+v3(0,hop,0), camR,camU, 0.75f);
    }
    // bonus
    if(bonus.on){
        bool blink = bonus.t<7.f || ((int)(bonus.t*6)&1);
        if(blink) sprBonus(world, cellCenter(bonX,bonY)+v3(0,0.42f,0), camR,camU, 0.85f);
    }
    // cats
    for(int i=0;i<nCats;i++){ Cat&c=cats[i];
        RGBA fur[6]={rgba(0.95f,0.55f,0.15f),rgba(0.65f,0.65f,0.75f),rgba(0.93f,0.87f,0.78f),
                     rgba(0.38f,0.33f,0.42f),rgba(0.85f,0.45f,0.45f),rgba(0.5f,0.7f,0.55f)};
        if(c.state==2) continue;
        float y=0.45f;
        if(c.state==0) y=0.45f+0.05f*sinf(gTime*5+i*2);
        sprCat(world, v3(c.x,y,c.z), camR,camU, 0.95f, fur[c.col%6], 1+0.05f*sinf(gTime*8+i));
    }
    // hawk
    if(hawk.on)
        sprHawk(world, v3(hawk.x,1.75f,hawk.z), camR,camU, 1.25f, 0.22f*sinf(gTime*13));
    // player sprite (visible in 2D / transition only)
    float pa=(s-0.2f)/0.5f; if(pa<0)pa=0; if(pa>1)pa=1;
    if(pa>0.01f){
        RGBA tint = dogTimer>0 ? rgba(0.74f,0.52f,0.28f) : rgba(0.66f,0.68f,0.75f);
        if(dogTimer>0) sprDog(world, playerPos()+v3(0,0.5f,0), camR,camU, 1.f);
        else           sprMouse(world, playerPos()+v3(0,0.5f,0), camR,camU, 1.f, rgba(tint.r,tint.g,tint.b,pa));
        // facing arrow on the floor
        float fx=(P.dx||P.dz)?P.dx:cosf(yaw), fz=(P.dx||P.dz)?P.dz:sinf(yaw);
        V3 tip=playerPos()+v3(fx,0.02f,fz)*0.95f;
        V3 sd=v3(-fz,0,fx)*0.25f, base=playerPos()+v3(fx,0.02f,fz)*0.25f;
        world.tri(base-sd, base+sd, tip, rgba(1,1,1,0.5f*pa));
    }
    // floor labels (fade in with top view)
    if(s>0.05f){
        drawText3D(world, cellCenter(denX,denY)+v3(0,0.04f,0), camR,camU, 0.16f, rgba(1,0.4f,0.4f,0.75f*s), "CAT DEN");
        drawText3D(world, cellCenter(bonX,bonY)+v3(0,0.04f,0), camR,camU, 0.16f, rgba(1,0.85f,0.3f,0.7f*s), "BONUS");
    }
    // floating score texts (billboarded)
    for(int i=0;i<nFloats;i++){
        FloatTxt&f=floats[i];
        drawText3D(world, v3(f.x,0.9f+f.age*0.8f,f.z), camR,camU, 0.16f, rgba(1,0.9f,0.3f,1-f.age), f.t);
    }
}

static void buildHUD(float s){
    hud.reset();
    float w=fbW,h=fbH, u=h/720.f;
    RGBA white=rgba(0.92f,0.95f,1), gold=rgba(1,0.85f,0.3), dim=rgba(0.55f,0.6f,0.7f);
    auto center=[&](const char*t,float y,float sc,RGBA c){
        drawText2D(hud,(w-textW(t,sc))*0.5f,y,sc,c,t);
    };
    if(state==0){ // TITLE
        hud.quad(v3(0,0,0),v3(w,0,0),v3(w,h,0),v3(0,h,0),rgba(0,0,0,0.55f));
        sprMouse(hud,v3(w*0.5f,h*0.86f,0),v3(1,0,0),v3(0,1,0),54*u);
        center("MOUSE TRAP",h*0.66f,10*u,gold);
        center("A TRIBUTE TO EXIDY'S 1981 ARCADE CLASSIC",h*0.60f,3*u,dim);
        if(((int)(gTime*2)&1)) center("PRESS ENTER TO START",h*0.50f,4.5*u,rgba(0.4f,1,0.5f));
        const char*lines[]={
            "ARROWS OR WASD - MOVE THE MOUSE",
            "STOP MOVING - RISE UP TO THE 2D MAP.  MOVE - DIVE INTO MOUSE-EYE 3D",
            "HOLD TAB - LOOK BEHIND YOU IN 3D",
            "1 2 3 - SLAM THE COLORED DOORS SHUT ON CATS",
            "SPACE - USE A BONE TO TURN INTO A DOG",
            "EAT ALL THE CHEESE. AVOID THE CATS!"
        };
        for(int i=0;i<6;i++) center(lines[i],h*0.40f-i*26*u,3*u,white);
        center(fmt("HI SCORE %06d",hiscore),h*0.09f,4*u,gold);
        return;
    }
    // in-game chrome
    drawText2D(hud,16*u,h-34*u,3.5*u,white,fmt("SCORE %06d",score));
    drawText2D(hud,16*u,h-60*u,2.5*u,dim,fmt("CHEESE %03d",cheeseLeft));
    center(fmt("HI %06d",hiscore),h-34*u,3.5*u,gold);
    drawText2D(hud,w-16*u-6*3.5f*u*(level>99?9:8),h-34*u,3.5*u,white,fmt("LEVEL %02d",level));
    const char*mode = s>0.75f?"TOP VIEW":(s<0.25f?"MOUSE-EYE VIEW - TAB LOOKS BEHIND":"...");
    center(mode,h-60*u,2.5*u,dim);
    if(msgT>0){
        RGBA mc=rgba(1,1,0.6f,msgT>2?1:(msgT/0.6f>1?1:msgT/0.6f));
        center(msg,h*0.58f,4*u,mc);
    }
    if(state==1) center("READY?",h*0.5f,7*u,rgba(0.4f,1,0.5f));
    if(dogTimer>0) center(fmt("DOG %d",(int)dogTimer+1),h*0.72f,6*u,rgba(1,0.6f,0.2f));
    // lives
    for(int i=0;i<lives&&i<6;i++)
        sprMouse(hud,v3((34+i*40)*u,30*u,0),v3(1,0,0),v3(0,1,0),26*u);
    // door lamps
    float dx0=w*0.5f-90*u;
    drawText2D(hud,dx0-14*u,74*u,2.5*u,dim,"DOORS 1 2 3");
    for(int g=0;g<3;g++){
        float x=dx0+g*70*u;
        RGBA c=DOOR_COL[g];
        if(!doorOpen[g]){ c.r*=0.3f; c.g*=0.3f; c.b*=0.3f; }
        hud.quad(v3(x,20*u,0),v3(x+44*u,20*u,0),v3(x+44*u,64*u,0),v3(x,64*u,0),c);
        drawText2D(hud,x+14*u,36*u,3.5*u,rgba(0,0,0,0.8f),fmt("%d",g+1));
    }
    // bones
    sprBone(hud,v3(w-190*u,44*u,0),v3(1,0,0),v3(0,1,0),34*u);
    drawText2D(hud,w-150*u,36*u,3.5*u,white,fmt("X%d",bones));
    drawText2D(hud,w-190*u,14*u,2.2*u,dim,"SPACE = DOG");
    if(state==5){ // game over
        hud.quad(v3(0,0,0),v3(w,0,0),v3(w,h,0),v3(0,h,0),rgba(0,0,0,0.6f));
        center("GAME OVER",h*0.55f,10*u,rgba(1,0.3f,0.3f));
        center(fmt("FINAL SCORE %06d",score),h*0.45f,4.5*u,white);
        if(((int)(gTime*2)&1)) center("PRESS ENTER TO PLAY AGAIN",h*0.35f,4*u,rgba(0.4f,1,0.5f));
    }
    if(state==6) center("PAUSED - PRESS P",h*0.5f,6*u,white);
}

static void render(){
    float w=fbW,h=fbH, aspect=w/(h>1?h:1);
    float s=smoothstep01(tBlend);
    // camera pose interpolation: first-person <-> top-down
    float bobY=P.moving?0.035f*sinf(P.bob):0;
    float yv=yaw+lookBack;
    V3 fpEye=v3(P.x,EYE_H+bobY,P.z);
    V3 fpTgt=fpEye+v3(cosf(yv),0,sinf(yv))*4.f;
    V3 mzc=v3(MW*CELL*0.5f,0,MH*CELL*0.5f);
    V3 topEye=mzc+v3(0,52.f,0.001f);
    V3 eye=vlerp(fpEye,topEye,s), tgt=vlerp(fpTgt,mzc,s);
    V3 up=vnorm(vlerp(v3(0,1,0),v3(0,0,-1),s));
    Mat4 view=matLookAt(eye,tgt,up);
    Mat4 persp=matPerspective(66,aspect,0.1f,140.f);
    float mazeW=MW*CELL, mazeH=MH*CELL;
    float halfH=mazeH*0.56f, halfW=halfH*aspect;
    if(aspect<mazeW/mazeH){ halfW=mazeW*0.56f; halfH=halfW/aspect; }
    Mat4 ortho=matOrtho(-halfW,halfW,-halfH,halfH,10.f,140.f);
    Mat4 proj=matLerp(persp,ortho,s);
    Mat4 mvp=matMul(proj,view);
    camPosW=eye;
    camR=v3(view.m[0],view.m[4],view.m[8]);
    camU=v3(view.m[1],view.m[5],view.m[9]);

    buildWorld(s);
    buildHUD(s);

    V3 bg=vlerp(v3(0.03f,0.03f,0.08f),v3(0.015f,0.015f,0.04f),s);
    glViewport(0,0,fbW,fbH);
    glClearColor(bg.x,bg.y,bg.z,1);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    world.draw(prog,mvp,eye,bg,8.f,34.f,(1-s)*0.9f);
    glDisable(GL_DEPTH_TEST);
    hud.draw(prog,matOrtho(0,(float)fbW,0,(float)fbH,-1,1),v3(0,0,0),bg,0,1,0);
    glEnable(GL_DEPTH_TEST);
}

// Mesh::draw needs program+uniforms; implement here (after locations exist)
void Mesh_draw_impl(Mesh&m,GLuint p,const Mat4&mvp,V3 cam,V3 fogC,float fs,float fe,float fa);

// ----------------------------------------------------------------- input ---
static GLFWwindow* WIN=nullptr;
static void keyCB(GLFWwindow*,int key,int, int act,int){
    if(act!=GLFW_PRESS) return;
    if(key==GLFW_KEY_ESCAPE){ state=0; return; }
    if(key==GLFW_KEY_ENTER){
        if(state==0||state==5) startGame();
        return;
    }
    if(key==GLFW_KEY_P && (state==2||state==6)){ state=(state==2)?6:2; return; }
    if(state!=2 && state!=1) { if(key==GLFW_KEY_SPACE&&state==0) startGame(); return; }
    switch(key){
        case GLFW_KEY_LEFT: case GLFW_KEY_A: P.ndx=-1; P.ndz= 0; lastDX=-1; lastDZ= 0; break;
        case GLFW_KEY_RIGHT:case GLFW_KEY_D: P.ndx= 1; P.ndz= 0; lastDX= 1; lastDZ= 0; break;
        case GLFW_KEY_UP:   case GLFW_KEY_W: P.ndx= 0; P.ndz=-1; lastDX= 0; lastDZ=-1; break;
        case GLFW_KEY_DOWN: case GLFW_KEY_S: P.ndx= 0; P.ndz= 1; lastDX= 0; lastDZ= 1; break;
        case GLFW_KEY_1: if(state==2) toggleDoors(0); break;
        case GLFW_KEY_2: if(state==2) toggleDoors(1); break;
        case GLFW_KEY_3: if(state==2) toggleDoors(2); break;
        case GLFW_KEY_SPACE: if(state==2) useBone(); break;
    }
}
static void fbCB(GLFWwindow*,int w,int h){ fbW=w; fbH=h; }

// ------------------------------------------------------------------ main ---
void Mesh_draw_impl(Mesh&m,GLuint p,const Mat4&mvp,V3 cam,V3 fogC,float fs,float fe,float fa){
    if(m.V.empty()) return;
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER,m.V.size()*sizeof(Vert),m.V.data(),GL_STREAM_DRAW);
    glUseProgram(p);
    glUniformMatrix4fv(uMVP,1,GL_FALSE,mvp.m);
    glUniform3f(uCamPos,cam.x,cam.y,cam.z);
    glUniform3f(uFogColor,fogC.x,fogC.y,fogC.z);
    glUniform1f(uFogStart,fs); glUniform1f(uFogEnd,fe); glUniform1f(uFogAmt,fa);
    glDrawArrays(GL_TRIANGLES,0,(GLsizei)m.V.size());
    glBindVertexArray(0);
}
// hook Mesh::draw to the impl (kept out-of-line for clarity)
// (declared inside Mesh earlier as draw(...); provide body via free function)
namespace{ struct MeshDrawBinder{}; }
// We declared draw() inside Mesh without a body; simplest: define it here.
// (C++ allows defining a member function after the class in the same TU.)

int main(){
    srand((unsigned)time(nullptr));
    if(!glfwInit()){ fprintf(stderr,"glfw init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES,4);
    WIN=glfwCreateWindow(1280,720,"MOUSE TRAP - 1981 recreation",nullptr,nullptr);
    if(!WIN){ glfwTerminate(); return 1; }
    glfwMakeContextCurrent(WIN);
    glfwSwapInterval(1);
    glewExperimental=GL_TRUE;
    if(glewInit()!=GLEW_OK){ fprintf(stderr,"glew init failed\n"); return 1; }
    glfwSetKeyCallback(WIN,keyCB);
    glfwSetFramebufferSizeCallback(WIN,fbCB);
    glfwGetFramebufferSize(WIN,&fbW,&fbH);

    GLuint vs=compile(GL_VERTEX_SHADER,VSRC), fs=compile(GL_FRAGMENT_SHADER,FSRC);
    prog=glCreateProgram(); glAttachShader(prog,vs); glAttachShader(prog,fs); glLinkProgram(prog);
    GLint ok; glGetProgramiv(prog,GL_LINK_STATUS,&ok);
    if(!ok){ char log[1024]; glGetProgramInfoLog(prog,1024,nullptr,log); fprintf(stderr,"LINK: %s\n",log); return 1; }
    uMVP=glGetUniformLocation(prog,"uMVP"); uCamPos=glGetUniformLocation(prog,"uCamPos");
    uFogColor=glGetUniformLocation(prog,"uFogColor");
    uFogStart=glGetUniformLocation(prog,"uFogStart"); uFogEnd=glGetUniformLocation(prog,"uFogEnd");
    uFogAmt=glGetUniformLocation(prog,"uFogAmt");

    world.init(); hud.init();
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL);

    initMaze(); loadHi(); initLevel(); state=0;

    double last=glfwGetTime();
    while(!glfwWindowShouldClose(WIN)){
        double now=glfwGetTime();
        float dt=(float)(now-last); last=now;
        if(dt>0.05f) dt=0.05f;

        // camera mode: moving -> 3D, stopped -> 2D
        float target = (state==2 && P.moving) ? 0.f : 1.f;
        tBlend += (target-tBlend)*(1.f-expf(-dt*5.f));
        // look behind (TAB) in 3D
        float lbTarget = (glfwGetKey(WIN,GLFW_KEY_TAB)==GLFW_PRESS)?PI:0.f;
        lookBack += (lbTarget-lookBack)*(1.f-expf(-dt*9.f));
        // smooth facing yaw
        if(P.dx||P.dz){
            float ty=atan2f(P.dz,P.dx);
            float d=ty-yaw;
            while(d>PI)d-=2*PI; while(d<-PI)d+=2*PI;
            yaw+=d*(1.f-expf(-dt*12.f));
        }
        update(dt);
        render();
        glfwSwapBuffers(WIN);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
// ---------------------------------------------------------------------------
// Mesh::draw definition (member declared in struct Mesh)
// ---------------------------------------------------------------------------