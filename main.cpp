#ifdef __APPLE__
  #define GL_SILENCE_DEPRECATION
  #include <OpenGL/gl.h>
#else
  #include <GL/gl.h>
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

// ── Resolution ──────────────────────────────────────────────────────────────
constexpr int WIDTH  = 800;
constexpr int HEIGHT = 800;
constexpr int TILE   = 16;
constexpr int TILES_X = (WIDTH  + TILE - 1) / TILE;
constexpr int TILES_Y = (HEIGHT + TILE - 1) / TILE;
constexpr int TILE_COUNT = TILES_X * TILES_Y;

// ── Pixel buffer ────────────────────────────────────────────────────────────
using PixelBuffer = std::vector<uint8_t>;

static PixelBuffer displayPixels(WIDTH * HEIGHT * 3, 0);
static PixelBuffer stagingPixels(WIDTH * HEIGHT * 3, 0);

inline void setPixel(PixelBuffer& buffer, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    const int index = (y * WIDTH + x) * 3;
    buffer[index + 0] = r;
    buffer[index + 1] = g;
    buffer[index + 2] = b;
}

// ═══════════════════════════════════════════════════════════════════════════
//  3D Vector (float)
// ═══════════════════════════════════════════════════════════════════════════
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    inline Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    inline Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    inline Vec3 operator*(float s)        const { return {x*s, y*s, z*s}; }
    inline float dot(const Vec3& v)       const { return x*v.x + y*v.y + z*v.z; }
    inline Vec3 cross(const Vec3& v)      const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    inline float length() const { return std::sqrt(x*x + y*y + z*z); }
    inline Vec3 normalised() const {
        float l2 = x*x + y*y + z*z;
        if (l2 <= 0.0f) return {0,0,0};
        float inv = 1.0f / std::sqrt(l2);
        return {x*inv, y*inv, z*inv};
    }
};

// Bright neon palette — one is picked per spawn. (Defined here so Object /
// ObjectSnapshot / CameraState can use it; the palette table itself lives
// further down with the rest of the test-particle parameters.)
struct NeonRGB { uint8_t r, g, b; };

struct Object {
    Vec3   pos;
    Vec3   vel;
    NeonRGB color;
};

struct ObjectSnapshot {
    Vec3   pos;
    NeonRGB color;
};

struct CameraState {
    float dist;
    float yaw;
    float pitch;
    float time;       // animation clock (seconds)
    std::vector<ObjectSnapshot> objects;

    // Precomputed once per frame (in requestRender) so render workers don't
    // redo trig/cross/normalise on every tile.
    Vec3  camPos;
    Vec3  forward;
    Vec3  right;
    Vec3  up;
    float halfFov;

    // Bounding sphere covering every object's interaction radius. The per-step
    // inner object loop only runs when the ray is inside this sphere.
    Vec3  objCenter;
    float objBoundR2;
};

constexpr int OBJECT_LIMIT = 64;  // cap so traceRay's per-step loop stays cheap

// ── Animation parameters ────────────────────────────────────────────────────
// Keplerian-ish disk rotation: ω(r) = DISK_OMEGA_K / r^1.5 (faster at smaller r).
// Tuned for visual liveliness, not physical accuracy.
constexpr float DISK_OMEGA_K = 30.0f;
// Hot-clump pattern in rotating-disk frame uses these existing terms in diskColour.
// Star twinkle frequency.
constexpr float STAR_TWINKLE_RATE = 2.5f;

// ── Test particle (orbiting/infalling object) ──────────────────────────────
// Schwarzschild test-particle equations of motion (massive, slow-v limit):
//     a = -pos/r^3 · (1 + 3h^2/r^2)
// The 3h^2/r^2 term is the GR correction that drives perihelion precession
// and makes orbits with periapsis below ISCO (r = 3 RS) plunge in.
constexpr float OBJECT_RADIUS  = 1.1f;
constexpr float OBJECT_RADIUS2 = OBJECT_RADIUS * OBJECT_RADIUS;

constexpr NeonRGB NEON_PALETTE[] = {
    {  57, 255,  20 },   // neon green
    { 255,  20, 147 },   // hot pink
    {  13, 255, 247 },   // electric cyan
    { 191,  64, 240 },   // neon purple
    { 255, 255,   0 },   // electric yellow
    { 255,  95,  10 },   // neon orange
    {   4, 217, 255 },   // electric blue
    { 255,  16, 240 },   // neon magenta
    { 173, 255,  47 },   // neon chartreuse
    { 255,   7,  58 },   // neon red
};
constexpr int NEON_COUNT = sizeof(NEON_PALETTE) / sizeof(NEON_PALETTE[0]);
// How fast simulation time advances vs real time. Matched to DISK_OMEGA_K so
// the disk rotation and the particle orbit feel like the same clock.
constexpr float OBJECT_TIME_SCALE = 30.0f;

// ═══════════════════════════════════════════════════════════════════════════
//  Black hole parameters
// ═══════════════════════════════════════════════════════════════════════════
constexpr float BH_MASS = 1.0f;
constexpr float RS      = 2.0f * BH_MASS;

constexpr float DISK_INNER = 3.0f * RS;
constexpr float DISK_OUTER = 12.0f * RS;
constexpr float DISK_HALF_THICKNESS = 0.15f;

constexpr float FOV       = 1.2f;
constexpr float MAX_DIST  = 150.0f;   // must exceed max camera distance (100)
constexpr int   MAX_STEPS = 4000;

// Adaptive step parameters — step size grows roughly proportional to distance
// once we're outside the strong-field region, so empty-space marching is cheap.
constexpr float STEP_NEAR = 0.05f;          // small step near BH (close to RS)
constexpr float STEP_FAR_COEFF = 0.08f;     // far-field: stepSize ≈ 0.08 * r
constexpr float STEP_MAX  = 8.0f;           // cap so we don't skip past the disk
constexpr float NEAR_RADIUS = 4.0f * RS;
constexpr float NEAR_RADIUS2 = NEAR_RADIUS * NEAR_RADIUS;

// Squared thresholds (avoid sqrt in hot loop)
constexpr float RS2       = RS * RS;
constexpr float MAX_DIST2 = MAX_DIST * MAX_DIST;
constexpr float DISK_INNER2 = DISK_INNER * DISK_INNER;
constexpr float DISK_OUTER2 = DISK_OUTER * DISK_OUTER;

// ═══════════════════════════════════════════════════════════════════════════
//  Camera state
// ═══════════════════════════════════════════════════════════════════════════
static float camDist      = 30.0f;
static float camYaw       = 0.0f;
static float camPitch     = 1.2f;
static bool  needsRedraw  = true;

// Animation clock that only advances when the user isn't interacting, so the
// disk/stars freeze during camera moves. Updated from the main thread only.
static float                 animTime     = 0.0f;
static std::atomic<bool>     interacting{false};
static double                lastInputTime = 0.0;
constexpr double             INTERACT_LINGER = 0.15;  // sec after last input

// Test-particle list. Owned by main thread; render workers see a snapshot
// (positions only) captured into CameraState at requestRender() time.
static std::vector<Object> objects;

static void spawnObject() {
    if ((int)objects.size() >= OBJECT_LIMIT) {
        objects.erase(objects.begin());
    }
    // Cheap LCG so each spawn is randomized across angle, radius, y, speed,
    // tilt, and palette index.
    static uint32_t seed = 0xC0FFEEu;
    auto rand01 = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)(seed & 0xFFFF) / 65535.0f;
    };

    float angle       = rand01() * 6.2831853f;
    float radius      = 10.0f + 14.0f * rand01();           // 10..24
    float yOffset     = -3.0f +  6.0f * rand01();           // -3..3
    float speedFactor = 0.50f + 0.40f * rand01();           // 0.5..0.9 of v_circ
    float tilt        = (-0.4f + 0.8f * rand01());          // ±0.4 rad out-of-plane vel
    int   colourIdx   = (int)(rand01() * (float)NEON_COUNT);
    if (colourIdx >= NEON_COUNT) colourIdx = NEON_COUNT - 1;

    Vec3 pos = {radius * std::cos(angle), yOffset, radius * std::sin(angle)};
    float vCirc = std::sqrt(1.0f / radius);
    // Tangential base velocity in the xz plane.
    Vec3 tangent = {-std::sin(angle), 0.0f, std::cos(angle)};
    // Add a small out-of-plane component so orbits aren't all coplanar.
    Vec3 vel = tangent * (vCirc * speedFactor * std::cos(tilt));
    vel.y    += vCirc * speedFactor * std::sin(tilt);

    objects.push_back({pos, vel, NEON_PALETTE[colourIdx]});
}

static void stepObjects(float dtReal) {
    float dt = dtReal * OBJECT_TIME_SCALE;
    constexpr int substeps = 6;
    float h = dt / (float)substeps;
    for (int i = 0; i < substeps; ++i) {
        for (size_t k = 0; k < objects.size();) {
            Object& o = objects[k];
            float r2 = o.pos.dot(o.pos);
            // Crossed the horizon (consumed) or escaped to infinity → cull.
            if (r2 < RS2 || r2 > 60.0f * 60.0f) {
                objects.erase(objects.begin() + k);
                continue;
            }
            float invR2 = 1.0f / r2;
            float invR  = std::sqrt(invR2);
            float invR3 = invR2 * invR;
            float invR5 = invR3 * invR2;
            Vec3  hVec  = o.pos.cross(o.vel);
            float h2    = hVec.dot(hVec);
            // a = -pos/r^3 - 3 h^2 pos / r^5  (GR correction in the second term)
            float accelMag = -invR3 - 3.0f * h2 * invR5;
            Vec3 accel = o.pos * accelMag;
            // Symplectic Euler (kick-drift): cheap and conserves energy reasonably.
            o.vel = o.vel + accel * h;
            o.pos = o.pos + o.vel * h;
            ++k;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mouse state
// ═══════════════════════════════════════════════════════════════════════════
static bool   mouseDown  = false;
static double lastMouseX = 0.0;
static double lastMouseY = 0.0;

// ═══════════════════════════════════════════════════════════════════════════
//  Render coordination
// ═══════════════════════════════════════════════════════════════════════════
static std::mutex              renderMutex;
static std::condition_variable renderCv;
static CameraState             pendingCamera{camDist, camYaw, camPitch, 0.0f, {}};
static std::atomic<uint64_t>   renderGeneration{0};
static uint64_t                requestedGeneration = 0;
static uint64_t                completedGeneration = 0;
static bool                    renderRequested = false;
static bool                    renderReady = false;
static bool                    shuttingDown = false;

// Persistent worker pool state
static std::atomic<int>        nextTile{0};
static std::atomic<int>        workersFinished{0};
static int                     numWorkers = 1;
static std::mutex              poolMutex;
static std::condition_variable poolCv;
static std::condition_variable poolDoneCv;
static int                     poolPass = 0;          // bumped each runPass()
static CameraState             poolCamera{};
static uint64_t                poolGeneration = 0;
static int                     poolStride = 1;        // 4 for coarse, 1 for fine
static PixelBuffer*            poolBuffer = nullptr;

// ═══════════════════════════════════════════════════════════════════════════
//  Starfield
// ═══════════════════════════════════════════════════════════════════════════
static inline uint32_t hashStar(int a, int b) {
    uint32_t h = (uint32_t)(a * 374761393 + b * 668265263);
    h = (h ^ (h >> 13)) * 1274126177;
    return h ^ (h >> 16);
}

static inline void starColour(const Vec3& dir, float time, uint8_t& r, uint8_t& g, uint8_t& b) {
    float theta = std::atan2(dir.z, dir.x);
    float yClamp = dir.y < -1.0f ? -1.0f : (dir.y > 1.0f ? 1.0f : dir.y);
    float phi   = std::asin(yClamp);
    int gridX = (int)std::floor(theta * 220.0f);
    int gridY = (int)std::floor(phi * 220.0f);
    uint32_t h = hashStar(gridX, gridY);
    if ((h & 0xFF) < 5) {
        int brightness = 140 + (int)(h >> 8) % 116;
        // Cheap per-star twinkle: phase derived from hash so each star is unique.
        float starPhase = (float)(h >> 24) * (6.2831853f / 256.0f);
        float twinkle = 0.55f + 0.45f * std::sin(time * STAR_TWINKLE_RATE + starPhase);
        float br = (float)brightness * twinkle;
        int tint = (h >> 16) % 4;
        if (tint == 0)      { r = (uint8_t)br; g = (uint8_t)(br * 0.85f); b = (uint8_t)(br * 0.7f); }
        else if (tint == 1) { r = (uint8_t)(br * 0.7f); g = (uint8_t)(br * 0.85f); b = (uint8_t)br; }
        else                { r = (uint8_t)br; g = (uint8_t)br; b = (uint8_t)br; }
    } else {
        r = 2; g = 2; b = 5;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Accretion disk colour
// ═══════════════════════════════════════════════════════════════════════════
// Glowing test particle. Cheap fake-shading + emissive floor; the per-object
// neon RGB picks the hue while shading just modulates brightness.
static inline void objectColour(const Vec3& normal, NeonRGB c,
                                uint8_t& r, uint8_t& g, uint8_t& b) {
    // Constant light direction — initialised once across all calls.
    static const Vec3 lightDir = Vec3{0.6f, 0.7f, 0.4f}.normalised();
    float diffuse = normal.dot(lightDir);
    if (diffuse < 0.0f) diffuse = 0.0f;
    float t = 0.7f + 0.3f * diffuse;       // bright emissive floor (neon look)
    if (t > 1.0f) t = 1.0f;
    r = (uint8_t)((float)c.r * t);
    g = (uint8_t)((float)c.g * t);
    b = (uint8_t)((float)c.b * t);
}

static inline void diskColour(float radius, float angle, float time, uint8_t& r, uint8_t& g, uint8_t& b) {
    float t = 1.0f - (radius - DISK_INNER) / (DISK_OUTER - DISK_INNER);
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    // Differential (Keplerian) rotation: a point at lab-angle `angle` is, at
    // time t, the material that started at angle (angle - ω(r) t).
    // ω(r) = DISK_OMEGA_K / r^1.5.  r * sqrt(r) = r^1.5 — one sqrt, no pow().
    float invR15 = 1.0f / (radius * std::sqrt(radius));
    float phase = time * DISK_OMEGA_K * invR15;
    float localAngle = angle - phase;
    float swirl = std::sin(localAngle * 6.0f + radius * 2.0f) * 0.15f + 0.85f;
    // A second slower lump pattern so the disk has bright clumps that drift.
    float clump = 0.85f + 0.25f * std::sin(localAngle * 2.0f - radius * 0.5f);
    t *= swirl * clump;
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    if (t > 0.7f) {
        r = 255;
        g = (uint8_t)(200 + 55 * ((t - 0.7f) / 0.3f));
        b = (uint8_t)(150 + 105 * ((t - 0.7f) / 0.3f));
    } else if (t > 0.3f) {
        float s = (t - 0.3f) / 0.4f;
        r = (uint8_t)(100 + 155 * s);
        g = (uint8_t)(30 + 170 * s);
        b = (uint8_t)(5 + 145 * s);
    } else {
        float s = t / 0.3f;
        r = (uint8_t)(100 * s);
        g = (uint8_t)(20 * s);
        b = (uint8_t)(5 * s);
    }
    float doppler = 0.6f + 0.4f * std::sin(angle + 1.0f);
    float fr = (float)r * doppler; if (fr > 255.0f) fr = 255.0f;
    float fg = (float)g * doppler; if (fg > 255.0f) fg = 255.0f;
    float fb = (float)b * doppler; if (fb > 255.0f) fb = 255.0f;
    r = (uint8_t)fr; g = (uint8_t)fg; b = (uint8_t)fb;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Ray tracer (geodesic integrator)
//  HW3-analogue optimizations applied:
//    • `min_t`/`max_t`-style early termination via squared-distance compares
//      (RS2, MAX_DIST2, DISK bounds) — no sqrt in the hot termination check.
//    • Adaptive step size — analogue of adaptive sampling: invest compute
//      where the geodesic curves sharply (near BH); in flat space, step size
//      grows linearly with distance so empty space costs O(log r) steps.
// ═══════════════════════════════════════════════════════════════════════════
static inline void traceRay(Vec3 pos, Vec3 dir, float time,
                            const ObjectSnapshot* objs, int objCount,
                            const Vec3& objCenter, float objBoundR2,
                            uint8_t& r, uint8_t& g, uint8_t& b) {
    Vec3 hVec = pos.cross(dir);
    float h2 = hVec.dot(hVec);

    constexpr float OBJ_INFLUENCE  = OBJECT_RADIUS * 6.0f;
    constexpr float OBJ_INFLUENCE2 = OBJ_INFLUENCE * OBJ_INFLUENCE;

    for (int step = 0; step < MAX_STEPS; ++step) {
        float dist2 = pos.dot(pos);

        if (dist2 < RS2) { r = 0; g = 0; b = 0; return; }
        // Escape only when ray is heading outward — otherwise a camera placed
        // outside MAX_DIST would terminate on its very first step.
        if (dist2 > MAX_DIST2 && pos.dot(dir) > 0.0f) {
            starColour(dir.normalised(), time, r, g, b);
            return;
        }

        // Test particles. First check the cluster bounding sphere — single
        // distance test that lets us skip the per-object loop whenever the
        // ray is nowhere near the cluster (typical for off-axis pixels).
        float nearestToObj2 = OBJ_INFLUENCE2;
        if (objCount > 0) {
            Vec3 toCluster = pos - objCenter;
            if (toCluster.dot(toCluster) < objBoundR2) {
                for (int k = 0; k < objCount; ++k) {
                    Vec3 toObj = pos - objs[k].pos;
                    float toObj2 = toObj.dot(toObj);
                    if (toObj2 < OBJECT_RADIUS2) {
                        objectColour(toObj.normalised(), objs[k].color, r, g, b);
                        return;
                    }
                    if (toObj2 < nearestToObj2) nearestToObj2 = toObj2;
                }
            }
        }

        float diskR2 = pos.x * pos.x + pos.z * pos.z;
        if (std::fabs(pos.y) < DISK_HALF_THICKNESS &&
            diskR2 > DISK_INNER2 && diskR2 < DISK_OUTER2) {
            float diskR = std::sqrt(diskR2);
            float angle = std::atan2(pos.z, pos.x);
            diskColour(diskR, angle, time, r, g, b);
            return;
        }

        float invDist2 = 1.0f / dist2;
        float invDist  = std::sqrt(invDist2);

        // Adaptive step size: STEP_NEAR close to the BH; in flat space let
        // it grow ∝ r so distant marching is logarithmic in r.
        float stepSize;
        if (dist2 < NEAR_RADIUS2) {
            stepSize = STEP_NEAR;
        } else {
            float dist = dist2 * invDist;
            stepSize = STEP_FAR_COEFF * dist;
            if (stepSize > STEP_MAX) stepSize = STEP_MAX;
        }

        // Disk-skip guard: when the ray is anywhere inside the disk's radial
        // footprint (plus a margin for the next step), clamp stepSize so the
        // y-component cannot leap across the thin slab between samples.
        if (diskR2 < DISK_OUTER2 + STEP_MAX * STEP_MAX) {
            float dirYAbs = std::fabs(dir.y);
            if (dirYAbs > 1e-6f) {
                float maxStep = (DISK_HALF_THICKNESS * 0.5f) / dirYAbs;
                if (maxStep < stepSize) stepSize = maxStep;
            }
        }

        // Object-skip guard: a fast-marching ray could step past a small
        // sphere between samples. Clamp to half the gap to the nearest one.
        if (nearestToObj2 < OBJ_INFLUENCE2) {
            float toObjLen = std::sqrt(nearestToObj2);
            float maxStep  = (toObjLen - OBJECT_RADIUS) * 0.5f;
            if (maxStep < 0.02f) maxStep = 0.02f;
            if (maxStep < stepSize) stepSize = maxStep;
        }

        // Geodesic deflection: a = -1.5 RS h^2 / r^5 * pos
        float invDist5 = invDist2 * invDist2 * invDist;
        float accelMag = -1.5f * RS * h2 * invDist5;
        Vec3 accel = pos * accelMag;
        dir = dir + accel * stepSize;
        pos = pos + dir * stepSize;
    }

    starColour(dir.normalised(), time, r, g, b);
}

static void clampCamera() {
    constexpr float PI = 3.14159265358979f;
    if (camPitch < 0.1f)        camPitch = 0.1f;
    if (camPitch > PI - 0.1f)   camPitch = PI - 0.1f;
    if (camDist < RS * 2.5f)    camDist  = RS * 2.5f;
    if (camDist > 100.0f)       camDist  = 100.0f;
}

// Render one tile. `stride` controls block size: 1 = full quality,
// >1 = render every Nth pixel and replicate (coarse pass).
static void renderTile(const CameraState& camera,
                       PixelBuffer& buffer,
                       int tileX, int tileY,
                       int stride,
                       uint64_t generation) {
    // Camera basis is precomputed once per frame in requestRender().
    const Vec3& camPos  = camera.camPos;
    const Vec3& forward = camera.forward;
    const Vec3& right   = camera.right;
    const Vec3& up      = camera.up;
    const float halfFov = camera.halfFov;

    int x0 = tileX * TILE;
    int y0 = tileY * TILE;
    int x1 = std::min(x0 + TILE, WIDTH);
    int y1 = std::min(y0 + TILE, HEIGHT);

    for (int py = y0; py < y1; py += stride) {
        if (renderGeneration.load(std::memory_order_relaxed) != generation) return;

        const float ny = (1.0f - 2.0f * (py + 0.5f) / HEIGHT) * halfFov;
        for (int px = x0; px < x1; px += stride) {
            const float nx = (2.0f * (px + 0.5f) / WIDTH - 1.0f) * halfFov;
            Vec3 rayDir = (forward + right * nx + up * ny).normalised();

            uint8_t r, g, b;
            traceRay(camPos, rayDir, camera.time,
                     camera.objects.data(), (int)camera.objects.size(),
                     camera.objCenter, camera.objBoundR2,
                     r, g, b);

            // Replicate to the stride×stride block (coarse pass).
            int bx1 = std::min(px + stride, x1);
            int by1 = std::min(py + stride, y1);
            for (int yy = py; yy < by1; ++yy)
                for (int xx = px; xx < bx1; ++xx)
                    setPixel(buffer, xx, yy, r, g, b);
        }
    }
}

static void requestRender() {
    {
        std::lock_guard<std::mutex> lock(renderMutex);
        pendingCamera.dist  = camDist;
        pendingCamera.yaw   = camYaw;
        pendingCamera.pitch = camPitch;
        pendingCamera.time  = animTime;
        pendingCamera.objects.resize(objects.size());
        for (size_t i = 0; i < objects.size(); ++i)
            pendingCamera.objects[i] = {objects[i].pos, objects[i].color};

        // Camera basis precomputed once, used by every tile.
        float sp = std::sin(camPitch), cp = std::cos(camPitch);
        float sy = std::sin(camYaw),   cy = std::cos(camYaw);
        Vec3 camPos = {camDist * sp * sy, camDist * cp, camDist * sp * cy};
        Vec3 forward = (Vec3{0,0,0} - camPos).normalised();
        Vec3 worldUp = {0, 1, 0};
        if (std::fabs(forward.dot(worldUp)) > 0.99f) worldUp = {0, 0, 1};
        Vec3 right = forward.cross(worldUp).normalised();
        Vec3 up    = right.cross(forward).normalised();
        pendingCamera.camPos  = camPos;
        pendingCamera.forward = forward;
        pendingCamera.right   = right;
        pendingCamera.up      = up;
        pendingCamera.halfFov = std::tan(FOV * 0.5f);

        // Bounding sphere over all objects' interaction radii — lets the
        // per-step inner object loop early-out for rays nowhere near them.
        if (objects.empty()) {
            pendingCamera.objCenter  = {0,0,0};
            pendingCamera.objBoundR2 = 0.0f;
        } else {
            Vec3 c{0,0,0};
            for (auto& o : objects) c = c + o.pos;
            c = c * (1.0f / (float)objects.size());
            float maxR2 = 0.0f;
            for (auto& o : objects) {
                Vec3 d = o.pos - c;
                float d2 = d.dot(d);
                if (d2 > maxR2) maxR2 = d2;
            }
            constexpr float OBJ_INFLUENCE = OBJECT_RADIUS * 6.0f;
            float boundR = std::sqrt(maxR2) + OBJ_INFLUENCE;
            pendingCamera.objCenter  = c;
            pendingCamera.objBoundR2 = boundR * boundR;
        }

        requestedGeneration = renderGeneration.fetch_add(1, std::memory_order_relaxed) + 1;
        renderRequested = true;
    }
    renderCv.notify_one();
}

// Marks the user as actively interacting; pauses animation and switches the
// next render to coarse stride for snappy feedback.
static inline void noteInteraction() {
    interacting.store(true, std::memory_order_relaxed);
    lastInputTime = glfwGetTime();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Persistent worker pool — workers loop on a tile counter rather than being
//  spawned per frame. Same idea as a job-stealing render farm.
// ═══════════════════════════════════════════════════════════════════════════
static void poolWorker() {
    int lastSeenPass = 0;
    while (true) {
        int stride;
        CameraState camera{};
        uint64_t generation;
        PixelBuffer* buffer;

        {
            std::unique_lock<std::mutex> lock(poolMutex);
            poolCv.wait(lock, [&] {
                return shuttingDown || (poolPass != lastSeenPass);
            });
            if (shuttingDown) return;
            lastSeenPass = poolPass;
            stride     = poolStride;
            camera     = poolCamera;
            generation = poolGeneration;
            buffer     = poolBuffer;
        }

        // Process tiles until exhausted. Exit is governed solely by nextTile,
        // so every worker reaches the barrier below exactly once per pass.
        while (true) {
            int idx = nextTile.fetch_add(1, std::memory_order_relaxed);
            if (idx >= TILE_COUNT) break;
            if (renderGeneration.load(std::memory_order_relaxed) != generation) continue;
            int tx = idx % TILES_X;
            int ty = idx / TILES_X;
            renderTile(camera, *buffer, tx, ty, stride, generation);
        }

        // Barrier: last worker out wakes the coordinator. This guarantees
        // every worker has fully left the inner loop before runPass() returns,
        // so the next pass's state reset cannot race with stale workers.
        if (workersFinished.fetch_add(1, std::memory_order_acq_rel) == numWorkers - 1) {
            std::lock_guard<std::mutex> lock(poolMutex);
            poolDoneCv.notify_all();
        }
    }
}

static void runPass(const CameraState& camera,
                    PixelBuffer& buffer,
                    int stride,
                    uint64_t generation) {
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        poolCamera     = camera;
        poolGeneration = generation;
        poolStride     = stride;
        poolBuffer     = &buffer;
        nextTile.store(0, std::memory_order_relaxed);
        workersFinished.store(0, std::memory_order_relaxed);
        ++poolPass;
    }
    poolCv.notify_all();

    // Always wait for workers to finish — short-circuiting on shuttingDown
    // would let the coordinator return while workers are still writing into
    // localPixels, and destroying localPixels then frees their write target.
    std::unique_lock<std::mutex> lock(poolMutex);
    poolDoneCv.wait(lock, [&] {
        return workersFinished.load(std::memory_order_acquire) == numWorkers;
    });
}

static void renderCoordinator() {
    PixelBuffer localPixels(WIDTH * HEIGHT * 3, 0);   // hoisted: reused each frame
    while (true) {
        CameraState camera{};
        uint64_t generation = 0;

        {
            std::unique_lock<std::mutex> lock(renderMutex);
            renderCv.wait(lock, [] { return renderRequested || shuttingDown; });
            if (shuttingDown) return;

            camera = pendingCamera;
            generation = requestedGeneration;
            renderRequested = false;
        }


        // While the user is interacting, render at stride=4 (coarse, 1/16 the
        // work) for snappy feedback; idle frames render at stride=1 (fine).
        int stride = interacting.load(std::memory_order_relaxed) ? 4 : 1;
        runPass(camera, localPixels, stride, generation);

        if (renderGeneration.load(std::memory_order_relaxed) != generation) continue;

        {
            std::lock_guard<std::mutex> lock(renderMutex);
            stagingPixels.swap(localPixels);
            completedGeneration = generation;
            renderReady = true;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  GLFW callbacks
// ═══════════════════════════════════════════════════════════════════════════
static void key_callback(GLFWwindow* win, int key, int /*sc*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(win, GLFW_TRUE);
        return;
    }

    // Spacebar adds a new test particle to the scene.
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        spawnObject();
        requestRender();
        return;
    }

    float orbitSpeed = 0.08f;
    float zoomSpeed  = 2.0f;
    bool moved = false;

    switch (key) {
        case GLFW_KEY_A: case GLFW_KEY_LEFT:  camYaw   -= orbitSpeed; moved = true; break;
        case GLFW_KEY_D: case GLFW_KEY_RIGHT: camYaw   += orbitSpeed; moved = true; break;
        case GLFW_KEY_W: case GLFW_KEY_UP:    camPitch -= orbitSpeed; moved = true; break;
        case GLFW_KEY_S: case GLFW_KEY_DOWN:  camPitch += orbitSpeed; moved = true; break;
        case GLFW_KEY_Q:                      camDist  -= zoomSpeed;  moved = true; break;
        case GLFW_KEY_E:                      camDist  += zoomSpeed;  moved = true; break;
    }

    if (moved) {
        clampCamera();
        noteInteraction();
        requestRender();
    }
}

static void mouse_button_callback(GLFWwindow* win, int button, int action, int /*mods*/) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        mouseDown = true;
        glfwGetCursorPos(win, &lastMouseX, &lastMouseY);
    } else if (action == GLFW_RELEASE) {
        mouseDown = false;
    }
}

static void cursor_position_callback(GLFWwindow* /*win*/, double mx, double my) {
    if (!mouseDown) return;

    double dx = mx - lastMouseX;
    double dy = my - lastMouseY;
    lastMouseX = mx;
    lastMouseY = my;

    camYaw   += (float)(dx * 0.005);
    camPitch += (float)(dy * 0.005);

    clampCamera();
    noteInteraction();
    requestRender();
}

static void scroll_callback(GLFWwindow* /*win*/, double /*xoff*/, double yoff) {
    camDist -= (float)(yoff * 2.0);
    clampCamera();
    noteInteraction();
    requestRender();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════
static void drawFrame(GLFWwindow* win) {
    int fbw, fbh;
    glfwGetFramebufferSize(win, &fbw, &fbh);
    glViewport(0, 0, fbw, fbh);
    glClear(GL_COLOR_BUFFER_BIT);

    glPixelZoom((float)fbw / (float)WIDTH, -(float)fbh / (float)HEIGHT);
    glRasterPos2f(-1.0f, 1.0f);
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, displayPixels.data());

    glfwSwapBuffers(win);
}

int main() {
    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return 1;
    }

    GLFWwindow* win = glfwCreateWindow(WIDTH, HEIGHT, "CS184 Black Holes", nullptr, nullptr);
    if (!win) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    // Persistent worker pool.
    const unsigned threadCount = std::max(1u, std::thread::hardware_concurrency());
    numWorkers = (int)threadCount;
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (unsigned i = 0; i < threadCount; ++i) workers.emplace_back(poolWorker);

    std::thread coordinator(renderCoordinator);

    glfwSetKeyCallback(win, key_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetCursorPosCallback(win, cursor_position_callback);
    glfwSetScrollCallback(win, scroll_callback);

    requestRender();

    double lastTickTime  = glfwGetTime();
    double lastFrameClock = glfwGetTime();
    constexpr double TICK_INTERVAL = 1.0 / 60.0;   // continuous-animation rate

    while (!glfwWindowShouldClose(win)) {
        glfwWaitEventsTimeout(0.005);

        // Advance the animation clock only while idle — freezes disk/stars
        // during camera moves so all the work goes into spatial responsiveness.
        double now = glfwGetTime();
        double dt  = now - lastFrameClock;
        lastFrameClock = now;

        if (interacting.load(std::memory_order_relaxed) &&
            now - lastInputTime > INTERACT_LINGER) {
            interacting.store(false, std::memory_order_relaxed);
        }
        if (!interacting.load(std::memory_order_relaxed)) {
            animTime += (float)dt;
            stepObjects((float)dt);
        }

        // Continuous animation tick: only fire when no render is in flight.
        // Bumping the generation while a render is running would supersede
        // it and the coordinator would skip publication. User-input bumps
        // deliberately do supersede for snappy response, but the tick must not.
        if (now - lastTickTime >= TICK_INTERVAL) {
            bool inFlight;
            {
                std::lock_guard<std::mutex> lock(renderMutex);
                inFlight = (completedGeneration != requestedGeneration);
            }
            if (!inFlight) {
                lastTickTime = now;
                requestRender();
            }
        }

        {
            std::lock_guard<std::mutex> lock(renderMutex);
            if (renderReady) {
                displayPixels.swap(stagingPixels);
                renderReady = false;
                needsRedraw = true;
            }
        }

        if (needsRedraw) {
            needsRedraw = false;
            drawFrame(win);
        }
    }

    {
        std::lock_guard<std::mutex> lock(renderMutex);
        shuttingDown = true;
    }
    {
        std::lock_guard<std::mutex> lock(poolMutex);
    }
    renderCv.notify_all();
    poolCv.notify_all();
    poolDoneCv.notify_all();
    coordinator.join();
    for (auto& w : workers) w.join();

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
