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
constexpr int WIDTH  = 200;
constexpr int HEIGHT = 200;

// ── Pixel buffer ────────────────────────────────────────────────────────────
using PixelBuffer = std::vector<uint8_t>;

static PixelBuffer displayPixels(WIDTH * HEIGHT * 3, 0);
static PixelBuffer stagingPixels(WIDTH * HEIGHT * 3, 0);

inline void setPixel(PixelBuffer& buffer, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return;
    const int index = (y * WIDTH + x) * 3;
    buffer[index + 0] = r;
    buffer[index + 1] = g;
    buffer[index + 2] = b;
}

// ═══════════════════════════════════════════════════════════════════════════
//  3D Vector
// ═══════════════════════════════════════════════════════════════════════════
struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(double s)      const { return {x*s, y*s, z*s}; }
    double dot(const Vec3& v)     const { return x*v.x + y*v.y + z*v.z; }
    Vec3 cross(const Vec3& v)     const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    double length() const { return sqrt(x*x + y*y + z*z); }
    Vec3 normalised() const {
        double l = length();
        return l > 0 ? Vec3{x/l, y/l, z/l} : Vec3{0,0,0};
    }
};

struct CameraState {
    double dist;
    double yaw;
    double pitch;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Black hole parameters — tweak these!
// ═══════════════════════════════════════════════════════════════════════════
constexpr double BH_MASS = 1.0;
constexpr double RS      = 2.0 * BH_MASS;

constexpr double DISK_INNER = 3.0 * RS;
constexpr double DISK_OUTER = 12.0 * RS;
constexpr double DISK_HALF_THICKNESS = 0.15;

constexpr double FOV       = 1.2;
constexpr double STEP_SIZE = 0.08;
constexpr int    MAX_STEPS = 4000;
constexpr double MAX_DIST  = 80.0;

// ═══════════════════════════════════════════════════════════════════════════
//  Camera state — modified by input
// ═══════════════════════════════════════════════════════════════════════════
static double camDist      = 30.0;
static double camYaw       = 0.0;
static double camPitch     = 1.2;
static bool   needsRedraw  = true;

// ═══════════════════════════════════════════════════════════════════════════
//  Mouse state
// ═══════════════════════════════════════════════════════════════════════════
static bool   mouseDown  = false;
static double lastMouseX = 0.0;
static double lastMouseY = 0.0;

// ═══════════════════════════════════════════════════════════════════════════
//  Render threading
// ═══════════════════════════════════════════════════════════════════════════
static std::mutex              renderMutex;
static std::condition_variable renderCv;
static CameraState             pendingCamera{camDist, camYaw, camPitch};
static std::atomic<uint64_t>   renderGeneration{0};
static uint64_t                requestedGeneration = 0;
static uint64_t                completedGeneration = 0;
static bool                    renderRequested = false;
static bool                    renderReady = false;
static bool                    shuttingDown = false;

// ═══════════════════════════════════════════════════════════════════════════
//  Starfield
// ═══════════════════════════════════════════════════════════════════════════
static uint32_t hashStar(int a, int b) {
    uint32_t h = (uint32_t)(a * 374761393 + b * 668265263);
    h = (h ^ (h >> 13)) * 1274126177;
    return h ^ (h >> 16);
}

void starColour(const Vec3& dir, uint8_t& r, uint8_t& g, uint8_t& b) {
    double theta = atan2(dir.z, dir.x);
    double phi   = asin(fmax(-1.0, fmin(1.0, dir.y)));
    int gridX = (int)floor(theta * 80.0);
    int gridY = (int)floor(phi * 80.0);
    uint32_t h = hashStar(gridX, gridY);
    if ((h & 0xFF) < 8) {
        int brightness = 140 + (int)(h >> 8) % 116;
        int tint = (h >> 16) % 4;
        if (tint == 0)      { r = (uint8_t)brightness; g = (uint8_t)(brightness * 0.85); b = (uint8_t)(brightness * 0.7); }
        else if (tint == 1) { r = (uint8_t)(brightness * 0.7); g = (uint8_t)(brightness * 0.85); b = (uint8_t)brightness; }
        else                { r = (uint8_t)brightness; g = (uint8_t)brightness; b = (uint8_t)brightness; }
    } else {
        r = 2; g = 2; b = 5;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Accretion disk colour
// ═══════════════════════════════════════════════════════════════════════════
void diskColour(double radius, double angle, uint8_t& r, uint8_t& g, uint8_t& b) {
    double t = 1.0 - (radius - DISK_INNER) / (DISK_OUTER - DISK_INNER);
    t = fmax(0.0, fmin(1.0, t));
    double swirl = sin(angle * 6.0 + radius * 2.0) * 0.15 + 0.85;
    t *= swirl;
    if (t > 0.7) {
        r = 255;
        g = (uint8_t)(200 + 55 * ((t - 0.7) / 0.3));
        b = (uint8_t)(150 + 105 * ((t - 0.7) / 0.3));
    } else if (t > 0.3) {
        double s = (t - 0.3) / 0.4;
        r = (uint8_t)(100 + 155 * s);
        g = (uint8_t)(30 + 170 * s);
        b = (uint8_t)(5 + 145 * s);
    } else {
        double s = t / 0.3;
        r = (uint8_t)(100 * s);
        g = (uint8_t)(20 * s);
        b = (uint8_t)(5 * s);
    }
    double doppler = 0.6 + 0.4 * sin(angle + 1.0);
    r = (uint8_t)fmin(255.0, r * doppler);
    g = (uint8_t)fmin(255.0, g * doppler);
    b = (uint8_t)fmin(255.0, b * doppler);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Ray tracer
// ═══════════════════════════════════════════════════════════════════════════
void traceRay(Vec3 pos, Vec3 dir, uint8_t& r, uint8_t& g, uint8_t& b) {
    Vec3 hVec = pos.cross(dir);
    double h2 = hVec.dot(hVec);

    for (int step = 0; step < MAX_STEPS; ++step) {
        double dist2 = pos.dot(pos);
        double dist = sqrt(dist2);
        if (dist < RS) { r = 0; g = 0; b = 0; return; }
        if (dist > MAX_DIST) { starColour(dir.normalised(), r, g, b); return; }
        if (fabs(pos.y) < DISK_HALF_THICKNESS) {
            double diskR = sqrt(pos.x * pos.x + pos.z * pos.z);
            if (diskR > DISK_INNER && diskR < DISK_OUTER) {
                double angle = atan2(pos.z, pos.x);
                diskColour(diskR, angle, r, g, b);
                return;
            }
        }

        double invDist = 1.0 / dist;
        double invDist5 = invDist * invDist * invDist * invDist * invDist;
        double accelMag = -1.5 * RS * h2 * invDist5;
        Vec3 accel = pos * accelMag;
        dir = dir + accel * STEP_SIZE;
        pos = pos + dir * STEP_SIZE;
    }

    r = 0; g = 0; b = 0;
}

static void clampCamera() {
    const double PI = 3.14159265358979;
    if (camPitch < 0.1)       camPitch = 0.1;
    if (camPitch > PI - 0.1)  camPitch = PI - 0.1;
    if (camDist < RS * 2.5)   camDist  = RS * 2.5;
    if (camDist > 100.0)      camDist  = 100.0;
}

void renderRows(const CameraState& camera,
                PixelBuffer& buffer,
                int rowStart,
                int rowEnd,
                uint64_t generation) {
    Vec3 camPos = {
        camera.dist * sin(camera.pitch) * sin(camera.yaw),
        camera.dist * cos(camera.pitch),
        camera.dist * sin(camera.pitch) * cos(camera.yaw)
    };

    Vec3 forward = (Vec3{0,0,0} - camPos).normalised();
    Vec3 worldUp = {0, 1, 0};
    if (fabs(forward.dot(worldUp)) > 0.99) worldUp = {0, 0, 1};
    Vec3 right = forward.cross(worldUp).normalised();
    Vec3 up    = right.cross(forward).normalised();

    double halfFov = tan(FOV * 0.5);

    for (int py = rowStart; py < rowEnd; ++py) {
        if (renderGeneration.load(std::memory_order_relaxed) != generation) return;

        const double ny = (1.0 - 2.0 * (py + 0.5) / HEIGHT) * halfFov;
        for (int px = 0; px < WIDTH; ++px) {
            const double nx = (2.0 * (px + 0.5) / WIDTH - 1.0) * halfFov;
            Vec3 rayDir = (forward + right * nx + up * ny).normalised();

            uint8_t r, g, b;
            traceRay(camPos, rayDir, r, g, b);
            setPixel(buffer, px, py, r, g, b);
        }
    }
}

static void requestRender() {
    {
        std::lock_guard<std::mutex> lock(renderMutex);
        pendingCamera = {camDist, camYaw, camPitch};
        requestedGeneration = renderGeneration.fetch_add(1, std::memory_order_relaxed) + 1;
        renderRequested = true;
    }
    renderCv.notify_one();
}

static void renderWorker() {
    const unsigned threadCount = std::max(1u, std::thread::hardware_concurrency());

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
            renderReady = false;
        }

        PixelBuffer localPixels(WIDTH * HEIGHT * 3, 0);
        std::vector<std::thread> workers;
        workers.reserve(threadCount);

        const int rowsPerWorker = (HEIGHT + static_cast<int>(threadCount) - 1) / static_cast<int>(threadCount);
        for (unsigned i = 0; i < threadCount; ++i) {
            const int rowStart = static_cast<int>(i) * rowsPerWorker;
            const int rowEnd = std::min(HEIGHT, rowStart + rowsPerWorker);
            if (rowStart >= rowEnd) break;
            workers.emplace_back([&, rowStart, rowEnd] {
                renderRows(camera, localPixels, rowStart, rowEnd, generation);
            });
        }

        for (std::thread& worker : workers) {
            worker.join();
        }

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

    double orbitSpeed = 0.08;
    double zoomSpeed  = 2.0;
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

    camYaw   += dx * 0.005;
    camPitch += dy * 0.005;

    clampCamera();
    requestRender();
}

static void scroll_callback(GLFWwindow* /*win*/, double /*xoff*/, double yoff) {
    camDist -= yoff * 2.0;
    clampCamera();
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

    std::thread worker(renderWorker);

    glfwSetKeyCallback(win, key_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetCursorPosCallback(win, cursor_position_callback);
    glfwSetScrollCallback(win, scroll_callback);

    requestRender();

    while (!glfwWindowShouldClose(win)) {
        glfwWaitEventsTimeout(0.01);

        {
            std::lock_guard<std::mutex> lock(renderMutex);
            if (renderReady && completedGeneration == renderGeneration.load(std::memory_order_relaxed)) {
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
    renderCv.notify_one();
    worker.join();

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
