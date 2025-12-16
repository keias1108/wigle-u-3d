# wigle-u-3d - System Architecture

3D Energy Life Simulation using WebGPU

---

## 📐 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (HTML)                     │
│  Controls │ 3D Canvas │ FPS Display │ Speed/Grid Buttons     │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──> DOM Events (mouse, keyboard, sliders)
             │
┌────────────▼────────────────────────────────────────────────┐
│             WebGPUSimulation3D (Main Class)                  │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────┐ │
│  │  Params  │   GPU    │ Display  │  Camera  │  Frame     │ │
│  │  Control │  Compute │  Render  │  Control │  Loop      │ │
│  └──────────┴──────────┴──────────┴──────────┴────────────┘ │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼──────────┐ ┌───▼──────────────────┐
│   WebGPU     │ │   Shader Modules     │
│  Pipelines   │ │  (compute/render)    │
└───┬──────────┘ └───┬──────────────────┘
    │                │
    │      ┌─────────▼─────────┐
    │      │ Compute Shader    │
    │      │  (WGSL)           │
    │      └─────────┬─────────┘
    │                │
┌───▼────────────────▼──────────────┐
│   GPU (WebGPU Context)            │
│  3D Float Textures (Ping-Pong)    │
│  32³-256³ rgba16float             │
└───────────────────────────────────┘
```

---

## 🏗️ Core Components

### 1. **WebGPUSimulation3D** (`src/core/WebGPUSimulation3D.js`)

**Main orchestrator class** - manages entire simulation lifecycle.

**Responsibilities:**
- Initialize WebGPU device and pipelines
- Handle user interactions (mouse drag, WASD, sliders)
- Update display and FPS counter
- Manage simulation parameters
- Coordinate compute and render passes

**Key Data Flows:**
1. **Init**: DOM → WebGPU setup → Shader compilation → Start loop
2. **Animate Loop**: Compute × speed → Reduce average → Render → Update UI
3. **User Input**: Slider change → Update params → Write to GPU buffer

**State:**
- Grid size: 32³, 64³, 128³, or 256³
- Field textures: 2 ping-pong buffers (rgba16float)
- Parameters: 12 simulation parameters
- Camera: yaw, pitch, distance, offsetX, offsetY
- Speed: 0× (paused), 1×, 2×, 5×

---

### 2. **Compute Shader** (`src/shaders/compute.wgsl`)

**Heart of the simulation** - WGSL compute shader running on GPU.

**Executed:** gridSize³ times per frame (one per cell), in parallel

**Algorithm (per cell):**
```wgsl
1. Sample (2×KERNEL+1)³ neighbor cells (21³ = 9,261 for KERNEL=10)
2. Compute weighted potential from neighbors (spherical kernel)
3. Apply growth function (Gaussian bell curve)
4. Calculate metabolism (quadratic energy decay)
5. Add diffusion (3D Laplacian, 6-neighbor stencil)
6. Add fission noise (if energy > threshold)
7. Add random noise for organic behavior
8. Update energy and clamp to [0, 1]
```

**Parameters (from uniform buffer):**
- **Kernel**: innerRadius, innerStrength, outerRadius, outerStrength
- **Growth**: growthCenter, growthWidth, growthRate, suppressionFactor
- **Economy**: decayRate, diffusionRate, fissionThreshold, instabilityFactor
- **Global**: globalAverage (from reduction pipeline)

**Performance Critical:**
- For 32³ grid × 21³ kernel = **303 million iterations per frame**
- For 64³ grid × 21³ kernel = **2.4 billion iterations per frame**

---

### 3. **Render Shader** (`src/shaders/render.wgsl`)

**3D Volume Rendering** - Ray marching through energy field.

**Algorithm:**
```
1. Generate ray from orbital camera (yaw/pitch/distance)
2. Intersect ray with unit cube (AABB test)
3. March through volume (64 samples per ray)
4. Sample energy field at each step (trilinear interpolation)
5. Track maximum energy along ray
6. Map max energy to color gradient (purple/blue theme)
7. Output final pixel color
```

**Camera Model:**
- Orbital camera (yaw/pitch rotation, distance zoom)
- WASD panning (offsetX, offsetY)
- Center: (0.5 + offset, 0.5 + offset, 0.5)
- Ray origin: center - camDir × distance

**Color Gradient:**
- 0.0-0.1: Dark blue (almost black)
- 0.3-0.5: Medium blue/purple
- 0.7-0.85: Bright purple
- 0.85-1.0: Near white with sparkle effect

---

### 4. **Reduction Shader** (`src/shaders/reduce.wgsl`)

**Hierarchical Average Computation** - Multi-pass GPU reduction.

**Problem:** Need to compute average of gridSize³ values efficiently.

**Solution:** Hierarchical downsampling
```
32³ → 16³ → 8³ → 4³ → 2³ → 1³ (single value)
(field) (RT)  (RT) (RT) (RT) (final avg)
```

**Each pass:**
- Sample 2×2×2 block from input (8 cells)
- Average the values
- Write to 1 cell in output
- Result size = input size / 2

**Final:** Read single cell from CPU for global average
- Used in growth function as suppression factor
- Updated every 2 frames (throttled)

---

### 5. **Shader Loader** (`src/shaders/shaderLoader.js`)

**Shader Management** - Loads WGSL code with template substitution.

**Functions:**
- `getComputeShader(kernelSize)` - Replace `{{KERNEL_SIZE}}` template
- `getRenderShader()` - Load render shader
- `getReduceShader()` - Load reduction shader

**File Structure:**
- `.wgsl` files - Reference documentation (actual WGSL code)
- `.wgsl.js` files - ES module exports (for browser compatibility)

---

### 6. **Buffer Utilities** (`src/utils/bufferUtils.js`)

**Uniform Buffer Packing** - Type-safe GPU buffer creation.

**Key Function:**
```javascript
packSimParams(params, gridSize, camera)
  → ArrayBuffer (128 bytes, 7 vec4s)
```

**WGSL Struct Layout:**
```wgsl
struct SimParams {
  dims : vec4<u32>,     // gridSize×3, padding
  inner : vec4<f32>,    // kernel radii/strengths
  growthA : vec4<f32>,  // growth params
  economy : vec4<f32>,  // decay/diffusion/fission
  instab : vec4<f32>,   // instability, growthWidthNorm, inv×2
  misc : vec4<f32>,     // yaw, pitch, distance, seed
  camera : vec4<f32>,   // offsetX, offsetY, time, unused
};
```

**Benefit:** Replaces 50 lines of manual ArrayBuffer indexing with type-safe builder.

---

### 7. **Texture Utilities** (`src/utils/textureUtils.js`)

**3D Texture Operations** - GPU texture creation and seeding.

**Functions:**
- `createFieldTexture(device, size)` - Create rgba16float 3D texture
- `seedTexture(device, texture, size, maxEnergy)` - Fill with random energy
- `createReduceTexture(device, size)` - Create r32float reduction texture

**Texture Usage:**
- **Field Textures**: Ping-pong between two buffers (double buffering)
- **Reduction Textures**: Chain of textures for hierarchical average
- **Sampler**: Linear filtering, repeat wrapping (toroidal topology)

---

## 🔄 Data Flow

### **Main Render Loop** (60 FPS target)

```
requestAnimationFrame()
   │
   ├─> For each speed multiplier (1×, 2×, or 5×):
   │   ├─> #computePass()
   │   │    ├─> Create bind group (current → next texture)
   │   │    ├─> Dispatch compute shader
   │   │    │    └─> Execute on GPU (gridSize³ / 64 workgroups)
   │   │    └─> Swap ping-pong buffers (current ↔ next)
   │   │
   │   └─> Every 2 frames (throttled):
   │        └─> #reducePassAndRead()
   │             ├─> Multi-pass reduction (32³ → 1³)
   │             ├─> GPU readback (async buffer mapping)
   │             └─> Update globalAverage parameter
   │
   ├─> #updateCamera(dt)
   │    └─> WASD panning based on keyState
   │
   ├─> #writeParamsBuffer()
   │    └─> Pack params + camera → GPU uniform buffer
   │
   ├─> #renderPass()
   │    ├─> Create bind group (field texture + params)
   │    ├─> Fullscreen quad render
   │    └─> Ray marching in fragment shader
   │
   └─> Update UI (FPS counter every 1 second)
```

---

## 🎨 GPU Pipeline Architecture

### **Compute Pipeline** (Simulation Update)

```
Input: 3D texture (current state)
  ↓
Compute Shader (4×4×4 workgroups)
  ├─> Each thread = 1 cell
  ├─> 21³ = 9,261 neighbor samples
  ├─> Growth/decay/diffusion calculations
  └─> Write new energy value
  ↓
Output: 3D texture (next state)
  ↓
Swap buffers (ping-pong)
```

**Workgroup Size:** 4×4×4 = 64 threads
- Grid 32³ requires 8×8×8 = 512 workgroups
- Grid 64³ requires 16×16×16 = 4,096 workgroups

**Bind Group Layout:**
```
@binding(0): Uniform buffer (SimParams)
@binding(1): Input texture (current state)
@binding(2): Output texture (next state, storage)
```

---

### **Reduction Pipeline** (Global Average)

```
Field Texture (32³)
  ↓
Reduce Pass 1: 32³ → 16³ (average 2×2×2 blocks)
  ↓
Reduce Pass 2: 16³ → 8³
  ↓
Reduce Pass 3: 8³ → 4³
  ↓
Reduce Pass 4: 4³ → 2³
  ↓
Reduce Pass 5: 2³ → 1³ (single value)
  ↓
GPU Readback (buffer mapping, async)
  ↓
CPU: params.globalAverage = result
```

**Why Hierarchical?**
- Direct sum would require atomic operations (slower)
- Reduction is O(log₂(gridSize)) passes
- Each pass is parallel on GPU

---

### **Render Pipeline** (Visualization)

```
Fullscreen Quad (6 vertices, 2 triangles)
  ↓
Vertex Shader: Generate NDC positions + UVs
  ↓
Fragment Shader: For each pixel
  ├─> Calculate ray direction from camera
  ├─> AABB intersection test
  ├─> Ray march through volume (64 steps)
  ├─> Sample field texture (trilinear)
  ├─> Track maximum energy
  ├─> Map to color gradient
  └─> Output RGB color
  ↓
Canvas (screen)
```

**Bind Group Layout:**
```
@binding(0): Sampler (linear, repeat)
@binding(1): Field texture (3D)
@binding(2): Uniform buffer (SimParams - for camera)
```

---

## 🧮 Performance Characteristics

### **Computational Complexity**

**Per-Frame Cost:**
```
Total iterations = gridSize³ × (2×KERNEL+1)³
                 = 32³ × 21³ (default)
                 = 32,768 × 9,261
                 = 303,042,048 iterations/frame at 60 FPS
```

**Scaling Table:**

| Grid Size | Kernel | Workgroups | Iterations/Frame | Est. FPS  |
|-----------|--------|------------|------------------|-----------|
| 32³       | 10     | 512        | 303M             | 60+ FPS   |
| 64³       | 10     | 4,096      | 2.4B             | 30-45 FPS |
| 96³       | 10     | 13,824     | 8.2B             | 15-25 FPS |
| 128³      | 10     | 32,768     | 19.5B            | 10-15 FPS |

*Performance depends on GPU (tested on mid-range GPUs)*

---

### **GPU Bottlenecks**

1. **Memory Bandwidth** (Primary)
   - 9,261 texture samples per cell
   - At 32³ grid = 303M samples/frame
   - Bandwidth = samples × 8 bytes (rgba16float) = 2.4 GB/frame
   - At 60 FPS = **145 GB/s bandwidth required**

2. **Compute Occupancy**
   - Small workgroup size (64 threads) may underutilize some GPUs
   - Optimal size varies by architecture (64-256 threads)

3. **Register Pressure**
   - Triple-nested loop may limit parallelism
   - GPUs have limited registers per thread

---

### **Optimization Opportunities**

(See `docs/PERFORMANCE.md` for detailed optimization strategies)

1. **Kernel Weight Lookup Texture** - Replace exp() calculations
2. **Workgroup Size Tuning** - Test 8×8×4, 4×4×8 configurations
3. **Loop Bounds Optimization** - Iterate sphere instead of cube
4. **Shared Memory Caching** - Cache neighborhood in workgroup memory

---

## 📂 File Responsibilities

| File | Purpose | Lines | Key Exports |
|------|---------|-------|-------------|
| `core/WebGPUSimulation3D.js` | Main simulation class | 449 | `WebGPUSimulation3D` |
| `shaders/compute.wgsl.js` | Compute shader source | 176 | `default` (WGSL string) |
| `shaders/render.wgsl.js` | Render shader source | 165 | `default` (WGSL string) |
| `shaders/reduce.wgsl.js` | Reduction shader source | 42 | `default` (WGSL string) |
| `shaders/shaderLoader.js` | Shader loading | 47 | `getComputeShader()`, etc. |
| `utils/bufferUtils.js` | Uniform buffer packing | 157 | `packSimParams()`, `UniformBufferBuilder` |
| `utils/textureUtils.js` | 3D texture creation | 89 | `createFieldTexture()`, `seedTexture()` |
| `config/constants.js` | System constants | - | `KERNEL_SIZE`, `DEFAULT_GRID_SIZE`, etc. |
| `config/defaults.js` | Simulation parameters | - | `DEFAULT_PARAMS`, `PARAM_SPECS` |
| `main.js` | Entry point, UI | 285 | Instantiates simulation |

**Total:** ~1,400 lines (down from 814 in monolithic version, +modularization)

---

## 🎯 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **WebGPU instead of WebGL** | Better compute support, modern API, 3D textures |
| **32³-256³ grid sizes** | Balance between detail and performance |
| **rgba16float textures** | Precision for energy values (vs r32float for less memory) |
| **Ping-pong buffers** | WebGPU read/write limitation (no in-place updates) |
| **Inline shaders in .js** | No build step, browser-compatible ES modules |
| **4×4×4 workgroups** | Conservative size for compatibility (can be tuned) |
| **64 ray samples** | Quality/performance balance for ray marching |
| **Orbital camera** | Natural 3D viewing without gimbal lock |

---

## 🔮 Future Extensibility

### **Adding New Parameters:**
1. Add to `src/config/defaults.js` (`DEFAULT_PARAMS`)
2. Add to `PARAM_SPECS` array
3. Add slider to `index.html`
4. Update `packSimParams()` in `bufferUtils.js`
5. Update `SimParams` struct in shaders
6. Shader automatically picks up uniform

### **Adding New Visualization:**
1. Create new fragment shader in `src/shaders/`
2. Add loader function in `shaderLoader.js`
3. Add pipeline in `WebGPUSimulation3D.js`
4. Toggle via UI button

### **Multi-Species Simulation:**
1. Use G/B channels for different species
2. Update compute shader to handle 3-channel state
3. Modify color gradient to visualize species

---

## 🧪 Testing Recommendations

**Manual Tests:**
- Verify simulation runs at 60 FPS on 32³ grid
- Test all 12 parameter sliders update in real-time
- Verify camera controls (mouse drag, WASD, zoom)
- Test grid size changes (32³ → 64³ → 128³)
- Verify speed controls (0×, 1×, 2×, 5×)

**Visual Regression:**
- Compare output at frame 1000 with known good seed
- Verify energy conservation (average should stabilize)

**Performance Benchmarks:**
- Measure FPS at each grid size
- Profile GPU frame time using DevTools
- Check memory usage doesn't grow over time

---

## 📚 Key Concepts for LLMs

When modifying this codebase, understand:

1. **Ping-Pong Pattern**: Always read from one texture, write to another
2. **Uniform Buffer Alignment**: vec4 types must be 16-byte aligned
3. **3D Texture Coordinates**: Normalized [0, 1], wrapped for toroidal topology
4. **Workgroup Dispatch**: `(gridSize / 4) × 3` workgroups for 4×4×4 size
5. **Async GPU Readback**: Buffer mapping is asynchronous (Promise-based)
6. **Template Replacement**: `{{KERNEL_SIZE}}` is replaced at runtime

---

**For Detailed Information:**
- Simulation parameters → `docs/PARAMETERS.md`
- Performance tuning → `docs/PERFORMANCE.md`
- 2D reference (WebGL) → `wigle-u-2d/docs/`

---

*This architecture document provides a complete mental model for LLMs to understand and modify the codebase efficiently without scanning all implementation details.*
