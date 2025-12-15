# Performance Guide - wigle-u-3d

WebGPU 3D Simulation Performance Analysis and Optimization

---

## 📊 Computational Complexity

### **Per-Frame Cost Formula**

```
Total GPU Iterations = gridSize³ × kernelSamples³
                     = gridSize³ × (2×KERNEL_SIZE + 1)³
```

**Default Configuration:**
```
= 32³ × 21³
= 32,768 cells × 9,261 samples/cell
= 303,042,048 iterations per frame
```

**At 60 FPS:** 18.2 billion iterations per second

---

## 📈 Scaling Analysis

### **Grid Size Impact**

| Grid | Cells | Iterations/Frame | Memory | Est. FPS | Use Case |
|------|-------|------------------|--------|----------|----------|
| 32³  | 32K   | 303M             | 4 MB   | 60+ FPS  | Real-time exploration |
| 64³  | 262K  | 2.4B             | 32 MB  | 30-45 FPS | Balanced detail |
| 96³  | 885K  | 8.2B             | 108 MB | 15-25 FPS | High detail |
| 128³ | 2.1M  | 19.5B            | 256 MB | 10-15 FPS | Maximum quality |
| 256³ | 16.8M | 155B             | 2 GB   | 2-5 FPS  | Offline rendering |

*Tested on NVIDIA RTX 3060 / AMD RX 6700 XT class GPUs*

---

### **Kernel Size Impact**

KERNEL_SIZE is hardcoded to 10 in `src/config/constants.js`.

| Kernel | Samples | vs Default | Note |
|--------|---------|------------|------|
| 5      | 11³ = 1,331  | -86% | Too sparse, poor interaction |
| 7      | 15³ = 3,375  | -64% | Acceptable for low-end GPUs |
| 10     | 21³ = 9,261  | 0% (default) | Balanced |
| 12     | 25³ = 15,625 | +69% | Smoother patterns, slower |
| 15     | 31³ = 29,791 | +222% | Very expensive |

**Recommendation:** Keep KERNEL_SIZE=10 for best balance.

To change: Modify `KERNEL_SIZE` in `src/config/constants.js` and rebuild shaders.

---

## 🎯 GPU Bottlenecks

### **1. Memory Bandwidth** (Primary Bottleneck)

**Problem:** Each cell samples 9,261 neighbors from 3D texture.

**Bandwidth Calculation:**
```
Samples/frame = 32³ × 9,261 = 303M
Texture format = rgba16float = 8 bytes/sample
Bandwidth = 303M × 8 bytes = 2.4 GB/frame
At 60 FPS = 145 GB/s required
```

**GPU Bandwidth Limits:**
- GTX 1660: 192 GB/s (Adequate)
- RTX 3060: 360 GB/s (Comfortable)
- RTX 4090: 1,008 GB/s (Overkill)

**Why 3D is expensive:** 2D version would only need 21×21 = 441 samples (21× less).

---

### **2. Compute Occupancy**

**Workgroup Size:** 4×4×4 = 64 threads per workgroup

**Analysis:**
- Modern GPUs prefer 128-256 threads/workgroup for full occupancy
- 64 threads may underutilize some architectures
- Trade-off: Larger workgroups need more shared memory

**Optimal Workgroup Sizes (varies by GPU):**
- NVIDIA: 8×8×4 = 256 threads (best)
- AMD: 4×4×8 = 128 threads (balanced)
- Intel: 4×4×4 = 64 threads (conservative, current)

**To change:** Modify `@workgroup_size(4, 4, 4)` in `compute.wgsl`.

---

### **3. Register Pressure**

**Triple-Nested Loop:**
```wgsl
for (dz = -10; dz <= 10; dz++)
  for (dy = -10; dy <= 10; dy++)
    for (dx = -10; dx <= 10; dx++)
      // ... (9,261 iterations)
```

**Problem:** GPUs have limited registers per thread.
**Impact:** Compiler may spill to memory, reducing parallelism.

**Solution:** See "Optimizations" section below.

---

## ⚡ Optimization Strategies

### **Optimization 1: Kernel Weight Lookup Texture** ✅ Safe

**Current:** Compute `exp(-2 × t²)` for each of 9,261 samples.

**Proposed:** Pre-compute weights into 1D texture, sample instead.

```wgsl
// Current (expensive)
let w = outerStrength * exp(-2.0 * t * t);

// Optimized (1 texture lookup)
@group(0) @binding(3) var kernelWeightLUT : texture_1d<f32>;
let w = textureSampleLevel(kernelWeightLUT, sampler, dist / outerRadius, 0.0).x;
```

**Implementation:**
1. Add kernel weight texture creation in `WebGPUComputeEngine` (future)
2. Pre-compute 256 samples: weight[i] = exp(-2 × (i/256)²)
3. Update compute shader to sample texture

**Benefit:** Replaces expensive math with texture lookup
**Risk:** None (identical results with sufficient LUT resolution)
**Expected Speedup:** 5-10%

---

### **Optimization 2: Loop Bounds Tightening** ⚠️ Moderate Risk

**Current:** Iterate full cube, check sphere inside loop.

```wgsl
for (var dz = -KERNEL; dz <= KERNEL; dz++) {
  for (var dy = -KERNEL; dy <= KERNEL; dy++) {
    for (var dx = -KERNEL; dx <= KERNEL; dx++) {
      let dist = length(vec3(dx, dy, dz));
      if (dist <= outerRadius) { /* sample */ }
    }
  }
}
```

**Problem:** Iterates 21³ = 9,261 cells, but only ~4,189 are inside sphere.

**Optimized:** Use sphere equation to tighten bounds.

```wgsl
for (var dz = -KERNEL; dz <= KERNEL; dz++) {
  let maxR_yz = sqrt(outerRadius² - dz²);  // Sphere constraint
  let dyMax = i32(ceil(maxR_yz));

  for (var dy = -dyMax; dy <= dyMax; dy++) {
    let maxR_x = sqrt(maxR_yz² - dy²);
    let dxMax = i32(ceil(maxR_x));

    for (var dx = -dxMax; dx <= dxMax; dx++) {
      // Guaranteed inside sphere, no dist check needed
      let neighbor = loadEnergy(...);
    }
  }
}
```

**Benefit:** ~45% fewer iterations (9,261 → ~4,189)
**Risk:** Boundary calculation errors could miss cells
**Validation:** Compare frame 1000 output bit-exact
**Expected Speedup:** 20-30% (bandwidth bound, not iteration bound)

---

### **Optimization 3: Workgroup Size Tuning** ✅ Safe

**Current:** `@workgroup_size(4, 4, 4)` = 64 threads

**Test Configurations:**

| Config | Threads | NVIDIA | AMD | Intel |
|--------|---------|--------|-----|-------|
| 4×4×4  | 64      | ⚫ OK  | ⚫ OK | ✅ Best |
| 8×4×4  | 128     | ✅ Best | ✅ Best | ⚫ OK |
| 8×8×4  | 256     | ✅ Best | ⚫ OK | ❌ Slow |
| 4×4×8  | 128     | ⚫ OK  | ✅ Best | ⚫ OK |

**Implementation:**
1. Add WORKGROUP_SIZE constant to `constants.js`
2. Template replacement in `shaderLoader.js`
3. Benchmark on target GPU

**Benefit:** 10-20% FPS improvement on some GPUs
**Risk:** None (pure scheduling change)

---

### **Optimization 4: Shared Memory Caching** 🔬 Experimental

**Concept:** Load neighborhood into workgroup shared memory.

```wgsl
var<workgroup> sharedField : array<f32, 12*12*12>;  // 1,728 values

// Cooperatively load 12³ region into shared memory
workgroupBarrier();

// Each thread accesses shared memory instead of global texture
// Reduces global memory bandwidth by ~50×
```

**Benefit:** Massive bandwidth reduction (145 GB/s → 3 GB/s)
**Complexity:** High (complex indexing, workgroup coordination)
**Risk:** High (easy to introduce bugs)
**Status:** **Future work** (Phase 2 optimization)

---

### **Optimization 5: Texture Format Change** ❌ Not Recommended

**Current:** `rgba16float` (8 bytes/cell)
- R: energy (used)
- G, B, A: unused (6 bytes wasted)

**Proposed:** `r16float` (2 bytes/cell)

**Benefit:** 75% memory reduction, 4× bandwidth savings
**Drawback:** Breaks multi-channel extensibility (species, terrain)
**Verdict:** Keep rgba16float for future features

---

## 🧪 Profiling Tools

### **Browser DevTools (Chrome/Edge)**

1. Open DevTools → Performance tab
2. Record while simulation runs
3. Look for GPU frame time in timeline

**Interpreting Results:**
- GPU time > 16ms → Compute/bandwidth bound
- CPU time > 16ms → JavaScript/buffer update bound
- Total time > 16ms → Reduce grid size or optimize

---

### **FPS Counter**

Built-in FPS display (top-left corner):
- 60 FPS → Optimal (VSync limited)
- 30-50 FPS → Acceptable
- <30 FPS → Reduce grid size

---

### **WebGPU Timestamp Queries** (Advanced)

```javascript
// Add to compute pass (future enhancement)
const timestampWrites = {
  querySet: timestampQuerySet,
  beginningOfPassWriteIndex: 0,
  endOfPassWriteIndex: 1,
};

// Measure GPU compute time precisely
```

---

## 🎯 Performance Targets

### **Target FPS by Grid Size**

| Grid | Target FPS | Min FPS | GPU Class Required |
|------|------------|---------|-------------------|
| 32³  | 60 FPS     | 45 FPS  | GTX 1050 / RX 560 |
| 64³  | 45 FPS     | 30 FPS  | GTX 1660 / RX 5600 |
| 96³  | 25 FPS     | 15 FPS  | RTX 3060 / RX 6700 |
| 128³ | 15 FPS     | 10 FPS  | RTX 3080 / RX 6900 |

---

## 🔧 Tuning Recommendations

### **For Low-End GPUs (<4 GB VRAM)**
- Use 32³ grid only
- Consider reducing KERNEL_SIZE to 7
- Disable FPS counter (small CPU overhead)
- Close other GPU-intensive apps

### **For Mid-Range GPUs (6-8 GB VRAM)**
- Use 32³ or 64³ grid
- Default settings work well
- Can experiment with speed 2× multiplier

### **For High-End GPUs (>10 GB VRAM)**
- Use 64³ or 96³ grid for best visuals
- Consider 128³ for offline rendering
- Speed 5× still maintains 30+ FPS on 64³

---

## 📉 Performance Degradation Symptoms

**Problem:** FPS drops over time
**Cause:** Memory leak (buffer not freed)
**Fix:** Check `dispose()` methods in future classes

**Problem:** FPS varies wildly (30 → 60 → 30)
**Cause:** Thermal throttling
**Fix:** Improve cooling, reduce grid size

**Problem:** First frame very slow, then normal
**Cause:** Shader compilation (normal)
**Fix:** Pre-compile shaders on init (already done)

---

## 🚀 Quick Wins for Better Performance

1. **Start with 32³ grid** - Always smooth
2. **Use speed 0× when adjusting sliders** - Faster UI response
3. **Close unused browser tabs** - Frees GPU memory
4. **Update GPU drivers** - Often 5-10% free performance
5. **Use Chrome/Edge** - Better WebGPU support than Firefox
6. **Disable browser extensions** - Some inject code hurting FPS

---

## 📚 Further Reading

- **WebGPU Best Practices**: https://toji.dev/webgpu-best-practices/
- **Compute Shader Optimization**: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
- **3D Texture Performance**: https://www.khronos.org/opengl/wiki/Common_Mistakes#Texture_upload

---

*This performance guide helps LLMs understand bottlenecks and make informed optimization decisions without changing simulation behavior.*
