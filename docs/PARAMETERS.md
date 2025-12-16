# Parameters Guide - wigle-u-3d

Complete guide to all simulation parameters and their effects.

---

## 🎛️ Dynamic Tension (Kernel Weights)

Controls how cells attract and repel each other based on distance.

### **Inner Radius** (`innerRadius`)
- **Range:** 1.0 - 10.0
- **Default:** 3.5
- **Effect:** Size of attraction zone around each cell
- **3D Note:** Creates spherical attraction field
- **Visual:** Larger = cells cluster more loosely

### **Inner Strength** (`innerStrength`)
- **Range:** 0.0 - 2.0
- **Default:** 0.9
- **Effect:** How strongly cells pull each other together
- **Visual:** Higher = tighter clusters, more cohesion

### **Outer Radius** (`outerRadius`)
- **Range:** 5.0 - 15.0
- **Default:** 7.5
- **Effect:** Size of repulsion zone (must be > innerRadius)
- **3D Note:** Sphere radius for negative weight
- **Visual:** Larger = cells push apart from farther away

### **Outer Strength** (`outerStrength`)
- **Range:** -2.0 - 0.0 (negative = repulsion)
- **Default:** -0.4
- **Effect:** How strongly cells push apart
- **Formula:** Gaussian falloff: `exp(-2 × t²)`
- **Visual:** More negative = stronger separation

**💡 3D Kernel Scaling:**
- KERNEL_SIZE = 10 → Sample radius of 10 cells
- Total samples per cell: (2×10+1)³ = **21³ = 9,261 samples**
- Changing inner/outer radius affects sampling pattern
- Performance impact: Larger radius = more GPU memory bandwidth

---

## 💰 Energy Economy

Controls energy lifecycle: growth, decay, diffusion, and instability.

### **Growth Center** (`growthCenter`)
- **Range:** -2.0 - 2.0
- **Default:** -0.17
- **Effect:** Optimal neighbor energy for growth
- **Formula:** Gaussian bell curve centered here: `exp(-((potential - center) / width)² × 0.5)`
- **Visual:** Cells grow best when neighbors match this value
- **Tuning:** Negative values favor sparse patterns

### **Growth Width** (`growthWidth`)
- **Range:** 0.0001 - 1.0
- **Default:** 0.0183
- **Effect:** Tolerance around optimal value (σ in Gaussian)
- **Visual:**
  - **Narrow (<0.02):** Very sensitive, sharp patterns
  - **Wide (>0.1):** Tolerant, smooth evolution
- **Tip:** Lower values create more distinct structures

### **3D Width Norm** (`growthWidthNorm`)
- **Range:** 0.0 - 4.0
- **Default:** 1.5
- **Effect:** 3D에서 커널 이웃 수 증가로 `potential` 분산이 줄어드는 현상을 보정해, `growthWidth`가 다시 “자르는” 역할을 하도록 스케일링
- **Mechanics:** 내부적으로 `growthWidth_eff = growthWidth × (kernelScale^growthWidthNorm)` 형태로 적용 (0이면 비활성)

### **Growth Rate** (`growthRate`)
- **Range:** 0.001 - 1.0
- **Default:** 0.607
- **Effect:** Speed multiplier for energy change
- **Formula:** `deltaEnergy = growthRate × growth - metabolism + diffusion`
- **Visual:** Higher = faster evolution, more dynamic

### **Suppression Factor** (`suppressionFactor`)
- **Range:** 0.0 - 2.0
- **Default:** 1.0
- **Effect:** Reduces growth when globalAverage is high
- **Purpose:** Prevents overpopulation, maintains balance
- **Formula:** `growth = growth - globalAverage × suppressionFactor`

---

### **Decay Rate** (`decayRate`)
- **Range:** 0.0 - 1.0
- **Default:** 0.378
- **Effect:** Quadratic energy loss per frame (metabolism)
- **Formula:** `loss = energy² × decayRate`
- **Visual:** Higher = energy dissipates faster, shorter-lived patterns
- **Balance:** Competes with growth rate

### **Diffusion Rate** (`diffusionRate`)
- **Range:** 0.0 - 1.0
- **Default:** 0.333
- **Effect:** How fast energy spreads to neighbors
- **3D Formula:** 3D Laplacian (6-neighbor stencil):
  ```
  Δenergy = (E_x+ + E_x- + E_y+ + E_y- + E_z+ + E_z-) - 6×E_current
  diffusion = Δenergy × diffusionRate
  ```
- **Visual:** Higher = energy "bleeds" more, softer edges
- **3D Note:** 내부에서 CFL 안전계수(1/6)가 곱어져 적용됨. 3D는 이웃 수가 많아 과결합되기 쉬우니 0.6 이상은 신중히 사용.

### **Fission Threshold** (`fissionThreshold`)
- **Range:** 0.5 - 0.95
- **Default:** 0.796
- **Effect:** Energy level triggering chaos/instability
- **Visual:** Lower = more chaotic behavior, splitting patterns
- **Mechanics:** Above threshold, noise is added and growth function is reduced

### **Instability Factor** (`instabilityFactor`)
- **Range:** 0.0 - 3.0
- **Default:** 1.5
- **Effect:** Chaos strength above fission threshold
- **Formula:** If `energy > fissionThreshold`:
  ```
  excess = (energy - threshold) / (1.0 - threshold)
  growth = growth - excess × instabilityFactor
  noise = sin(position + time) × excess × 0.1
  ```
- **Purpose:** Breaks up high-energy concentrations

---

## 🔧 System Constants (Not User-Editable)

Fixed values in `src/config/constants.js`.

| Constant | Value | Purpose |
|----------|-------|---------|
| `KERNEL_SIZE` | 10 | Neighbor sampling radius |
| `DEFAULT_GRID_SIZE` | 32 | Default grid resolution (32×32×32) |
| `GRID_SIZE_OPTIONS` | [32, 64, 96, 128, 256] | Available grid sizes |
| `SEED_ENERGY_MAX` | 0.05 | Random initialization max |
| `GLOBAL_AVG_INTERVAL` | 2 | Compute average every N frames |
| `INITIAL_DISTANCE` | 2.5 | Default camera distance |
| `ROTATE_SENSITIVITY` | 0.3 | Mouse rotation sensitivity |
| `PAN_SPEED` | 0.5 | WASD panning speed |

---

## 🎯 Parameter Interactions

### **Most Sensitive Combinations:**

1. **growthWidth × growthCenter**
   - Defines "sweet spot" for life
   - Narrow width + specific center = fragile equilibrium
   - Example: width=0.0183, center=-0.17 (default) is well-tuned

2. **innerStrength vs. outerStrength ratio**
   - Balance determines cluster size
   - `|inner| / |outer| > 2`: Tight clusters
   - `|inner| / |outer| < 1`: Loose networks
   - Default: 0.9 / 0.4 = 2.25 (moderate clustering)

3. **decayRate × diffusionRate**
   - Decay > diffusion: Localized hotspots
   - Diffusion > decay: Spreading clouds
   - Default: 0.378 vs 0.333 (slightly localized)

4. **growthRate × suppressionFactor**
   - Controls population dynamics
   - High growth + low suppression = exponential growth
   - Moderate growth + high suppression = stable population

---

## 📊 Tuning Workflows

### **For Stable, Organized Patterns:**
1. Start with defaults
2. Adjust `growthWidth` slowly (±0.001)
3. Fine-tune `growthCenter` (±0.01)
4. Increase `suppressionFactor` if overpopulated
5. Tweak strength ratios last

### **For Dynamic, Chaotic Behavior:**
1. Lower `fissionThreshold` (0.7)
2. Increase `instabilityFactor` (2.0)
3. Increase `growthRate` (0.8)
4. Reduce `decayRate` for longer life (0.2)

### **For Smooth, Aesthetic Visuals:**
1. Increase `diffusionRate` (0.5+)
2. Lower `growthRate` (0.3-0.5)
3. Use wider `growthWidth` (0.05+)
4. Moderate fission threshold (0.85)

### **For Sparse, Crystalline Structures:**
1. Narrow `growthWidth` (0.01)
2. Negative `growthCenter` (-0.3)
3. Low `diffusionRate` (0.1)
4. High `decayRate` (0.5)

---

## 🎨 3D-Specific Notes

### **Kernel Sampling in 3D**

Unlike 2D (21×21 = 441 samples), 3D kernel is **21³ = 9,261 samples per cell**.

**Implications:**
- 21× more memory bandwidth than 2D
- Performance scales cubically with kernel radius
- Each +1 to KERNEL_SIZE adds ~(2k+1)³ - (2k-1)³ samples

**Example:**
- KERNEL=9 → 19³ = 6,859 samples
- KERNEL=10 → 21³ = 9,261 samples (+35%)
- KERNEL=11 → 23³ = 12,167 samples (+31%)

### **Grid Size Impact**

3D grid scales cubically:

| Grid Size | Cells | Compute Cost | Est. FPS | Memory |
|-----------|-------|--------------|----------|--------|
| 32³       | 32,768 | 303M iter/frame | 60+ FPS | ~4 MB |
| 64³       | 262,144 | 2.4B iter/frame | 30-45 FPS | ~32 MB |
| 96³       | 884,736 | 8.2B iter/frame | 15-25 FPS | ~108 MB |
| 128³      | 2,097,152 | 19.5B iter/frame | 10-15 FPS | ~256 MB |

*Memory = 2 textures × gridSize³ × 4 channels × 2 bytes (rgba16float)*

### **Camera Parameters (Not in UI)**

Controlled via mouse/keyboard:

- **Yaw**: Horizontal rotation (mouse drag X)
- **Pitch**: Vertical rotation (mouse drag Y)
- **Distance**: Zoom (mouse wheel)
- **Offset X/Y**: Panning (WASD keys)

---

## 🎭 Example Configurations

### **Default (Balanced)**
```javascript
{
  innerRadius: 3.5,
  innerStrength: 0.9,
  outerRadius: 7.5,
  outerStrength: -0.4,
  growthCenter: -0.17,
  growthWidth: 0.0183,
  growthRate: 0.607,
  suppressionFactor: 1.0,
  decayRate: 0.378,
  diffusionRate: 0.333,
  fissionThreshold: 0.796,
  instabilityFactor: 1.5
}
```
**Effect:** Stable, self-organizing 3D patterns

### **High Energy Chaos**
```javascript
{
  ...default,
  fissionThreshold: 0.6,
  instabilityFactor: 2.0,
  growthRate: 0.8
}
```
**Effect:** Turbulent, constantly changing structures

### **Crystalline Sparse**
```javascript
{
  ...default,
  growthWidth: 0.01,
  growthCenter: -0.3,
  diffusionRate: 0.1,
  decayRate: 0.5
}
```
**Effect:** Sharp, isolated crystalline formations

### **Smooth Organic**
```javascript
{
  ...default,
  diffusionRate: 0.6,
  growthWidth: 0.05,
  growthRate: 0.4
}
```
**Effect:** Flowing, blob-like structures

---

## 🔬 Parameter Validation

**Constraints enforced by UI:**
- All ranges enforced by slider min/max
- Step sizes: 0.001 for precise parameters, 0.1 for coarse ones
- Real-time updates: Changes apply immediately (no "Apply" button)

**Unsafe Combinations:**
- `outerRadius <= innerRadius`: No repulsion zone (can cause instability)
- `growthWidth < 0.001`: Numerical instability
- `growthRate > 1.0`: Energy explosion
- `decayRate = 0` + `diffusionRate = 0`: No energy dissipation

---

## 📝 Quick Reference Card

| Want | Adjust | Direction |
|------|--------|-----------|
| Bigger clusters | `innerRadius` | ↑ |
| Tighter packing | `innerStrength` | ↑ |
| More separation | `outerStrength` | ↓ (more negative) |
| Faster evolution | `growthRate` | ↑ |
| Longer-lived patterns | `decayRate` | ↓ |
| Softer edges | `diffusionRate` | ↑ |
| More chaos | `fissionThreshold` | ↓ |
| Sharp patterns | `growthWidth` | ↓ |
| Population control | `suppressionFactor` | ↑ |

---

**Pro Tip:** Save interesting configurations using the browser's localStorage (Save/Load buttons in UI). Parameters persist across sessions.

**For Performance Tuning:** See `docs/PERFORMANCE.md`

---

*This guide provides complete parameter understanding for LLMs to make informed adjustments without trial-and-error.*
