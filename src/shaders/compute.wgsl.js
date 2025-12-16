/**
 * 3D Energy Life Simulation - Compute Shader
 *
 * Exported as JavaScript string for use without build tools.
 * Reference WGSL file: compute.wgsl (documentation)
 */

export default /* wgsl */ `// 3D Energy Life Simulation - Compute Shader
// This shader implements a 3D cellular automaton with energy dynamics
//
// Each cell computes:
// 1. Weighted potential from neighbors (spherical kernel)
// 2. Growth function (Gaussian bell curve)
// 3. Energy metabolism (quadratic decay)
// 4. Diffusion (Laplacian smoothing)
// 5. Fission instability (chaos at high energy)
// 6. Random noise for organic behavior

struct SimParams {
  dims : vec4<u32>,
  inner : vec4<f32>,
  growthA : vec4<f32>,
  economy : vec4<f32>,
  instab : vec4<f32>,
  misc : vec4<f32>,
  camera : vec4<f32>,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var inputTex : texture_3d<f32>;
@group(0) @binding(2) var outputTex : texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var kernelWeightLUT : texture_1d<f32>;

// KERNEL_SIZE will be replaced by template (e.g., 10)
const KERNEL : i32 = {{KERNEL_SIZE}};

// Wrap coordinate for toroidal topology
fn wrapCoord(coord : i32, dim : i32) -> i32 {
  var v = coord % dim;
  if (v < 0) { v += dim; }
  return v;
}

// Load energy value with wrapped coordinates
fn loadEnergy(coord : vec3<i32>, dims : vec3<i32>) -> f32 {
  let wrapped = vec3<i32>(
    wrapCoord(coord.x, dims.x),
    wrapCoord(coord.y, dims.y),
    wrapCoord(coord.z, dims.z)
  );
  return textureLoad(inputTex, wrapped, 0).x;
}

// Compute kernel weight for neighbor interaction
// Inner zone: attraction (positive weight)
// Outer zone: repulsion (negative weight, Gaussian falloff)
fn kernelWeight(dist : f32, innerRadius : f32, innerStrength : f32, outerRadius : f32, outerStrength : f32) -> f32 {
  var weight = 0.0;
  if (dist < innerRadius) {
    let t = 1.0 - (dist / innerRadius);
    weight = weight + innerStrength * t * t;
  }
  let ringStart = innerRadius + 1.0;
  let ringEnd = outerRadius;
  if (dist > ringStart && dist < ringEnd) {
    let t = (dist - ringStart) / (ringEnd - ringStart);
    // Optimized: texture lookup replaces exp() (5-10% speedup)
    let lutIndex = i32(t * 255.0);
    let lutValue = textureLoad(kernelWeightLUT, lutIndex, 0).x;
    weight = weight + outerStrength * lutValue;
  }
  return weight;
}

// Growth function: Gaussian bell curve centered around optimal potential
// Includes fission instability at high energy levels
fn growthFunction(potential : f32, currentEnergy : f32, center : f32, width : f32, threshold : f32, instability : f32) -> f32 {
  let x = (potential - center) / width;
  var bell = exp(-x * x * 0.5);
  if (currentEnergy > threshold) {
    let excess = (currentEnergy - threshold) / (1.0 - threshold);
    bell = bell - excess * instability;
  }
  return bell;
}

// 3D Laplacian for diffusion (6/18/26-neighbor stencil)
// neighborMode: 6=face, 18=face+edge, 26=full cube
fn laplacian(coord : vec3<i32>, dims : vec3<i32>, current : f32, neighborMode : f32) -> f32 {
  // Face neighbors (6) - always used
  let xp = loadEnergy(coord + vec3<i32>(1, 0, 0), dims);
  let xm = loadEnergy(coord + vec3<i32>(-1, 0, 0), dims);
  let yp = loadEnergy(coord + vec3<i32>(0, 1, 0), dims);
  let ym = loadEnergy(coord + vec3<i32>(0, -1, 0), dims);
  let zp = loadEnergy(coord + vec3<i32>(0, 0, 1), dims);
  let zm = loadEnergy(coord + vec3<i32>(0, 0, -1), dims);

  var sum = xp + xm + yp + ym + zp + zm;
  var centerWeight = 6.0;

  // Edge neighbors (12) - weight 1/sqrt(2) = 0.7071
  if (neighborMode >= 18.0) {
    let w = 0.7071;
    sum += w * (
      loadEnergy(coord + vec3<i32>(1, 1, 0), dims) +
      loadEnergy(coord + vec3<i32>(1, -1, 0), dims) +
      loadEnergy(coord + vec3<i32>(-1, 1, 0), dims) +
      loadEnergy(coord + vec3<i32>(-1, -1, 0), dims) +
      loadEnergy(coord + vec3<i32>(1, 0, 1), dims) +
      loadEnergy(coord + vec3<i32>(1, 0, -1), dims) +
      loadEnergy(coord + vec3<i32>(-1, 0, 1), dims) +
      loadEnergy(coord + vec3<i32>(-1, 0, -1), dims) +
      loadEnergy(coord + vec3<i32>(0, 1, 1), dims) +
      loadEnergy(coord + vec3<i32>(0, 1, -1), dims) +
      loadEnergy(coord + vec3<i32>(0, -1, 1), dims) +
      loadEnergy(coord + vec3<i32>(0, -1, -1), dims)
    );
    centerWeight += 12.0 * w;
  }

  // Corner neighbors (8) - weight 1/sqrt(3) = 0.5774
  if (neighborMode >= 26.0) {
    let w = 0.5774;
    sum += w * (
      loadEnergy(coord + vec3<i32>(1, 1, 1), dims) +
      loadEnergy(coord + vec3<i32>(1, 1, -1), dims) +
      loadEnergy(coord + vec3<i32>(1, -1, 1), dims) +
      loadEnergy(coord + vec3<i32>(1, -1, -1), dims) +
      loadEnergy(coord + vec3<i32>(-1, 1, 1), dims) +
      loadEnergy(coord + vec3<i32>(-1, 1, -1), dims) +
      loadEnergy(coord + vec3<i32>(-1, -1, 1), dims) +
      loadEnergy(coord + vec3<i32>(-1, -1, -1), dims)
    );
    centerWeight += 8.0 * w;
  }

  return sum - centerWeight * current;
}

// 3D hash function for deterministic noise
fn hash31(p : vec3<u32>) -> f32 {
  var h = p.x * 0x1e35a7bdu + p.y * 0x94d049bbu + p.z * 0x5bd1e995u;
  h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
  h = h ^ (h >> 12u);
  return f32(h & 0x007fffffu) / f32(0x00800000u);
}

// Main compute shader entry point
// Workgroup size configured via template (e.g., 8×8×4 = 256 threads for NVIDIA)
@compute @workgroup_size({{WORKGROUP_X}}, {{WORKGROUP_Y}}, {{WORKGROUP_Z}})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = vec3<i32>(i32(params.dims.x), i32(params.dims.y), i32(params.dims.z));
  if (any(gid >= params.dims.xyz)) {
    return;
  }
  let coord = vec3<i32>(gid);

  // Extract parameters from uniform buffer
  let innerRadius = params.inner.x;
  let innerStrength = params.inner.y;
  let outerRadius = params.inner.z;
  let outerStrength = params.inner.w;
  let growthCenter = params.growthA.x;
  let growthWidth = params.growthA.y;
  let growthRate = params.growthA.z;
  let suppressionFactor = params.growthA.w;
  let globalAverage = params.economy.x;
  let decayRate = params.economy.y;
  let diffusionRate = params.economy.z;
  let fissionThreshold = params.economy.w;
  let instability = params.instab.x;
  let growthWidthNorm = params.instab.y;
  let neighborMode = params.instab.z;
  let time = params.camera.z;
  let seed = params.misc.w;

  let currentEnergy = textureLoad(inputTex, coord, 0).x;

  // 1. Compute weighted potential from spherical kernel
  // WARNING: This is the performance hotspot
  // For KERNEL=10: (2*10+1)³ = 21³ = 9,261 iterations per cell
  var potential = 0.0;
  var totalWeight = 0.0;

  // OPTIMIZATION 2: Tight loop bounds using sphere equation (20-30% speedup)
  // Calculate exact bounds per Z/Y layer to skip cells outside sphere
  for (var dz : i32 = -KERNEL; dz <= KERNEL; dz = dz + 1) {
    let dz_f = f32(dz);
    let r2_yz = outerRadius * outerRadius - dz_f * dz_f;

    if (r2_yz < 0.0) {
      continue; // Skip entire Z layer if outside sphere
    }

    let maxR_yz = sqrt(r2_yz);
    let dyMax = i32(ceil(maxR_yz));

    for (var dy : i32 = -dyMax; dy <= dyMax; dy = dy + 1) {
      let dy_f = f32(dy);
      let r2_x = r2_yz - dy_f * dy_f;

      if (r2_x < 0.0) {
        continue; // Skip entire Y row if outside sphere
      }

      let maxR_x = sqrt(r2_x);
      let dxMax = i32(ceil(maxR_x));

      for (var dx : i32 = -dxMax; dx <= dxMax; dx = dx + 1) {
        let offset = vec3<i32>(dx, dy, dz);
        let dist = length(vec3<f32>(offset));

        // Keep safety check
        if (dist <= outerRadius) {
          let neighbor = loadEnergy(coord + offset, dims);
          let w = kernelWeight(dist, innerRadius, innerStrength, outerRadius, outerStrength);
          potential = potential + neighbor * w;
          totalWeight = totalWeight + abs(w);
        }
      }
    }
  }

  if (totalWeight > 0.0) {
    potential = potential / totalWeight;
  }

  // 2. Growth function (Gaussian bell curve)
  let widthEff = max(1e-6, growthWidth * growthWidthNorm);
  var growth = growthFunction(potential, currentEnergy, growthCenter, widthEff, fissionThreshold, instability) - 0.5;
  growth = growth - globalAverage * suppressionFactor;

  // 3. Metabolism (quadratic energy decay)
  let metabolism = currentEnergy * currentEnergy * decayRate;

  // 4. Diffusion (Laplacian smoothing)
  let diffusion = laplacian(coord, dims, currentEnergy, neighborMode) * diffusionRate;

  // 5. Fission instability (chaos at high energy)
  var fissionNoise = 0.0;
  if (currentEnergy > fissionThreshold) {
    let excess = (currentEnergy - fissionThreshold) / (1.0 - fissionThreshold);
    let chaos = sin((f32(coord.x + coord.y + coord.z) + time) * 0.5);
    fissionNoise = chaos * excess * 0.1;
  }

  // 6. Random noise for organic behavior
  let noise = (hash31(gid + vec3<u32>(u32(seed * 100000.0))) - 0.5) * 0.001;

  // Update energy
  let deltaEnergy = growthRate * growth - metabolism + diffusion + fissionNoise + noise;
  var newEnergy = currentEnergy + deltaEnergy;
  newEnergy = clamp(newEnergy, 0.0, 1.0);

  textureStore(outputTex, coord, vec4<f32>(newEnergy, 0.0, 0.0, 1.0));
}
`;
