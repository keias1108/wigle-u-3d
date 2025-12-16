/**
 * 3D Volume Rendering Shader
 * Reference WGSL file: render.wgsl (documentation)
 */

export default /* wgsl */ `
// 3D Volume Rendering Shader
//
// This shader performs ray marching through the 3D energy field:
// 1. Generate ray from orbital camera
// 2. Intersect ray with unit cube (AABB)
// 3. March through volume, sampling energy field
// 4. Map maximum energy to color gradient
// 5. Output final pixel color

struct SimParams {
  dims : vec4<u32>,
  inner : vec4<f32>,
  growthA : vec4<f32>,
  economy : vec4<f32>,
  instab : vec4<f32>,
  misc : vec4<f32>,
  camera : vec4<f32>,
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var fieldTex : texture_3d<f32>;
@group(0) @binding(2) var<uniform> params : SimParams;

struct VertexOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

// Vertex shader: fullscreen quad (2 triangles, 6 vertices)
@vertex
fn vs(@builtin(vertex_index) idx : u32) -> VertexOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0)
  );
  var uv = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0)
  );
  var out: VertexOut;
  out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
  out.uv = uv[idx];
  return out;
}

// Unpack palette mode and energy filters from packed value
fn unpackFilters(packed : f32) -> vec2<u32> {
  let packedInt = u32(packed);
  let paletteMode = packedInt & 0x3u;           // bits 0-1
  let filterBits = (packedInt >> 2u) & 0xFu;    // bits 2-5
  return vec2<u32>(paletteMode, filterBits);
}

// Check if energy value is in a visible range
fn isEnergyVisible(energy : f32, filterBits : u32) -> bool {
  if (energy < 0.25) {
    return (filterBits & 0x1u) != 0u;  // bit 0: low
  } else if (energy < 0.5) {
    return (filterBits & 0x2u) != 0u;  // bit 1: mid-low
  } else if (energy < 0.75) {
    return (filterBits & 0x4u) != 0u;  // bit 2: mid-high
  } else {
    return (filterBits & 0x8u) != 0u;  // bit 3: high
  }
}

// Energy to color gradient (purple/blue theme)
// 0.0-0.1: Dark blue (almost black)
// 0.1-0.3: Blue
// 0.3-0.5: Medium blue/purple
// 0.5-0.7: Bright purple
// 0.7-0.85: Very bright purple
// 0.85-1.0: Near white with sparkle effect
fn energyGradient3D(energy : f32) -> vec3<f32> {
  var color: vec3<f32>;
  if (energy < 0.1) {
    color = mix(vec3<f32>(0.01, 0.005, 0.05), vec3<f32>(0.05, 0.02, 0.15), energy * 10.0);
  } else if (energy < 0.3) {
    color = mix(vec3<f32>(0.05, 0.02, 0.15), vec3<f32>(0.12, 0.08, 0.35), (energy - 0.1) * 5.0);
  } else if (energy < 0.5) {
    color = mix(vec3<f32>(0.12, 0.08, 0.35), vec3<f32>(0.25, 0.12, 0.55), (energy - 0.3) * 5.0);
  } else if (energy < 0.7) {
    color = mix(vec3<f32>(0.25, 0.12, 0.55), vec3<f32>(0.45, 0.18, 0.75), (energy - 0.5) * 5.0);
  } else if (energy < 0.85) {
    color = mix(vec3<f32>(0.45, 0.18, 0.75), vec3<f32>(0.72, 0.35, 0.95), (energy - 0.7) * 6.67);
  } else {
    color = mix(vec3<f32>(0.72, 0.35, 0.95), vec3<f32>(0.95, 0.9, 1.0), (energy - 0.85) * 6.67);
    // Sparkle effect at very high energy
    color = color + vec3<f32>(0.08, 0.04, 0.1) * sin(energy * 40.0);
  }
  // Subtle brightening
  color = color + energy * 0.08;
  return color;
}

// Energy to color gradient (2D display.frag equivalent: blue→cyan→green→yellow→white)
fn energyGradient2D(energy : f32) -> vec3<f32> {
  var color: vec3<f32>;
  if (energy < 0.1) {
    color = mix(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.2), energy * 10.0);
  } else if (energy < 0.3) {
    color = mix(vec3<f32>(0.0, 0.0, 0.2), vec3<f32>(0.0, 0.3, 0.8), (energy - 0.1) * 5.0);
  } else if (energy < 0.5) {
    color = mix(vec3<f32>(0.0, 0.3, 0.8), vec3<f32>(0.0, 0.8, 1.0), (energy - 0.3) * 5.0);
  } else if (energy < 0.7) {
    color = mix(vec3<f32>(0.0, 0.8, 1.0), vec3<f32>(0.2, 1.0, 0.3), (energy - 0.5) * 5.0);
  } else if (energy < 0.85) {
    color = mix(vec3<f32>(0.2, 1.0, 0.3), vec3<f32>(1.0, 1.0, 0.0), (energy - 0.7) * 6.67);
  } else {
    color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), (energy - 0.85) * 6.67);
    // Sparkle effect at very high energy
    color = color + vec3<f32>(0.2, 0.2, 0.2) * sin(energy * 50.0);
  }
  // Subtle brightening
  color = color + energy * 0.15;
  return color;
}

// Energy to color gradient (Structure high-contrast: deep navy → low-sat teal → amber → warm white)
fn energyGradientStructure(energy : f32) -> vec3<f32> {
  var color: vec3<f32>;
  if (energy < 0.02) {
    color = vec3<f32>(0.0, 0.0, 0.0);
  } else if (energy < 0.20) {
    color = mix(vec3<f32>(0.03, 0.05, 0.08), vec3<f32>(0.05, 0.08, 0.12), (energy - 0.02) * (1.0 / 0.18));
  } else if (energy < 0.40) {
    color = mix(vec3<f32>(0.10, 0.18, 0.22), vec3<f32>(0.16, 0.32, 0.35), (energy - 0.20) * 5.0);
  } else if (energy < 0.65) {
    color = mix(vec3<f32>(0.16, 0.32, 0.35), vec3<f32>(0.18, 0.36, 0.38), (energy - 0.40) * 4.0);
  } else if (energy < 0.85) {
    color = mix(vec3<f32>(0.65, 0.45, 0.18), vec3<f32>(0.88, 0.72, 0.35), (energy - 0.65) * 5.0);
  } else {
    color = mix(vec3<f32>(0.88, 0.72, 0.35), vec3<f32>(0.95, 0.93, 0.88), (energy - 0.85) * 6.67);
    // Very light sparkle to pick out peaks
    color = color + vec3<f32>(0.06, 0.05, 0.04) * sin(energy * 45.0);
  }
  return color;
}

// Rotate direction vector by yaw (around Y) and pitch (around X)
fn rotateDir(dir : vec3<f32>, yaw : f32, pitch : f32) -> vec3<f32> {
  let cy = cos(yaw);
  let sy = sin(yaw);
  let cp = cos(pitch);
  let sp = sin(pitch);

  // Yaw rotation (around Y axis)
  let x = dir.x * cy + dir.z * sy;
  let z = -dir.x * sy + dir.z * cy;
  let y = dir.y;

  // Pitch rotation (around X axis)
  let x2 = x;
  let y2 = y * cp - z * sp;
  let z2 = y * sp + z * cp;

  return vec3<f32>(x2, y2, z2);
}

// Ray-AABB intersection test
// Returns (tMin, tMax) where ray enters and exits the box
// If tMax < tMin, ray misses the box
fn intersectAabb(ro : vec3<f32>, rd : vec3<f32>, minB : vec3<f32>, maxB : vec3<f32>) -> vec2<f32> {
  let invD = 1.0 / rd;
  let t0s = (minB - ro) * invD;
  let t1s = (maxB - ro) * invD;
  let tsmaller = min(t0s, t1s);
  let tbigger = max(t0s, t1s);
  let tMin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
  let tMax = min(min(tbigger.x, tbigger.y), tbigger.z);
  return vec2<f32>(tMin, tMax);
}

// Fragment shader: ray marching through 3D volume
@fragment
fn fs(in : VertexOut) -> @location(0) vec4<f32> {
  // Convert UV to NDC [-1, 1]
  let uv = in.uv * 2.0 - vec2<f32>(1.0, 1.0);

  // Extract camera parameters
  let yaw = params.misc.x;
  let pitch = params.misc.y;
  let distance = params.misc.z;
  let offset = vec2<f32>(params.camera.x, params.camera.y);

  // Unpack paletteMode and filterBits from camera.w
  let unpacked = unpackFilters(params.camera.w);
  let paletteMode = f32(unpacked.x);
  let filterBits = unpacked.y;

  // Camera setup
  let center = vec3<f32>(0.5 + offset.x, 0.5 + offset.y, 0.5);
  let dirLocal = normalize(vec3<f32>(uv.x, uv.y, 1.0));
  let dir = rotateDir(dirLocal, yaw, pitch);
  let camDir = rotateDir(vec3<f32>(0.0, 0.0, 1.0), yaw, pitch);
  let ro = center - camDir * distance; // Orbit camera outside box

  // Ray-box intersection
  let boundsMin = vec3<f32>(0.0, 0.0, 0.0);
  let boundsMax = vec3<f32>(1.0, 1.0, 1.0);
  let hit = intersectAabb(ro, dir, boundsMin, boundsMax);

  if (hit.y < max(hit.x, 0.0)) {
    // Ray misses box
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  // Ray marching parameters
  let tStart = max(hit.x, 0.0);
  let tEnd = hit.y;
  let steps = i32(params.instab.w); // raySteps from uniform (64/96/128)
  let dt = (tEnd - tStart) / f32(steps);

  // Ray march and find maximum energy
  var t = tStart;
  var maxE = 0.0;
  for (var i: i32 = 0; i < steps; i = i + 1) {
    let pos = fract(ro + dir * t);
    let e = textureSampleLevel(fieldTex, samp, pos, 0.0).x;
    if (e > maxE) { maxE = e; }
    t = t + dt;
  }

  // Apply energy range filter
  if (!isEnergyVisible(maxE, filterBits)) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  // Map energy to color
  // Sharpness remap + black cut to reduce blur and lift high-energy structures
  let eSharp = select(0.0, pow(maxE, 1.8), maxE >= 0.02);
  let color = select(
    energyGradient3D(eSharp),
    select(energyGradient2D(eSharp), energyGradientStructure(eSharp), paletteMode > 1.5),
    paletteMode > 0.5
  );
  return vec4<f32>(color, 1.0);
}
`;
