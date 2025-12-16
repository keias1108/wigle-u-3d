/**
 * Uniform Buffer Utilities for WebGPU
 *
 * Provides type-safe buffer packing for GPU uniform buffers.
 * WGSL structs require 16-byte alignment for vec4 types.
 */

import { KERNEL_SIZE, CFL_SCALES } from '../config/constants.js';

/**
 * Helper class for building uniform buffers with proper alignment
 *
 * Usage:
 * ```js
 * const builder = new UniformBufferBuilder(128);
 * builder.writeVec4u([size, size, size, 0]);
 * builder.writeVec4f([param1, param2, param3, param4]);
 * const buffer = builder.getBuffer();
 * device.queue.writeBuffer(uniformBuffer, 0, buffer);
 * ```
 */
export class UniformBufferBuilder {
  /**
   * @param {number} size - Total buffer size in bytes
   */
  constructor(size) {
    this.buffer = new ArrayBuffer(size);
    this.u32View = new Uint32Array(this.buffer);
    this.f32View = new Float32Array(this.buffer);
    this.offset = 0; // Current offset in 32-bit words (4 bytes each)
  }

  /**
   * Write a vec4<u32> to the buffer
   * @param {number[]} values - Array of 4 unsigned integers
   */
  writeVec4u(values) {
    if (values.length !== 4) {
      throw new Error('vec4 requires exactly 4 values');
    }
    this.u32View[this.offset] = values[0];
    this.u32View[this.offset + 1] = values[1];
    this.u32View[this.offset + 2] = values[2];
    this.u32View[this.offset + 3] = values[3];
    this.offset += 4; // vec4 is 16 bytes = 4 words
  }

  /**
   * Write a vec4<f32> to the buffer
   * @param {number[]} values - Array of 4 floats
   */
  writeVec4f(values) {
    if (values.length !== 4) {
      throw new Error('vec4 requires exactly 4 values');
    }
    this.f32View[this.offset] = values[0];
    this.f32View[this.offset + 1] = values[1];
    this.f32View[this.offset + 2] = values[2];
    this.f32View[this.offset + 3] = values[3];
    this.offset += 4; // vec4 is 16 bytes = 4 words
  }

  /**
   * Get the packed buffer
   * @returns {ArrayBuffer}
   */
  getBuffer() {
    return this.buffer;
  }

  /**
   * Get current offset (for debugging)
   * @returns {number} Offset in bytes
   */
  getOffsetBytes() {
    return this.offset * 4;
  }
}

const GROWTH_WIDTH_NEFF_TARGET = 30.0;
const MIN_GROWTH_WIDTH_NORM = 0.01;
const MAX_GROWTH_WIDTH_NORM = 10.0;

function kernelWeight(dist, params) {
  let weight = 0.0;
  if (dist < params.innerRadius) {
    const t = 1.0 - dist / params.innerRadius;
    weight += params.innerStrength * t * t;
  }
  const ringStart = params.innerRadius + 1.0;
  const ringEnd = params.outerRadius;
  if (dist > ringStart && dist < ringEnd) {
    const t = (dist - ringStart) / (ringEnd - ringStart);
    weight += params.outerStrength * Math.exp(-2.0 * t * t);
  }
  return weight;
}

function computeKernelNeff3D(params) {
  let sumAbs = 0.0;
  let sumW2 = 0.0;
  const outerRadius = params.outerRadius;
  const kernel = KERNEL_SIZE;

  for (let dz = -kernel; dz <= kernel; dz++) {
    for (let dy = -kernel; dy <= kernel; dy++) {
      for (let dx = -kernel; dx <= kernel; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > outerRadius) continue;
        const w = kernelWeight(dist, params);
        const absW = Math.abs(w);
        sumAbs += absW;
        sumW2 += w * w;
      }
    }
  }

  if (sumW2 <= 1e-12) return 1.0;
  return (sumAbs * sumAbs) / sumW2;
}

function computeGrowthWidthNorm(params) {
  const strength = typeof params.growthWidthNorm === 'number' ? params.growthWidthNorm : 0.0;
  if (strength <= 0.0) return 1.0;

  const neff = computeKernelNeff3D(params);
  const rawScale = Math.sqrt(GROWTH_WIDTH_NEFF_TARGET / Math.max(1.0, neff));
  const clamped = Math.max(MIN_GROWTH_WIDTH_NORM, Math.min(MAX_GROWTH_WIDTH_NORM, rawScale));
  return Math.pow(clamped, strength);
}

/**
 * Pack simulation parameters into a uniform buffer
 *
 * Corresponds to WGSL struct:
 * ```wgsl
 * struct SimParams {
 *   dims : vec4<u32>,
 *   inner : vec4<f32>,
 *   growthA : vec4<f32>,
 *   economy : vec4<f32>,
 *   instab : vec4<f32>,
 *   misc : vec4<f32>,
 *   camera : vec4<f32>,
 * };
 * ```
 *
 * @param {Object} params - Simulation parameters
 * @param {number} gridSize - Grid dimension (e.g., 32 for 32×32×32)
 * @param {Object} camera - Camera parameters {yaw, pitch, distance, offsetX, offsetY}
 * @returns {ArrayBuffer} Packed buffer ready for GPU upload
 */
export function packSimParams(params, gridSize, camera) {
  const builder = new UniformBufferBuilder(128); // 7 vec4s × 16 bytes = 112, rounded to 128

  const inv = 1.0 / gridSize;
  const growthWidthNorm = computeGrowthWidthNorm(params);

  // dims vec4<u32>
  builder.writeVec4u([gridSize, gridSize, gridSize, 0]);

  // inner vec4<f32> (innerRadius, innerStrength, outerRadius, outerStrength)
  builder.writeVec4f([
    params.innerRadius,
    params.innerStrength,
    params.outerRadius,
    params.outerStrength,
  ]);

  // growthA vec4<f32> (growthCenter, growthWidth, growthRate, suppressionFactor)
  builder.writeVec4f([
    params.growthCenter,
    params.growthWidth,
    params.growthRate,
    params.suppressionFactor,
  ]);

  // economy vec4<f32> (globalAverage, decayRate, diffusionRate, fissionThreshold)
  // CFL scale depends on neighbor mode (6/18/26)
  const cflScale = CFL_SCALES[params.neighborMode] || CFL_SCALES[6];
  builder.writeVec4f([
    params.globalAverage || 0.0,
    params.decayRate,
    params.diffusionRate * cflScale,
    params.fissionThreshold,
  ]);

  // instab vec4<f32> (instabilityFactor, growthWidthNorm, neighborMode, raySteps)
  builder.writeVec4f([
    params.instabilityFactor,
    growthWidthNorm,
    params.neighborMode || 6,
    params.raySteps || 96,
  ]);

  // misc vec4<f32> (yaw, pitch, distance, seed)
  builder.writeVec4f([
    (camera.yaw * Math.PI) / 180,
    (camera.pitch * Math.PI) / 180,
    camera.distance,
    Math.random(), // seed
  ]);

  // camera vec4<f32> (offsetX, offsetY, time, packed)
  // Pack both paletteMode (bits 0-1) and energyRangeFilters (bits 2-5)
  const filterBits = (params.energyRangeFilters || 0b1111) & 0xF;
  const packedValue = (params.paletteMode || 0) | (filterBits << 2);

  builder.writeVec4f([
    camera.offsetX,
    camera.offsetY,
    performance.now() * 0.001,
    packedValue,
  ]);

  return builder.getBuffer();
}
