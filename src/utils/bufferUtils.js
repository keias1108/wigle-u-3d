/**
 * Uniform Buffer Utilities for WebGPU
 *
 * Provides type-safe buffer packing for GPU uniform buffers.
 * WGSL structs require 16-byte alignment for vec4 types.
 */

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
  builder.writeVec4f([
    params.globalAverage || 0.0,
    params.decayRate,
    params.diffusionRate,
    params.fissionThreshold,
  ]);

  // instab vec4<f32> (instabilityFactor, inv, inv, inv)
  builder.writeVec4f([params.instabilityFactor, inv, inv, inv]);

  // misc vec4<f32> (yaw, pitch, distance, seed)
  builder.writeVec4f([
    (camera.yaw * Math.PI) / 180,
    (camera.pitch * Math.PI) / 180,
    camera.distance,
    Math.random(), // seed
  ]);

  // camera vec4<f32> (offsetX, offsetY, time, unused)
  builder.writeVec4f([camera.offsetX, camera.offsetY, performance.now() * 0.001, 0.0]);

  return builder.getBuffer();
}
