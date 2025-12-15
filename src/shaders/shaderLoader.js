/**
 * WGSL Shader Loader for 3D Energy Life Simulation
 *
 * Loads shader code and performs template substitution.
 * This module allows shaders to be maintained in separate .wgsl files (documentation)
 * while the actual source is in .wgsl.js files (for ES module compatibility).
 */

import computeShaderSource from './compute.wgsl.js';
import reduceShaderSource from './reduce.wgsl.js';
import renderShaderSource from './render.wgsl.js';

/**
 * Get compute shader with KERNEL_SIZE and workgroup size template replacements
 *
 * @param {number} kernelSize - The kernel radius (e.g., 10 for 21×21×21 kernel)
 * @param {number} workgroupX - Workgroup size in X dimension
 * @param {number} workgroupY - Workgroup size in Y dimension
 * @param {number} workgroupZ - Workgroup size in Z dimension
 * @returns {string} - WGSL compute shader source code
 *
 * @example
 * const shader = getComputeShader(10, 8, 8, 4);
 * // Returns shader with "const KERNEL : i32 = 10;" and "@compute @workgroup_size(8, 8, 4)"
 */
export function getComputeShader(kernelSize, workgroupX, workgroupY, workgroupZ) {
  return computeShaderSource
    .replace('{{KERNEL_SIZE}}', kernelSize.toString())
    .replace('{{WORKGROUP_X}}', workgroupX.toString())
    .replace('{{WORKGROUP_Y}}', workgroupY.toString())
    .replace('{{WORKGROUP_Z}}', workgroupZ.toString());
}

/**
 * Get reduction shader with workgroup size template replacements
 *
 * @param {number} workgroupX - Workgroup size in X dimension
 * @param {number} workgroupY - Workgroup size in Y dimension
 * @param {number} workgroupZ - Workgroup size in Z dimension
 * @returns {string} - WGSL reduce shader source code
 */
export function getReduceShader(workgroupX, workgroupY, workgroupZ) {
  return reduceShaderSource
    .replace('{{WORKGROUP_X}}', workgroupX.toString())
    .replace('{{WORKGROUP_Y}}', workgroupY.toString())
    .replace('{{WORKGROUP_Z}}', workgroupZ.toString());
}

/**
 * Get render shader for 3D volume ray marching
 *
 * @returns {string} - WGSL render shader source code
 */
export function getRenderShader() {
  return renderShaderSource;
}
