/**
 * 3D Texture Utilities for WebGPU
 *
 * Provides helpers for creating and seeding 3D textures used in the simulation.
 */

/**
 * Create a 3D field texture for the simulation
 *
 * @param {GPUDevice} device - WebGPU device
 * @param {number} size - Grid dimension (e.g., 32 for 32×32×32)
 * @param {string} [format='rgba16float'] - Texture format
 * @returns {GPUTexture} The created 3D texture
 *
 * @example
 * const texture = createFieldTexture(device, 32);
 */
export function createFieldTexture(device, size, format = 'rgba16float') {
  const usage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_DST |
    GPUTextureUsage.RENDER_ATTACHMENT;

  const descriptor = {
    dimension: '3d',
    size: { width: size, height: size, depthOrArrayLayers: size },
    format,
    usage,
  };

  return device.createTexture(descriptor);
}

/**
 * Seed a 3D texture with random energy values
 *
 * Initializes all cells in the texture with random energy between 0 and maxEnergy.
 * Uses rgba16float format with energy in the red channel.
 *
 * @param {GPUDevice} device - WebGPU device
 * @param {GPUTexture} texture - Target texture to seed
 * @param {number} size - Grid dimension
 * @param {number} maxEnergy - Maximum initial energy value
 *
 * @example
 * const texture = createFieldTexture(device, 32);
 * seedTexture(device, texture, 32, 0.05);
 */
export function seedTexture(device, texture, size, maxEnergy) {
  const total = size * size * size * 4; // 4 components (rgba)
  const data = new Float32Array(total);

  // Fill only the red channel with random energy
  for (let i = 0; i < total; i += 4) {
    data[i] = Math.random() * maxEnergy;
    // data[i+1], data[i+2], data[i+3] remain 0
  }

  device.queue.writeTexture(
    { texture },
    data,
    {
      bytesPerRow: size * 4 * 4, // size × 4 channels × 4 bytes per float
      rowsPerImage: size,
    },
    {
      width: size,
      height: size,
      depthOrArrayLayers: size,
    },
  );
}

/**
 * Create a 3D reduction texture for hierarchical average computation
 *
 * @param {GPUDevice} device - WebGPU device
 * @param {number} size - Grid dimension
 * @returns {GPUTexture} The created reduction texture (r32float format)
 */
export function createReduceTexture(device, size) {
  return device.createTexture({
    dimension: '3d',
    size: { width: size, height: size, depthOrArrayLayers: size },
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });
}
