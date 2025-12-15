/**
 * WebGPU Error Handling and Device Loss Recovery
 *
 * Provides graceful handling of GPU device loss and errors.
 */

/**
 * Setup device loss handler
 *
 * WebGPU devices can be lost due to:
 * - GPU driver crash
 * - System sleep/wake
 * - GPU removal (external GPU)
 * - Timeout (long-running compute)
 *
 * @param {GPUDevice} device - WebGPU device
 * @param {Function} onLost - Callback when device is lost
 *
 * @example
 * setupDeviceLossHandler(device, (info) => {
 *   console.error('GPU lost:', info.message);
 *   // Attempt recovery or show error to user
 * });
 */
export function setupDeviceLossHandler(device, onLost) {
  device.lost.then((info) => {
    console.error(`WebGPU device was lost: ${info.message}`);
    console.error('Reason:', info.reason);

    // Notify callback
    if (onLost) {
      onLost(info);
    }

    // Suggest actions based on reason
    if (info.reason === 'destroyed') {
      console.log('Device was intentionally destroyed');
    } else {
      console.warn('Unexpected device loss - may need to reload page');
    }
  });
}

/**
 * Attempt to recover from device loss
 *
 * Creates a new device and reinitializes the simulation.
 * This is a best-effort recovery - not all cases can be handled.
 *
 * @param {GPUAdapter} adapter - Original WebGPU adapter
 * @param {WebGPUSimulation3D} simulation - Simulation instance
 * @returns {Promise<GPUDevice|null>} New device or null if recovery failed
 *
 * @example
 * const newDevice = await recoverFromDeviceLoss(adapter, simulation);
 * if (newDevice) {
 *   console.log('Successfully recovered GPU device');
 * } else {
 *   alert('Failed to recover - please reload the page');
 * }
 */
export async function recoverFromDeviceLoss(adapter, simulation) {
  try {
    console.log('Attempting to request new WebGPU device...');
    const newDevice = await adapter.requestDevice();

    if (!newDevice) {
      console.error('Failed to obtain new device');
      return null;
    }

    console.log('New device obtained, reinitializing simulation...');

    // Note: This requires simulation class to support reinitialization
    // Current WebGPUSimulation3D doesn't have this method yet
    // Future enhancement: Add simulation.reinitialize(newDevice)

    return newDevice;
  } catch (error) {
    console.error('Device recovery failed:', error);
    return null;
  }
}

/**
 * Setup uncaught error handler for WebGPU
 *
 * Catches validation errors and out-of-memory errors.
 *
 * @param {GPUDevice} device - WebGPU device
 */
export function setupUncaughtErrorHandler(device) {
  device.addEventListener('uncaughterror', (event) => {
    console.error('Uncaught WebGPU error:', event.error);

    // Differentiate error types
    if (event.error instanceof GPUValidationError) {
      console.error('Validation Error:', event.error.message);
      // Validation errors usually indicate bugs in shader or pipeline setup
    } else if (event.error instanceof GPUOutOfMemoryError) {
      console.error('Out of Memory - try reducing grid size');
      // Suggest user action
      alert('GPU ran out of memory. Try reducing the grid size to 32³ or 64³.');
    } else {
      console.error('Unknown GPU error:', event.error);
    }
  });
}

/**
 * Check WebGPU feature support
 *
 * @param {GPUAdapter} adapter - WebGPU adapter
 * @returns {Object} Feature support status
 */
export function checkFeatureSupport(adapter) {
  const features = {
    float32Filterable: adapter.features.has('float32-filterable'),
    timestamp: adapter.features.has('timestamp-query'),
    indirectFirstInstance: adapter.features.has('indirect-first-instance'),
  };

  console.log('WebGPU features:', features);

  // float32-filterable is needed for linear sampling of float textures
  // We use rgba16float which is always filterable, so this is optional
  if (!features.float32Filterable) {
    console.warn('float32-filterable not supported (OK - using rgba16float)');
  }

  return features;
}

/**
 * Check WebGPU limits
 *
 * @param {GPUAdapter} adapter - WebGPU adapter
 * @returns {Object} Relevant limits
 */
export function checkLimits(adapter) {
  const limits = {
    maxTextureDimension3D: adapter.limits.maxTextureDimension3D,
    maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    maxStorageTexturesPerShaderStage: adapter.limits.maxStorageTexturesPerShaderStage,
  };

  console.log('WebGPU limits:', limits);

  // Validate our requirements
  if (limits.maxTextureDimension3D < 256) {
    console.warn(`Max 3D texture size is ${limits.maxTextureDimension3D} (need 256 for max grid)`);
  }

  if (limits.maxComputeInvocationsPerWorkgroup < 256) {
    console.warn('GPU may not support larger workgroup sizes for optimization');
  }

  return limits;
}
