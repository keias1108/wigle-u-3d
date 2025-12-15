export const GRID_SIZE_PRESETS = [
  { label: '64³ (fast)', value: 64 },
  { label: '96³ (safe)', value: 96 },
  { label: '128³', value: 128 },
  { label: '256³ (heavy)', value: 256 },
];

export const DEFAULT_GRID_SIZE = 64;

// 3D kernel radius kept small to contain sampling cost while preserving rule shape.
export const KERNEL_SIZE = 10;
export const REDUCE_SIZE = 8; // for global average reduction (assumes grid divisible by 8)
export const GLOBAL_AVG_INTERVAL = 2; // frames between reductions

export const SPEED_OPTIONS = [
  { label: '⏸ 0x', value: 0 },
  { label: '▶ 1x', value: 1 },
  { label: '⏩ 2x', value: 2 },
  { label: '⏩⏩ 5x', value: 5 },
];

export const INITIAL_YAW = 0; // degrees
export const INITIAL_PITCH = 0; // degrees
export const INITIAL_DISTANCE = 2.2; // camera distance from center (outside volume)
export const CAMERA_BOUNDS = { min: 1.2, max: 4.0 };
export const PAN_SPEED = 0.35; // units per second for WASD pan
export const ROTATE_SENSITIVITY = 0.004;

export const SEED_ENERGY_MAX = 0.05;
