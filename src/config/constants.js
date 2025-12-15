export const GRID_SIZE_PRESETS = [
  { label: '96³ (safe)', value: 96 },
  { label: '128³', value: 128 },
  { label: '256³ (heavy)', value: 256 },
];

export const DEFAULT_GRID_SIZE = 96;

// 3D kernel radius kept small to contain sampling cost while preserving rule shape.
export const KERNEL_SIZE = 3;
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

export const SEED_ENERGY_MAX = 0.05;
