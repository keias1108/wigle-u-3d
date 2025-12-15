// Default parameters mirror the 2D version (R channel only for now).
export const DEFAULT_PARAMS = {
  innerRadius: 3.5,
  innerStrength: 0.9,
  outerRadius: 7.5,
  outerStrength: -0.4,
  growthCenter: -0.17,
  growthWidth: 0.0183,
  growthRate: 0.607,
  decayRate: 0.378,
  diffusionRate: 0.333,
  fissionThreshold: 0.796,
  suppressionFactor: 1.0,
  instabilityFactor: 1.5,
};

export const PARAM_SPECS = [
  // Dynamic Tension
  { key: 'innerRadius', min: 1.0, max: 10.0, step: 0.1, label: 'Inner Radius' },
  { key: 'innerStrength', min: 0.0, max: 2.0, step: 0.01, label: 'Inner Strength' },
  { key: 'outerRadius', min: 5.0, max: 15.0, step: 0.1, label: 'Outer Radius' },
  { key: 'outerStrength', min: -2.0, max: 0.0, step: 0.01, label: 'Outer Strength' },
  // Energy Economy
  { key: 'decayRate', min: 0.0, max: 1.0, step: 0.001, label: 'Decay Rate' },
  { key: 'diffusionRate', min: 0.0, max: 1.0, step: 0.001, label: 'Diffusion Rate' },
  { key: 'fissionThreshold', min: 0.5, max: 0.95, step: 0.001, label: 'Fission Threshold' },
  // Growth Function
  { key: 'growthCenter', min: -2.0, max: 2.0, step: 0.0001, label: 'Growth Center' },
  { key: 'growthWidth', min: 0.0001, max: 1.0, step: 0.0001, label: 'Growth Width' },
  { key: 'growthRate', min: 0.001, max: 1.0, step: 0.001, label: 'Growth Rate' },
  // Global mod
  { key: 'suppressionFactor', min: 0.0, max: 2.0, step: 0.01, label: 'Suppression Factor' },
  { key: 'instabilityFactor', min: 0.0, max: 3.0, step: 0.01, label: 'Instability Factor' },
];
