// Default parameters mirror the 2D version (R channel only for now).
export const DEFAULT_PARAMS = {
  innerRadius: 3.3,
  innerStrength: 0.88,
  outerRadius: 7.5,
  outerStrength: -0.4,
  growthCenter: -0.0606,
  growthWidth: 0.1,
  growthRate: 0.607,
  decayRate: 0.37,
  diffusionRate: 0.589,
  fissionThreshold: 0.888,
  suppressionFactor: 1.0,
  instabilityFactor: 1.5,
  growthWidthNorm: 0.5,
  paletteMode: 0, // 0=3D purple/blue, 1=2D, 2=Structure high-contrast
  neighborMode: 6, // 6=face, 18=face+edge, 26=full cube
  raySteps: 96, // 64=fast, 96=balanced, 128=quality
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
  { key: 'growthWidthNorm', min: 0.0, max: 4.0, step: 0.01, label: '3D Width Norm' },
];
