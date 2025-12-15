import {
  GRID_SIZE_PRESETS,
  DEFAULT_GRID_SIZE,
  SPEED_OPTIONS,
  INITIAL_YAW,
  INITIAL_PITCH,
} from './config/constants.js';
import { DEFAULT_PARAMS, PARAM_SPECS } from './config/defaults.js';
import { WebGPUSimulation3D } from './core/WebGPUSimulation3D.js';

const canvas = document.getElementById('canvas');
const paramsContainer = document.getElementById('params');
const gridSizeSelect = document.getElementById('gridSize');
const seedBtn = document.getElementById('seedBtn');
const yawInput = document.getElementById('yaw');
const yawLabel = document.getElementById('yawLabel');
const pitchInput = document.getElementById('pitch');
const pitchLabel = document.getElementById('pitchLabel');
const saveBtn = document.getElementById('saveBtn');
const loadInput = document.getElementById('loadInput');
const fpsLabel = document.getElementById('fps');

const speedButtons = Array.from(document.querySelectorAll('.speed-buttons button'));

const sim = new WebGPUSimulation3D({
  canvas,
  initialParams: DEFAULT_PARAMS,
  gridSize: DEFAULT_GRID_SIZE,
  yaw: INITIAL_YAW,
  pitch: INITIAL_PITCH,
});

initControls();
await sim.init();
updateUIFromParams(sim.params);

function initControls() {
  GRID_SIZE_PRESETS.forEach((p) => {
    const option = document.createElement('option');
    option.value = p.value;
    option.textContent = p.label;
    if (p.value === DEFAULT_GRID_SIZE) option.selected = true;
    gridSizeSelect.appendChild(option);
  });

  PARAM_SPECS.forEach((spec) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'param';
    const label = document.createElement('label');
    const valueSpan = document.createElement('span');
    valueSpan.className = 'value';
    valueSpan.id = `value-${spec.key}`;
    label.textContent = spec.label;
    label.appendChild(valueSpan);
    const input = document.createElement('input');
    input.type = 'range';
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step;
    input.value = DEFAULT_PARAMS[spec.key];
    input.dataset.key = spec.key;
    input.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -1 : 1;
        const step = parseFloat(spec.step);
        const next = parseFloat(input.value) + delta * step;
        const clamped = Math.min(spec.max, Math.max(spec.min, next));
        input.value = clamped;
        input.dispatchEvent(new Event('input'));
      },
      { passive: false },
    );
    input.addEventListener('input', (e) => {
      const key = e.target.dataset.key;
      const value = parseFloat(e.target.value);
      sim.updateParam(key, value);
      updateValueLabel(key, value);
    });
    wrapper.appendChild(label);
    wrapper.appendChild(input);
    paramsContainer.appendChild(wrapper);
  });

  speedButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const speed = Number(btn.dataset.speed);
      sim.setSpeed(speed);
      speedButtons.forEach((b) => (b.style.background = '#1b2636'));
      btn.style.background = '#274064';
    });
  });
  speedButtons[1].style.background = '#274064';

  gridSizeSelect.addEventListener('change', async (e) => {
    await sim.resizeGrid(Number(e.target.value));
  });

  seedBtn.addEventListener('click', () => {
    sim.reseed();
  });

  const updateYaw = (deg) => {
    yawLabel.textContent = `${deg}°`;
    sim.setRotation(deg, sim.pitch);
  };
  const updatePitch = (deg) => {
    pitchLabel.textContent = `${deg}°`;
    sim.setRotation(sim.yaw, deg);
  };
  yawInput.addEventListener('input', (e) => updateYaw(parseFloat(e.target.value)));
  pitchInput.addEventListener('input', (e) => updatePitch(parseFloat(e.target.value)));
  updateYaw(INITIAL_YAW);
  updatePitch(INITIAL_PITCH);

  saveBtn.addEventListener('click', () => {
    const data = { params: sim.params, gridSize: sim.gridSize };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'wigle-u-3d.json';
    a.click();
    URL.revokeObjectURL(url);
  });

  loadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (parsed.params) {
      Object.entries(parsed.params).forEach(([k, v]) => {
        sim.updateParam(k, v);
        const input = paramsContainer.querySelector(`input[data-key="${k}"]`);
        if (input) input.value = v;
      });
    }
    if (parsed.gridSize) {
      gridSizeSelect.value = parsed.gridSize;
      await sim.resizeGrid(parsed.gridSize);
    }
    updateUIFromParams(sim.params);
  });

  sim.onFps = (fps) => {
    fpsLabel.textContent = fps.toFixed(0);
  };
}

function updateValueLabel(key, value) {
  const el = document.getElementById(`value-${key}`);
  if (el) el.textContent = value.toFixed(4);
}

function updateUIFromParams(params) {
  Object.entries(params).forEach(([k, v]) => updateValueLabel(k, v));
}
