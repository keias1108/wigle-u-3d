import {
  GRID_SIZE_PRESETS,
  DEFAULT_GRID_SIZE,
  SPEED_OPTIONS,
  INITIAL_YAW,
  INITIAL_PITCH,
  INITIAL_DISTANCE,
} from './config/constants.js';
import { DEFAULT_PARAMS, PARAM_SPECS } from './config/defaults.js';
import { WebGPUSimulation3D } from './core/WebGPUSimulation3D.js';

const canvas = document.getElementById('canvas');
const paramsContainer = document.getElementById('params');
const gridSizeSelect = document.getElementById('gridSize');
const seedBtn = document.getElementById('seedBtn');
const paletteSelect = document.getElementById('palette');
const rayStepsSelect = document.getElementById('raySteps');
const neighborModeSelect = document.getElementById('neighborMode');
const yawInput = document.getElementById('yaw');
const yawLabel = document.getElementById('yawLabel');
const pitchInput = document.getElementById('pitch');
const pitchLabel = document.getElementById('pitchLabel');
const saveBtn = document.getElementById('saveBtn');
const loadInput = document.getElementById('loadInput');
const fpsLabel = document.getElementById('fps');

const speedButtons = Array.from(document.querySelectorAll('.speed-buttons button'));
let lastNonZeroSpeed = 1;

const sim = new WebGPUSimulation3D({
  canvas,
  initialParams: DEFAULT_PARAMS,
  gridSize: DEFAULT_GRID_SIZE,
  yaw: INITIAL_YAW,
  pitch: INITIAL_PITCH,
  distance: INITIAL_DISTANCE,
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

  const makeControl = (spec) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'param';
    const label = document.createElement('label');
    label.textContent = spec.label;
    const valueSpan = document.createElement('span');
    valueSpan.className = 'value';
    valueSpan.id = `value-${spec.key}`;
    label.appendChild(valueSpan);

    const row = document.createElement('div');
    row.className = 'row';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = spec.min;
    slider.max = spec.max;
    slider.step = spec.step;
    slider.value = DEFAULT_PARAMS[spec.key];
    slider.dataset.key = spec.key;

    const number = document.createElement('input');
    number.type = 'number';
    number.min = spec.min;
    number.max = spec.max;
    number.step = spec.step;
    number.value = DEFAULT_PARAMS[spec.key];
    number.dataset.key = spec.key;

    const onChange = (val, key) => {
      sim.updateParam(key, val);
      slider.value = val;
      number.value = val;
      updateValueLabel(key, val);
    };

    slider.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -1 : 1;
        const baseStep = parseFloat(spec.step);
        const step =
          spec.key === 'growthWidth' && baseStep < 0.001 ? 0.001 : baseStep;
        const next = parseFloat(slider.value) + delta * step;
        onChange(next, spec.key);
      },
      { passive: false },
    );
    slider.addEventListener('input', (e) => {
      const key = e.target.dataset.key;
      const value = parseFloat(e.target.value);
      onChange(value, key);
    });

    number.addEventListener('input', (e) => {
      const key = e.target.dataset.key;
      const value = parseFloat(e.target.value);
      if (Number.isNaN(value)) return;
      onChange(value, key);
    });

    row.appendChild(slider);
    row.appendChild(number);
    wrapper.appendChild(label);
    wrapper.appendChild(row);
    paramsContainer.appendChild(wrapper);
  };

  // Group controls like wigle-u
  const tension = PARAM_SPECS.slice(0, 4);
  const economy = PARAM_SPECS.slice(4, 7);
  const growth = PARAM_SPECS.slice(7, 10);
  const globalMod = PARAM_SPECS.slice(10);

  const addGroup = (title, specs) => {
    const header = document.createElement('h3');
    header.textContent = title;
    header.style.margin = '12px 0 4px';
    paramsContainer.appendChild(header);
    specs.forEach(makeControl);
  };

  addGroup('Dynamic Tension', tension);
  addGroup('Energy Economy', economy);
  addGroup('Growth Function', growth);
  addGroup('Global', globalMod);

  speedButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const speed = Number(btn.dataset.speed);
      sim.setSpeed(speed);
      if (speed > 0) lastNonZeroSpeed = speed;
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

  paletteSelect.addEventListener('change', (e) => {
    const mode = Number(e.target.value);
    sim.updateParam('paletteMode', mode);
    updateValueLabel('paletteMode', mode);
  });

  rayStepsSelect.addEventListener('change', (e) => {
    const steps = Number(e.target.value);
    sim.updateParam('raySteps', steps);
  });

  neighborModeSelect.addEventListener('change', (e) => {
    const mode = Number(e.target.value);
    sim.updateParam('neighborMode', mode);
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

  // Canvas interactions: wheel zoom, drag rotate
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      // 반대 방향 줌
      sim.adjustDistance(e.deltaY * 0.001);
    },
    { passive: false },
  );
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return; // 좌클릭 회전
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener('mouseup', () => {
    dragging = false;
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    sim.adjustRotation(dx, dy); // 지구본 스타일: 드래그 방향대로 회전
    yawInput.value = sim.yaw.toFixed(0);
    pitchInput.value = sim.pitch.toFixed(0);
    updateYaw(sim.yaw);
    updatePitch(sim.pitch);
  });

  // WASD pan
  const keyMap = { KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd' };
  window.addEventListener('keydown', (e) => {
    const k = keyMap[e.code];
    if (k) sim.setKeyState(k, true);
    if (e.code === 'Space') {
      e.preventDefault();
      if (sim.speed > 0) {
        sim.setSpeed(0);
      } else {
        sim.setSpeed(lastNonZeroSpeed || 1);
      }
      highlightSpeed(sim.speed);
    }
    if (!['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
      const speedMap = { Digit0: 0, Digit1: 1, Digit2: 2, Digit5: 5 };
      if (e.code in speedMap) {
        const s = speedMap[e.code];
        sim.setSpeed(s);
        if (s > 0) lastNonZeroSpeed = s;
        highlightSpeed(s);
      }
    }
  });
  window.addEventListener('keyup', (e) => {
    const k = keyMap[e.code];
    if (k) sim.setKeyState(k, false);
  });

  const highlightSpeed = (speed) => {
    speedButtons.forEach((b) => (b.style.background = '#1b2636'));
    const btn = speedButtons.find((b) => Number(b.dataset.speed) === speed);
    if (btn) btn.style.background = '#274064';
  };

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
        const num = Number(v);
        const value = Number.isNaN(num) ? v : num;
        sim.updateParam(k, value);
        paramsContainer.querySelectorAll(`input[data-key="${k}"]`).forEach((input) => {
          input.value = value;
        });
        updateValueLabel(k, value);
      });
    }
    if (parsed.gridSize) {
      gridSizeSelect.value = parsed.gridSize;
      await sim.resizeGrid(parsed.gridSize);
    }
    if (typeof parsed.paletteMode === 'number') {
      sim.updateParam('paletteMode', parsed.paletteMode);
      paletteSelect.value = parsed.paletteMode;
      updateValueLabel('paletteMode', parsed.paletteMode);
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
  if (paletteSelect) {
    paletteSelect.value = params.paletteMode ?? 0;
  }
  if (rayStepsSelect) {
    rayStepsSelect.value = params.raySteps ?? 96;
  }
  if (neighborModeSelect) {
    neighborModeSelect.value = params.neighborMode ?? 6;
  }
}
